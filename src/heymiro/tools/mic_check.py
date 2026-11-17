#!/usr/bin/env python3
#
#	Hey MiRo - microphone check.
#
#	  mic_check.py --list                      audio devices
#	  mic_check.py [--device D]                live level meter (dBFS) of a host mic
#	  mic_check.py --record out.wav --seconds 5  record 20 kHz mono (for --wav replay)
#	  mic_check.py --wakeword [--wav f]        wake words + VAD + utterance segmenter offline,
#	                                           with the keys from the Hey MiRo config
#	  mic_check.py --topic sensors/mics|host/mics   meter a ROS mic topic per channel
#
#	ROS is only needed for --topic; nothing here publishes anything.
#

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CORE = os.path.join(os.path.dirname(HERE), "core")
if CORE not in sys.path:
	sys.path.insert(0, CORE)

import numpy as np

import audio_util



def import_sounddevice():

	try:
		import sounddevice as sd
		return sd
	except (ImportError, OSError) as e:
		print("sounddevice unavailable (" + e.__class__.__name__ + "); install libportaudio2")
		return None


def bar(db, width=40):

	n = int(round((max(-80.0, min(0.0, db)) + 80.0) / 80.0 * width))
	return "#" * n + "." * (width - n)


def mic_blocks(sd, device, rate=audio_util.MIC_RATE, block=audio_util.MIC_BLOCK):

	# blocking reads, mono int16 at 20 kHz (this is a CLI tool, blocking is fine)
	with sd.InputStream(device=device, channels=1, samplerate=rate, dtype="int16", blocksize=block) as s:
		while True:
			data, overflow = s.read(block)
			yield data[:, 0].copy()



def run_meter(sd, device):

	print("level meter (ctrl-c to stop)")
	t = time.monotonic()
	buf = []
	for x in mic_blocks(sd, device):
		buf.append(x)
		if time.monotonic() - t >= 0.25:
			db = audio_util.level_dbfs(np.concatenate(buf))
			print("\r%6.1f dBFS |%s|" % (db, bar(db)), end="")
			sys.stdout.flush()
			buf = []
			t = time.monotonic()


def run_record(sd, device, path, seconds):

	n = int(seconds * audio_util.MIC_RATE)
	print("recording " + str(seconds) + " s to " + path)
	got = []
	total = 0
	for x in mic_blocks(sd, device):
		got.append(x)
		total += x.size
		if total >= n:
			break
	pcm = np.concatenate(got)[:n]
	with open(path, "wb") as f:
		f.write(audio_util.wav_bytes(pcm, audio_util.MIC_RATE))
	print("done, %.1f dBFS" % audio_util.level_dbfs(pcm))


def run_wakeword(sd, device, wav):

	import heymiro_config
	from forebrain.wakeword import KeywordSpotter, make_vad
	from forebrain.segmenter import UtteranceSegmenter

	hm = heymiro_config.load()
	hm.log_warnings()
	cfg = hm.cfg.hearing

	keywords = []
	for w in (cfg.get("wake_words") or []):
		keywords.append({"id": w.get("id"), "path": hm.path(w.get("file")),
			"sensitivity": w.get("sensitivity", 0.6), "access_key": hm.picovoice_key(w.get("access_key_env"))})
	try:
		spotter = KeywordSpotter(keywords)
		print("wake words: " + ", ".join(spotter.ids))
	except RuntimeError as e:
		print("wake words unavailable: " + str(e))
		spotter = None
	vad = make_vad(cfg.get("vad", "energy"), hm.picovoice_key(), cfg.get("energy_threshold_dbfs", -38.0))
	print("VAD: " + getattr(vad, "kind", "?"))
	seg = UtteranceSegmenter(vad_threshold=cfg.get("vad_threshold", 0.6), onset_frames=cfg.get("onset_frames", 3),
		end_silence_s=cfg.get("end_silence_s", 3.0), no_speech_timeout_s=cfg.get("no_speech_timeout_s", 3.0),
		max_utterance_s=cfg.get("max_utterance_s", 20.0), preroll_s=cfg.get("preroll_s", 0.5))

	resampler = audio_util.StreamResampler(audio_util.MIC_RATE, audio_util.HEARING_RATE)
	chunker = audio_util.FrameChunker(audio_util.HEARING_FRAME)

	if wav:
		pcm, rate = audio_util.load_wav(wav)
		pcm = audio_util.to_int16(audio_util.resample(pcm, rate, audio_util.MIC_RATE))
		source = (pcm[k:k + audio_util.MIC_BLOCK] for k in range(0, pcm.size, audio_util.MIC_BLOCK))
	else:
		source = mic_blocks(sd, device)

	# without wake words, keep the segmenter armed so the VAD can still be checked
	if spotter is None:
		print("no wake words: listening for utterances continuously")
		seg.arm()

	n = 0
	try:
		for block in source:
			for frame in chunker.push(audio_util.to_int16(resampler.process(block))):
				t = n * audio_util.HEARING_FRAME / float(audio_util.HEARING_RATE)
				n += 1
				if spotter is not None:
					for kid in spotter.process(frame):
						print("%7.2f s  wake word: %s  (listening for an utterance)" % (t, kid))
						seg.arm()
				prob = vad.process(frame)
				ev = seg.feed(frame, prob)
				if ev is not None:
					extra = ""
					if ev[0] == "utterance":
						extra = " %.1f s" % (ev[1].size / float(audio_util.HEARING_RATE))
					print("%7.2f s  %s%s" % (t, ev[0], extra))
					if spotter is None and not seg.armed:
						seg.arm()
				if n % 16 == 0 and not wav:
					print("\rVAD %.2f %s" % (prob, bar(20.0 * prob - 20.0, 20)), end="")
					sys.stdout.flush()
	finally:
		if spotter is not None:
			spotter.delete()
		vad.delete()


def run_topic(topic):

	import rospy
	from std_msgs.msg import Int16MultiArray

	robot = os.getenv("MIRO_ROBOT_NAME", "miro")
	name = "/" + robot + "/" + topic.strip("/")
	state = {"n": 0, "t": time.monotonic(), "frames": []}

	def callback(msg):
		frame = audio_util.miro_frame_from_msg(msg.data)
		if frame is None:
			print("bad message length " + str(len(msg.data)))
			return
		state["frames"].append(frame)
		state["n"] += 1

	rospy.init_node("mic_check", anonymous=True)
	rospy.Subscriber(name, Int16MultiArray, callback, queue_size=10)
	print("metering " + name + " (L R C T, dBFS)")
	while not rospy.is_shutdown():
		time.sleep(0.5)
		frames, state["frames"] = state["frames"], []
		dt = time.monotonic() - state["t"]
		state["t"] = time.monotonic()
		if not frames:
			print("no messages")
			continue
		x = np.concatenate(frames, axis=1)
		levels = " ".join("%6.1f" % audio_util.level_dbfs(x[ch]) for ch in range(4))
		print("%5.1f msg/s  %s" % (len(frames) / dt, levels))



def main(argv=None):

	ap = argparse.ArgumentParser(description="Hey MiRo microphone check")
	ap.add_argument("--list", action="store_true")
	ap.add_argument("--device", default=None)
	ap.add_argument("--record", default=None, metavar="OUT.wav")
	ap.add_argument("--seconds", type=float, default=5.0)
	ap.add_argument("--wakeword", action="store_true")
	ap.add_argument("--wav", default=None)
	ap.add_argument("--topic", default=None, help="e.g. sensors/mics or host/mics")
	args = ap.parse_args(argv)
	device = int(args.device) if args.device is not None and args.device.isdigit() else args.device

	try:
		if args.topic:
			run_topic(args.topic)
			return 0
		if args.wakeword and args.wav:
			run_wakeword(None, None, args.wav)
			return 0
		sd = import_sounddevice()
		if sd is None:
			return 2
		if args.list:
			print(sd.query_devices())
		elif args.record:
			run_record(sd, device, args.record, args.seconds)
		elif args.wakeword:
			run_wakeword(sd, device, None)
		else:
			run_meter(sd, device)
	except KeyboardInterrupt:
		print("")
	return 0



if __name__ == "__main__":
	sys.exit(main())
