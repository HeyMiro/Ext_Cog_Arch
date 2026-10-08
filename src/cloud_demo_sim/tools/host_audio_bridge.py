#!/usr/bin/env python3
#
#	Hey MiRo - host audio bridge.
#
#	Publishes a microphone on the host computer as a MiRo-format mic topic
#	(default /<robot>/host/mics: std_msgs/Int16MultiArray, 4 channels x 500
#	samples at 20 kHz, channel-major, the mono mic copied to all four
#	channels). The simulator has no microphones, and on the robot an
#	external mic (the HRI'25 hypercardioid) is much better in a noisy room;
#	the hearing node and the stock sound localiser read this topic exactly
#	like sensors/mics.
#
#	--speaker additionally plays /<robot>/control/stream (8 kHz) on the host
#	and emulates /<robot>/sensors/stream [space, total] at 50 Hz, so the
#	"miro" voice backend can be tried in the simulator. It refuses to do so
#	on a real robot (sensors/package simulator flag clear), where it would
#	fight the robot's own speaker feedback.
#
#	PortAudio callbacks only enqueue / dequeue; publishing happens on the
#	main thread (mic) or a feedback thread (sensors/stream).
#
#	usage:
#	  host_audio_bridge.py [--device D] [--gain-db G] [--topic host/mics] [--speaker]
#	  host_audio_bridge.py --wav FILE [--loop]      (paced replay, e.g. for tests)
#	  host_audio_bridge.py --list-devices
#

import argparse
import os
import queue
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CORE = os.path.join(os.path.dirname(HERE), "core")
if CORE not in sys.path:
	sys.path.insert(0, CORE)

import numpy as np

import audio_util


# sensors/package flags bit set by the Gazebo simulator (miro2.constants)
PLATFORM_U_FLAG_SIMULATOR = 1 << 15

# one MiRo mic message every 25 ms
FRAME_PERIOD = float(audio_util.MIC_BLOCK) / audio_util.MIC_RATE



class MonotonicPacer(object):

	"""
	Fixed-rate pacing on the monotonic clock (never ROS time, so sim time
	and Gazebo's real-time factor cannot distort audio). If we fall more
	than max_lag behind we resync instead of bursting to catch up.
	"""

	def __init__(self, period, clock=time.monotonic, sleep=time.sleep, max_lag=0.25):

		self.period = float(period)
		self.clock = clock
		self.sleep = sleep
		self.max_lag = float(max_lag)
		self.next = None
		self.resyncs = 0

	def wait(self):

		now = self.clock()
		if self.next is None:
			self.next = now
		self.next += self.period
		delay = self.next - now
		if delay > 0.0:
			self.sleep(delay)
		elif -delay > self.max_lag:
			self.next = now
			self.resyncs += 1
		return delay



class MicFramer(object):

	"""
	Host audio blocks (any rate, mono or stereo, int16 or float) -> MiRo
	mic payloads (int16, 2000). Resamples to 20 kHz when needed.
	"""

	def __init__(self, rate_in, channels=1, gain_db=0.0, stereo=False):

		self.channels = int(channels)
		self.stereo = bool(stereo) and self.channels >= 2
		self.gain = 10.0 ** (float(gain_db) / 20.0)
		ch = 2 if self.stereo else 1
		self.resampler = None
		if int(rate_in) != audio_util.MIC_RATE:
			self.resampler = audio_util.StreamResampler(rate_in, audio_util.MIC_RATE, ch)
		self.chunker = audio_util.FrameChunker(audio_util.MIC_BLOCK, ch)

	def push(self, block):

		x = np.asarray(block)
		if x.ndim == 2:
			if self.stereo:
				x = x[:, :2]
			else:
				x = x.astype(np.float32).mean(axis=1)
		if np.issubdtype(x.dtype, np.floating) and x.size and np.max(np.abs(x)) <= 1.0:
			# sounddevice float32 is +-1; MiRo works in int16 units
			x = x * 32768.0
		x = x.astype(np.float32)
		if self.resampler is not None:
			x = self.resampler.process(x)
		return [audio_util.to_miro_frame(f, self.gain) for f in self.chunker.push(x)]



def replay_frames(pcm, rate, gain_db=0.0, loop=False):

	# yield MiRo payloads from a recording, feeding it in 25 ms blocks
	framer = MicFramer(rate, 1, gain_db)
	block = max(1, int(round(rate * FRAME_PERIOD)))
	while True:
		for k in range(0, len(pcm), block):
			for payload in framer.push(pcm[k:k + block]):
				yield payload
		if not loop:
			return



class MicCapture(object):

	# sounddevice input; the callback only copies into a bounded queue
	def __init__(self, sd, device=None, channels=1):

		self.sd = sd
		self.device = device
		self.channels = channels
		self.queue = queue.Queue(maxsize=40)
		self.drops = 0
		self.xruns = 0
		self.stream = None
		self.rate = None

	def open(self):

		# ask for 20 kHz (PulseAudio resamples well); else the device rate
		rates = [audio_util.MIC_RATE]
		try:
			info = self.sd.query_devices(self.device, "input")
			rates.append(int(info["default_samplerate"]))
		except Exception:
			pass
		last = None
		for rate in rates:
			try:
				self.sd.check_input_settings(device=self.device, channels=self.channels,
					samplerate=rate, dtype="int16")
				self.stream = self.sd.InputStream(device=self.device, channels=self.channels,
					samplerate=rate, dtype="int16", blocksize=int(round(rate * FRAME_PERIOD)),
					callback=self._callback)
				self.stream.start()
				self.rate = rate
				return rate
			except Exception as e:
				last = e
		raise RuntimeError("cannot open input device " + str(self.device) + " (" +
			(last.__class__.__name__ if last else "no rate") + ")")

	def _callback(self, indata, frames, time_info, status):

		if status:
			self.xruns += 1
		try:
			self.queue.put_nowait(indata.copy())
		except queue.Full:
			try:
				self.queue.get_nowait()
			except queue.Empty:
				pass
			self.drops += 1
			try:
				self.queue.put_nowait(indata.copy())
			except queue.Full:
				pass

	def close(self):

		if self.stream is not None:
			try:
				self.stream.stop()
				self.stream.close()
			except Exception:
				pass
		self.stream = None



class SpeakerRelay(object):

	"""
	control/stream -> SpeakerBuffer -> host speaker at 8 kHz (or 48 kHz via
	the resampler). Without an output device a "null DAC" thread drains
	the buffer at 8 kHz so senders paced by sensors/stream never stall.
	"""

	def __init__(self, total=8192):

		self.buffer = audio_util.SpeakerBuffer(total)
		self.stream = None
		self._null = None
		self._running = True
		self._resampler = None
		self._warned_oversize = 0

	def on_stream(self, msg):

		# ROS callback: enqueue only
		before = self.buffer.oversize
		self.buffer.write(msg.data)
		if self.buffer.oversize != before and self._warned_oversize < 5:
			self._warned_oversize += 1
			print("[bridge] WARN control/stream message of " + str(len(msg.data)) +
				" samples dropped (MiRo drops messages > " + str(audio_util.STREAM_MSG_MAX) + ")")

	def feedback(self):

		return [self.buffer.space(), self.buffer.total]

	def open(self, sd, device=None):

		for rate in (audio_util.SPKR_RATE, 48000):
			try:
				up = rate // audio_util.SPKR_RATE
				self._resampler = audio_util.StreamResampler(audio_util.SPKR_RATE, rate) if up > 1 else None
				self.stream = sd.OutputStream(device=device, channels=1, samplerate=rate, dtype="int16",
					blocksize=160 * up, callback=self._callback)
				self.stream.start()
				print("[bridge] speaker on " + str(device if device is not None else "(default)") + " at " + str(rate) + " Hz")
				return True
			except Exception as e:
				print("[bridge] speaker at " + str(rate) + " Hz failed (" + e.__class__.__name__ + ")")
		self.stream = None
		return False

	def _callback(self, outdata, frames, time_info, status):

		if self._resampler is None:
			outdata[:, 0] = self.buffer.read(frames)
		else:
			up = frames // 160
			y = audio_util.to_int16(self._resampler.process(self.buffer.read(frames // up)))
			outdata[:, 0] = 0
			outdata[:y.size, 0] = y[:frames]

	def start_null_dac(self):

		def drain():
			pacer = MonotonicPacer(0.02)
			while self._running:
				self.buffer.read(160)
				pacer.wait()
		self._null = threading.Thread(target=drain, name="null_dac")
		self._null.daemon = True
		self._null.start()
		print("[bridge] no speaker: draining control/stream silently at 8 kHz")

	def close(self):

		self._running = False
		if self.stream is not None:
			try:
				self.stream.stop()
				self.stream.close()
			except Exception:
				pass
		self.stream = None



def import_sounddevice():

	try:
		import sounddevice as sd
		return sd
	except (ImportError, OSError) as e:
		print("[bridge] sounddevice unavailable (" + e.__class__.__name__ + "); "
			"install python3-sounddevice / libportaudio2")
		return None


def main(argv=None):

	ap = argparse.ArgumentParser(description="Publish a host microphone as a MiRo mic topic")
	ap.add_argument("--robot", default=os.getenv("MIRO_ROBOT_NAME", "miro"))
	ap.add_argument("--topic", default="host/mics", help="topic under /<robot>/ (default host/mics)")
	ap.add_argument("--device", default=None, help="input device index or name substring")
	ap.add_argument("--channels", type=int, default=1, choices=[1, 2])
	ap.add_argument("--stereo", action="store_true", help="ch0=L ch1=R (robot ext-mic pair only)")
	ap.add_argument("--gain-db", type=float, default=0.0)
	ap.add_argument("--wav", default=None, help="replay this file instead of a microphone")
	ap.add_argument("--loop", action="store_true")
	ap.add_argument("--speaker", action="store_true", help="relay control/stream and emulate sensors/stream")
	ap.add_argument("--output-device", default=None)
	ap.add_argument("--stream-total", type=int, default=8192)
	ap.add_argument("--list-devices", action="store_true")
	args = ap.parse_args(argv)

	device = int(args.device) if args.device is not None and args.device.isdigit() else args.device
	out_device = int(args.output_device) if args.output_device is not None and args.output_device.isdigit() else args.output_device

	if args.list_devices:
		sd = import_sounddevice()
		if sd is None:
			return 2
		print(sd.query_devices())
		return 0

	# ROS (imported here so the helpers above stay importable without ROS)
	import rospy
	from std_msgs.msg import Int16MultiArray, UInt16MultiArray

	base = "/" + args.robot + "/"
	rospy.init_node(args.robot + "_host_audio_bridge", anonymous=False, disable_signals=False)
	if args.topic.strip("/") == "sensors/mics":
		print("[bridge] WARN publishing on sensors/mics: make sure nothing else does")
	pub = rospy.Publisher(base + args.topic.strip("/"), Int16MultiArray, queue_size=10, tcp_nodelay=True)

	# speaker relay (simulator only)
	relay = None
	if args.speaker:
		allowed = True
		try:
			import miro2
			msg = rospy.wait_for_message(base + "sensors/package", miro2.msg.sensors_package, timeout=10.0)
			flags = int(getattr(msg.flags, "data", msg.flags))
			if not (flags & PLATFORM_U_FLAG_SIMULATOR):
				print("[bridge] ERROR --speaker refused: this is a real robot (simulator flag clear)")
				allowed = False
		except Exception as e:
			print("[bridge] WARN no sensors/package seen (" + e.__class__.__name__ + "); assuming simulator")
		if allowed:
			relay = SpeakerRelay(args.stream_total)
			sd = import_sounddevice()
			if sd is None or not relay.open(sd, out_device):
				relay.start_null_dac()
			rospy.Subscriber(base + "control/stream", Int16MultiArray, relay.on_stream, queue_size=20, tcp_nodelay=True)
			pub_fb = rospy.Publisher(base + "sensors/stream", UInt16MultiArray, queue_size=1)

			def feedback():
				pacer = MonotonicPacer(0.02)
				while not rospy.is_shutdown():
					pub_fb.publish(UInt16MultiArray(data=relay.feedback()))
					pacer.wait()
			t = threading.Thread(target=feedback, name="stream_feedback")
			t.daemon = True
			t.start()

	def publish(payload):
		pub.publish(Int16MultiArray(data=payload.tolist()))

	sent = 0
	t_log = time.monotonic()
	try:
		if args.wav:
			pcm, rate = audio_util.load_wav(args.wav)
			print("[bridge] replaying " + args.wav + " (" + str(rate) + " Hz) on " + base + args.topic)
			pacer = MonotonicPacer(FRAME_PERIOD)
			for payload in replay_frames(pcm, rate, args.gain_db, args.loop):
				if rospy.is_shutdown():
					break
				pacer.wait()
				publish(payload)
				sent += 1
		else:
			sd = import_sounddevice()
			if sd is None:
				return 2
			while not rospy.is_shutdown():
				cap = MicCapture(sd, device, args.channels)
				try:
					rate = cap.open()
				except RuntimeError as e:
					# never publish fake silence; retry (hotplug, PulseAudio restart)
					print("[bridge] " + str(e) + "; check PULSE_SERVER / --list-devices; retrying in 5 s")
					time.sleep(5.0)
					continue
				print("[bridge] mic at " + str(rate) + " Hz -> " + base + args.topic)
				framer = MicFramer(rate, args.channels, args.gain_db, args.stereo)
				level = -120.0
				while not rospy.is_shutdown():
					try:
						block = cap.queue.get(timeout=1.0)
					except queue.Empty:
						print("[bridge] mic stalled; reopening")
						break
					for payload in framer.push(block):
						publish(payload)
						sent += 1
						level = audio_util.level_dbfs(payload[:audio_util.MIC_BLOCK])
					if time.monotonic() - t_log > 10.0:
						t_log = time.monotonic()
						print("[bridge] %d frames, %.1f dBFS, drops %d, xruns %d" % (sent, level, cap.drops, cap.xruns))
				cap.close()
	except KeyboardInterrupt:
		pass
	finally:
		if relay is not None:
			relay.close()
	return 0



if __name__ == "__main__":
	sys.exit(main())
