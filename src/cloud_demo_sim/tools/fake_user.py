#!/usr/bin/env python3
#
#	Hey MiRo - a scripted, synthetic user for demos and end-to-end tests.
#
#	Each line of the script is spoken by a Kokoro voice (the local speech
#	service, a different voice from MiRo's) and published on
#	/<robot>/host/mics in exactly MiRo's microphone format, as the host
#	audio bridge would, with quiet background noise in between. The robot
#	side therefore runs its real pipeline on it: VAD, wake words, end of
#	turn, speech-to-text, LLM, text-to-speech.
#
#	Lines are timed from /<robot>/core/forebrain/state: each step waits
#	for a dialogue state (or a task) before speaking, like a person who
#	waits for MiRo to finish. Afterwards a latency table is printed:
#	end of the user's line -> MiRo starts answering (dialogue state
#	"responding", or the requested task starting).
#
#	  python3 tools/fake_user.py                  # the default demo script
#	  python3 tools/fake_user.py --script my.txt  # one step per line: STATE|TEXT
#	  python3 tools/fake_user.py --log run.json
#
#	Use instead of the bridge: ./run_sim.sh --no-bridge, then this.
#

import argparse
import json
import os
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "core"))
sys.path.insert(0, HERE)

import audio_util                          # noqa: E402
from host_audio_bridge import replay_frames, FRAME_PERIOD   # noqa: E402

# (wait for, text): "idle" / "listening" = that dialogue state; "task:X" =
# forebrain task X running (then wait_s more); "sleep:N" = N seconds
DEFAULT_SCRIPT = [
	("idle", "Hey MiRo!"),
	("listening", "Hi! What is your favourite band?"),
	("listening", "Can you spin around for me?"),
	("listening", "Can you dance to Dancing Queen?"),
	("task:dance+12", "Stop MiRo!"),
]


def synthesize(speech_url, text, voice, timeout=30.0):

	import requests
	r = requests.post(speech_url.rstrip("/") + "/audio/speech", timeout=timeout,
		json={"input": text, "voice": voice, "response_format": "pcm", "stream": False, "pitch": 1.0})
	r.raise_for_status()
	return np.frombuffer(r.content[:len(r.content) // 2 * 2], dtype="<i2").copy(), int(r.headers.get("X-Sample-Rate", 24000))


def load_script(path):

	steps = []
	with open(path) as f:
		for line in f:
			line = line.strip()
			if line and not line.startswith("#") and "|" in line:
				cond, text = line.split("|", 1)
				steps.append((cond.strip(), text.strip()))
	return steps


class StateWatch(object):

	def __init__(self):

		self.lock = threading.Lock()
		self.state = {"dialogue": "idle", "task": None}
		self.history = []          # (time, dialogue, task)

	def on_state(self, msg):

		try:
			s = json.loads(msg.data)
		except ValueError:
			return
		with self.lock:
			self.state = s
			self.history.append((time.time(), s.get("dialogue"), s.get("task")))

	def get(self):

		with self.lock:
			return dict(self.state)


def main():

	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--robot", default=os.getenv("MIRO_ROBOT_NAME", "miro"))
	ap.add_argument("--topic", default="host/mics")
	ap.add_argument("--speech-url", default=None, help="default: ai.local.speech_url from the config")
	ap.add_argument("--voice", default="am_michael", help="Kokoro voice of the user")
	ap.add_argument("--script", default=None)
	ap.add_argument("--gain-db", type=float, default=-3.0)
	ap.add_argument("--noise-dbfs", type=float, default=-60.0, help="background noise between lines")
	ap.add_argument("--timeout", type=float, default=60.0, help="max wait per step (s)")
	ap.add_argument("--log", default=None, help="write the timing log (JSON) here")
	ap.add_argument("--play", action="store_true", help="also play each line on this computer's speaker (demos)")
	args = ap.parse_args()

	if args.speech_url is None:
		import heymiro_config
		args.speech_url = heymiro_config.load().cfg.get("ai.local.speech_url", "http://127.0.0.1:8090/v1")
	steps = load_script(args.script) if args.script else DEFAULT_SCRIPT

	print("[fake_user] synthesizing %d lines with voice %s ..." % (len(steps), args.voice))
	audio = [synthesize(args.speech_url, text, args.voice) for _, text in steps]

	import rospy
	from std_msgs.msg import Int16MultiArray, String
	rospy.init_node("heymiro_fake_user", anonymous=True)
	base = "/" + args.robot + "/"
	pub = rospy.Publisher(base + args.topic, Int16MultiArray, queue_size=10)
	watch = StateWatch()
	rospy.Subscriber(base + "core/forebrain/state", String, watch.on_state)

	# the publisher thread: noise, or the current line, every 25 ms
	rng = np.random.default_rng(1)
	noise_amp = 32768.0 * (10.0 ** (args.noise_dbfs / 20.0))
	queue = []
	qlock = threading.Lock()
	done_at = {}

	def publisher():
		t_next = time.monotonic()
		while not rospy.is_shutdown():
			with qlock:
				item = queue[0] if queue else None
			payload = None
			if item is not None:
				idx, frames = item
				try:
					payload = next(frames)
				except StopIteration:
					with qlock:
						queue.pop(0)
					done_at[idx] = time.time()
					continue
			if payload is None:
				noise = (rng.standard_normal(audio_util.MIC_BLOCK) * noise_amp).astype(np.float32)
				payload = audio_util.to_miro_frame(noise)
			pub.publish(Int16MultiArray(data=payload.reshape(-1).tolist() if hasattr(payload, "reshape") else payload))
			t_next += FRAME_PERIOD
			time.sleep(max(0.0, t_next - time.monotonic()))

	threading.Thread(target=publisher, daemon=True).start()
	time.sleep(1.0)

	def wait_for(cond, after):
		# the state must be reached AFTER `after` (the end of our previous
		# line): MiRo may still be "listening" for a moment after we stop
		t0 = time.time()
		extra = 0.0
		if "+" in cond:
			cond, extra = cond.split("+", 1)
			extra = float(extra)
		if cond.startswith("sleep:"):
			time.sleep(float(cond.split(":", 1)[1]))
			return True
		while not rospy.is_shutdown() and time.time() - t0 < args.timeout:
			with watch.lock:
				hist = list(watch.history)
			fresh = [(d, k) for t, d, k in hist if t > after]
			if after == 0.0:
				fresh.append((watch.get().get("dialogue"), watch.get().get("task")))
			if cond.startswith("task:") and any(k == cond.split(":", 1)[1] for d, k in fresh):
				break
			if any(d == cond and not k for d, k in fresh):
				break
			time.sleep(0.05)
		else:
			return False
		# a person waits a moment before answering
		time.sleep(0.6 + extra)
		return True

	log = []
	last_end = 0.0
	for i, ((cond, text), (pcm, rate)) in enumerate(zip(steps, audio)):
		if not wait_for(cond, last_end):
			print("[fake_user] gave up waiting for '%s' (state: %s)" % (cond, watch.get()))
			break
		print("[fake_user] says: " + text)
		if args.play:
			try:
				import sounddevice as sd
				sd.play(audio_util.to_int16(pcm), rate)
			except Exception as e:
				print("[fake_user] cannot play (" + e.__class__.__name__ + ")")
		t_start = time.time()
		with qlock:
			queue.append((i, replay_frames(pcm, rate, args.gain_db)))
		while i not in done_at and not rospy.is_shutdown():
			time.sleep(0.01)
		last_end = done_at.get(i) or time.time()
		log.append({"text": text, "start": t_start, "end": done_at.get(i)})

	# keep the mic topic alive until MiRo has reacted to the last line
	t_end = time.time()
	while time.time() - t_end < 20.0:
		with watch.lock:
			changed = any(t > last_end + 0.5 for t, d, k in watch.history)
		if changed:
			break
		time.sleep(0.1)
	time.sleep(3.0)

	# latency: end of each line -> next "responding" / task start / thinking
	hist = list(watch.history)
	print()
	print("%-36s %10s %10s" % ("user said", "->thinking", "->answer"))
	for row in log:
		end = row["end"] or row["start"]
		think = next((t for t, d, k in hist if t > end and d == "thinking"), None)
		answer = next((t for t, d, k in hist if t > end and (d in ("responding", "greeting", "goodbye") or k)), None)
		row["to_thinking_s"] = None if think is None else round(think - end, 3)
		row["to_answer_s"] = None if answer is None else round(answer - end, 3)
		print("%-36s %10s %10s" % (row["text"][:36],
			"-" if think is None else "%.2f s" % (think - end),
			"-" if answer is None else "%.2f s" % (answer - end)))
	if args.log:
		with open(args.log, "w") as f:
			json.dump({"steps": log, "states": hist}, f, indent=1)


if __name__ == "__main__":
	main()
