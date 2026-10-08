#
#	Tests for the hearing side (forebrain/segmenter, touch_noise, wakeword,
#	node_hearing) and node_voice, with ROS/miro2 stubs and fake engines.
#

import _testpath
_testpath.install_stubs()

import sys
import time
import types
import unittest

import numpy as np

import audio_util as au
import heymiro_config
from forebrain.segmenter import UtteranceSegmenter
from forebrain.touch_noise import TouchNoiseDetector
from forebrain import wakeword
import node_hearing
import node_voice



FRAME = 512


def frames(n, value=0):

	return [np.full(FRAME, value, dtype=np.int16) for _ in range(n)]


def secs_to_frames(s):

	return int(np.ceil(s * 16000.0 / FRAME))


# the robot copy's hearing settings (MiRo's own mics, touch detection on);
# set explicitly so the same tests pass in the sim copy too
ROBOT_MICS = "hearing.input_topic=sensors/mics;hearing.touch_from_mics=true"


def make_sys(environ=None):

	environ = dict(environ or {})
	environ["HEYMIRO_SET"] = ROBOT_MICS + (";" + environ["HEYMIRO_SET"] if environ.get("HEYMIRO_SET") else "")
	hm = heymiro_config.load(environ=environ)
	return types.SimpleNamespace(
		pars=types.SimpleNamespace(forebrain=hm),
		kc_s=None, kc_m=None,
		input=types.SimpleNamespace(stream=None),
		state=types.SimpleNamespace(in_speaking=0.0),
		output=types.SimpleNamespace(stream=None),
		nodes=types.SimpleNamespace())


class Msg(object):

	def __init__(self, data):
		self.data = data


def scratch(rng, amp=32000):

	# broadband scratch: full-scale white noise (zero crossings ~ half the samples)
	return np.clip(rng.randn(500) * amp, -32768, 32767).astype(np.int16)



class TestSegmenter(unittest.TestCase):

	def feed_all(self, seg, items):

		events = []
		for k, (f, p) in enumerate(items):
			ev = seg.feed(f, p)
			if ev is not None:
				events.append((k, ev))
		return events

	def test_utterance_with_preroll(self):

		seg = UtteranceSegmenter(onset_frames=3, end_silence_s=1.0, preroll_s=0.5)
		pre = [(f, 0.0) for f in frames(20, 1)]
		speech = [(f, 0.9) for f in frames(30, 2)]
		post = [(f, 0.1) for f in frames(40, 3)]
		self.feed_all(seg, pre)         # not armed: ring fills, no events
		seg.arm()
		self.assertTrue(seg.armed)
		events = self.feed_all(seg, speech + post)
		names = [e[1][0] for e in events]
		self.assertEqual(names, ["voiced", "utterance"])
		self.assertEqual(events[0][0], 2)    # third voiced frame
		pcm = events[1][1][1]
		self.assertEqual(pcm.dtype, np.int16)
		n_pre = len(seg._ring)
		# pre-roll (ring holds 16 frames incl. the 3 onset frames) + speech + 0.5 s tail
		self.assertTrue(np.all(pcm[:FRAME] == 1))
		self.assertEqual(np.count_nonzero(pcm == 2), 30 * FRAME)
		self.assertEqual(np.count_nonzero(pcm == 3), 8000)
		self.assertEqual(n_pre, 16)
		# ended after 1.0 s of silence
		self.assertEqual(events[1][0], 30 + secs_to_frames(1.0) - 1)
		self.assertFalse(seg.armed)
		self.assertFalse(seg.capturing)

	def test_no_speech_timeout(self):

		seg = UtteranceSegmenter(no_speech_timeout_s=3.0)
		seg.arm()
		events = self.feed_all(seg, [(f, 0.2) for f in frames(200)])
		self.assertEqual(len(events), 1)
		k, ev = events[0]
		self.assertEqual(ev, ("no_speech", None))
		self.assertEqual(k, secs_to_frames(3.0) - 1)
		self.assertFalse(seg.armed)

	def test_onset_needs_consecutive_frames(self):

		seg = UtteranceSegmenter(onset_frames=3, no_speech_timeout_s=10.0)
		seg.arm()
		items = []
		for _ in range(20):
			items += [(frames(1)[0], 0.9), (frames(1)[0], 0.9), (frames(1)[0], 0.0)]
		self.assertEqual(self.feed_all(seg, items), [])
		self.assertFalse(seg.capturing)

	def test_max_cap(self):

		seg = UtteranceSegmenter(max_utterance_s=2.0, end_silence_s=3.0)
		seg.arm()
		events = self.feed_all(seg, [(f, 1.0) for f in frames(200, 5)])
		self.assertEqual([e[1][0] for e in events], ["voiced", "utterance"])
		self.assertEqual(events[1][1][1].size, 32000)

	def test_voiced_again_after_pause(self):

		seg = UtteranceSegmenter(end_silence_s=3.0, max_utterance_s=60.0)
		seg.arm()
		items = [(f, 1.0) for f in frames(10)] + [(f, 0.0) for f in frames(secs_to_frames(1.0))] + \
			[(f, 1.0) for f in frames(10)] + [(f, 0.0) for f in frames(10)] + [(f, 1.0) for f in frames(5)]
		names = [e[1][0] for e in self.feed_all(seg, items)]
		# a new onset after a 1 s pause, but not after a 0.3 s one
		self.assertEqual(names, ["voiced", "voiced"])
		self.assertTrue(seg.capturing)

	def test_cancel_and_rearm(self):

		seg = UtteranceSegmenter()
		seg.arm()
		self.feed_all(seg, [(f, 1.0) for f in frames(5)])
		self.assertTrue(seg.capturing)
		seg.cancel()
		self.assertFalse(seg.armed)
		self.assertEqual(self.feed_all(seg, [(f, 0.0) for f in frames(200)]), [])
		seg.arm()
		events = self.feed_all(seg, [(f, 0.0) for f in frames(200)])
		self.assertEqual([e[1][0] for e in events], ["no_speech"])

	def test_clear_preroll(self):

		seg = UtteranceSegmenter(onset_frames=3)
		self.feed_all(seg, [(f, 0.0) for f in frames(20, 7)])
		seg.clear_preroll()
		seg.arm()
		events = self.feed_all(seg, [(f, 1.0) for f in frames(3, 1)] + [(f, 0.0) for f in frames(200)])
		pcm = events[-1][1][1]
		self.assertEqual(np.count_nonzero(pcm == 7), 0)



class TestTouchNoise(unittest.TestCase):

	def test_scratch_on_ear_and_tail(self):

		rng = np.random.RandomState(0)
		d = TouchNoiseDetector()
		hits = []
		for _ in range(8):
			f = np.zeros((4, 500), dtype=np.int16)
			f[1] = scratch(rng)
			hits += d.feed(f)
		self.assertEqual(hits, ["ear"] * 6)    # 4000 samples -> 6 blocks of 640
		d = TouchNoiseDetector()
		hits = []
		for _ in range(4):
			f = np.zeros((4, 500), dtype=np.int16)
			f[3] = scratch(rng)
			hits += d.feed(f)
		self.assertEqual(hits, ["tail"] * 3)

	def test_speech_like_and_quiet_noise_ignored(self):

		rng = np.random.RandomState(1)
		d = TouchNoiseDetector()
		hits = []
		t = np.arange(500) / 20000.0
		for k in range(10):
			f = np.zeros((4, 500), dtype=np.int16)
			# loud low-frequency sound: high peak, few zero crossings
			f[0] = (30000 * np.sin(2 * np.pi * 200 * (t + k * 0.025))).astype(np.int16)
			# hiss: many zero crossings, low peak
			f[3] = (rng.randn(500) * 2000).astype(np.int16)
			hits += d.feed(f)
		self.assertEqual(hits, [])

	def test_centre_channel_is_not_an_ear(self):

		rng = np.random.RandomState(2)
		d = TouchNoiseDetector()
		f = np.zeros((4, 500), dtype=np.int16)
		f[2] = scratch(rng)
		self.assertEqual(d.feed(f) + d.feed(f), [])



class FakeEngine(object):

	def __init__(self, access_key, keyword_paths=None, sensitivities=None, results=None):
		self.access_key = access_key
		self.keyword_paths = keyword_paths
		self.sensitivities = sensitivities
		self.frame_length = 512
		self.results = list(results or [])
		self.deleted = False
		self.calls = 0

	def process(self, pcm):
		assert len(pcm) == 512
		self.calls += 1
		return self.results.pop(0) if self.results else -1

	def delete(self):
		self.deleted = True



class TestWakeword(unittest.TestCase):

	def setUp(self):

		self.hm = heymiro_config.load(environ={})
		self.ppn = self.hm.path("assets/wakewords/Hey-Miro_en_linux_v3_0_0.ppn")
		self.created = []

	def create(self, **kw):

		e = FakeEngine(**kw)
		self.created.append(e)
		return e

	def test_groups_by_access_key(self):

		kws = [
			{"id": "hey_miro", "path": self.ppn, "sensitivity": 0.6, "access_key": "k1"},
			{"id": "dance_miro", "path": self.ppn, "sensitivity": 0.5, "access_key": "k2"},
			{"id": "stop_miro", "path": self.ppn, "sensitivity": 0.7, "access_key": "k1"},
			{"id": "missing", "path": "/nonexistent.ppn", "sensitivity": 0.7, "access_key": "k1"},
			{"id": "nokey", "path": self.ppn, "sensitivity": 0.7, "access_key": None},
		]
		ks = wakeword.KeywordSpotter(kws, create=self.create)
		self.assertEqual(len(self.created), 2)
		self.assertEqual(self.created[0].sensitivities, [0.6, 0.7])
		self.assertEqual(ks.ids, ["hey_miro", "stop_miro", "dance_miro"])
		self.created[0].results = [1]
		self.created[1].results = [0]
		self.assertEqual(ks.process(np.zeros(512, dtype=np.int16)), ["stop_miro", "dance_miro"])
		self.assertEqual(ks.process(np.zeros(512, dtype=np.int16)), [])
		ks.delete()
		self.assertTrue(all(e.deleted for e in self.created))

	def test_errors_never_contain_the_key(self):

		def bad_create(**kw):
			raise ValueError("AccessKey 'sekrit-key-123' is invalid")
		kws = [{"id": "hey_miro", "path": self.ppn, "sensitivity": 0.6, "access_key": "sekrit-key-123"}]
		with self.assertRaises(RuntimeError) as cm:
			wakeword.KeywordSpotter(kws, create=bad_create)
		self.assertNotIn("sekrit-key-123", str(cm.exception))
		with self.assertRaises(RuntimeError):
			wakeword.KeywordSpotter([], create=self.create)

	def test_vads(self):

		vad = wakeword.EnergyVad(-38.0)
		rng = np.random.RandomState(0)
		for _ in range(10):
			p_sil = vad.process(np.zeros(512, dtype=np.int16))
		self.assertLess(p_sil, 0.01)
		for _ in range(10):
			p_loud = vad.process((rng.randn(512) * 3000).astype(np.int16))   # about -21 dBFS
		self.assertGreater(p_loud, 0.95)
		self.assertEqual(vad.kind, "energy")

		cobra = wakeword.CobraVad("k", create=lambda access_key: FakeEngine(access_key))
		cobra._handle.process = lambda pcm: 0.75
		self.assertEqual(cobra.process(np.zeros(512)), 0.75)
		cobra.delete()
		# no key -> energy VAD fallback (and never raises)
		self.assertEqual(wakeword.make_vad("cobra", None, -40.0).kind, "energy")
		self.assertEqual(wakeword.make_vad("energy", "k").kind, "energy")



class FakeSpotter(object):

	# says "hey_miro" on the given 512-frame call numbers
	def __init__(self, at=(), ids=("hey_miro", "dance_miro", "stop_miro"), word="hey_miro"):
		self.ids = list(ids)
		self.at = set(at)
		self.word = word
		self.calls = 0
		self.deleted = False

	def process(self, frame):
		self.calls += 1
		return [self.word] if self.calls in self.at else []

	def delete(self):
		self.deleted = True


class LevelVad(object):

	kind = "fake"

	def process(self, frame):
		return 1.0 if au.level_dbfs(frame) > -40.0 else 0.0

	def delete(self):
		pass



class TestNodeHearing(unittest.TestCase):

	def make(self, start=False, spotter=None):

		sys_ = make_sys()
		h = node_hearing.NodeHearing(sys_, spotter=spotter or FakeSpotter(at=(5,)), vad=LevelVad(), start=start)
		self.addCleanup(h.shutdown)
		return h, sys_

	def speech_messages(self, quiet_s=1.0, speech_s=2.0, after_s=4.0):

		t = np.arange(int(speech_s * 20000)) / 20000.0
		x = np.concatenate([np.zeros(int(quiet_s * 20000)), 8000 * np.sin(2 * np.pi * 300 * t),
			np.zeros(int(after_s * 20000))])
		return [Msg(au.to_miro_frame(x[k:k + 500]).tolist()) for k in range(0, x.size, 500)]

	def test_config_wiring(self):

		h, _ = self.make()
		self.assertEqual(h.keyword_ids, ["hey_miro", "dance_miro", "stop_miro"])
		self.assertEqual(h.vad_kind, "fake")
		self.assertIsNotNone(h.touch)    # robot config: touch from sensors/mics
		self.assertEqual(h.channels, [0, 1])

	def test_default_engines_without_keys(self):

		h = node_hearing.NodeHearing(make_sys(), start=False)
		self.addCleanup(h.shutdown)
		self.assertIsNone(h.spotter)
		self.assertEqual(h.vad_kind, "energy")

	def test_worker_end_to_end(self):

		h, sys_ = self.make(start=True)
		h.arm_utterance()
		msgs = self.speech_messages()
		for m in msgs:
			# keep the queue from overflowing, as a 40 Hz topic would
			while h._input.qsize() > 20:
				time.sleep(0.001)
			h.on_mics(m)
		h.on_mics(Msg([0] * 10))    # wrong length: ignored
		deadline = time.time() + 10.0
		while h.frames < len(msgs) and time.time() < deadline:
			time.sleep(0.01)
		time.sleep(0.05)
		events = h.poll_events()
		names = [e[0] for e in events]
		self.assertEqual(h.drops, 0)
		self.assertEqual(h.frames, len(msgs))
		self.assertEqual(names, ["wake", "voiced", "utterance"])
		self.assertEqual(events[0][1], "hey_miro")
		pcm = events[2][1]
		self.assertEqual(pcm.dtype, np.int16)
		# 0.5 s pre-roll (some of it the onset) + 2 s speech + 0.5 s tail, at 16 kHz
		self.assertGreater(pcm.size, int(2.4 * 16000))
		self.assertLess(pcm.size, int(3.2 * 16000))
		self.assertEqual(h.poll_events(), [])

	def test_mask_and_mute(self):

		h, sys_ = self.make(spotter=FakeSpotter(at=range(1, 10000)))
		h.set_keyword_mask({"stop_miro"})
		for m in self.speech_messages(0.2, 0.0, 0.0):
			h.process_frame(au.miro_frame_from_msg(m.data))
		self.assertEqual(h.poll_events(), [])
		h.set_keyword_mask(None)
		for m in self.speech_messages(0.2, 0.0, 0.0):
			h.process_frame(au.miro_frame_from_msg(m.data))
		self.assertIn(("wake", "hey_miro"), h.poll_events())

		# muted while MiRo speaks, and for the guard time after it
		h.arm_utterance()
		sys_.state.in_speaking = 1.0
		h.tick()
		calls = h.spotter.calls
		for m in self.speech_messages(0.0, 1.0, 0.0):
			h.process_frame(au.miro_frame_from_msg(m.data))
		self.assertEqual(h.spotter.calls, calls)
		self.assertEqual(h.poll_events(), [])
		sys_.state.in_speaking = 0.0
		h.tick()
		self.assertTrue(h.muted())
		h._guard_until = 0.0
		self.assertFalse(h.muted())
		h.mute(True)
		self.assertTrue(h.muted())
		h.mute(False)
		for m in self.speech_messages(0.0, 1.0, 4.0):
			h.process_frame(au.miro_frame_from_msg(m.data))
		names = [e[0] for e in h.poll_events() if e[0] != "wake"]
		self.assertEqual(names, ["voiced", "utterance"])

	def test_cancel_and_no_speech(self):

		h, _ = self.make()
		h.arm_utterance()
		h.cancel_utterance()
		for m in self.speech_messages(4.0, 0.0, 0.0):
			h.process_frame(au.miro_frame_from_msg(m.data))
		self.assertEqual([e for e in h.poll_events() if e[0] != "wake"], [])
		h.arm_utterance()
		for m in self.speech_messages(4.0, 0.0, 0.0):
			h.process_frame(au.miro_frame_from_msg(m.data))
		self.assertEqual([e[0] for e in h.poll_events() if e[0] != "wake"], ["no_speech"])

	def test_touch_events(self):

		h, sys_ = self.make()
		rng = np.random.RandomState(3)
		for _ in range(4):
			f = np.zeros((4, 500), dtype=np.int16)
			f[0] = scratch(rng)
			h.process_frame(f)
		touches = [e for e in h.poll_events() if e[0] == "touch"]
		self.assertEqual(touches, [("touch", "ear")] * 3)
		# not while speaking (MiRo's own voice through its body)
		sys_.state.in_speaking = 1.0
		h.tick()
		for _ in range(4):
			f = np.zeros((4, 500), dtype=np.int16)
			f[3] = scratch(rng)
			h.process_frame(f)
		self.assertEqual([e for e in h.poll_events() if e[0] == "touch"], [])

	def test_input_queue_drops_oldest(self):

		h, _ = self.make()
		m = Msg([0] * 2000)
		for _ in range(node_hearing.INPUT_QUEUE_SIZE + 10):
			h.on_mics(m)
		self.assertEqual(h.drops, 10)
		self.assertEqual(h._input.qsize(), node_hearing.INPUT_QUEUE_SIZE)

	def test_shutdown_deletes_engines(self):

		spotter = FakeSpotter()
		h, _ = self.make(start=True, spotter=spotter)
		h.shutdown()
		self.assertTrue(spotter.deleted)
		self.assertFalse(h._thread.is_alive())



class FakeClock(object):

	def __init__(self):
		self.t = 1000.0

	def __call__(self):
		return self.t



class TestNodeVoice(unittest.TestCase):

	def make(self, environ=None):

		sys_ = make_sys(environ)
		clock = FakeClock()
		v = node_voice.NodeVoice(sys_, clock=clock, open_device=False)
		self.addCleanup(v.shutdown)
		return v, sys_, clock

	def run_ticks(self, v, clock, seconds, dt=0.02):

		for _ in range(int(round(seconds / dt))):
			clock.t += dt
			v.tick()

	def test_mixer(self):

		m = node_voice.Mixer(1.0, 0.5)
		m.play("speech", np.full(100, 20000))
		m.play("music", np.full(50, 30000), loop=True)
		out = m.pull(120)
		self.assertTrue(np.all(out[:100] == 32767))     # 20000 + 15000 saturates
		self.assertTrue(np.all(out[100:] == 15000))
		self.assertFalse(m.active("speech"))
		self.assertTrue(m.active("music"))
		self.assertEqual(m.position("music"), 20)       # 120 = 2 loops + 20
		m.stop("music")
		self.assertFalse(m.active())
		self.assertTrue(np.all(m.pull(10) == 0))

	def test_silent_clock_durations(self):

		v, sys_, clock = self.make()
		self.assertTrue(v._silent)
		self.assertEqual(v.play_clip("no_such_category"), 0.0)
		dur = v.play_clip("greeting")
		self.assertGreater(dur, 0.3)
		v.tick()
		self.assertEqual(sys_.state.in_speaking, 1.0)
		self.assertTrue(v.is_speaking())
		self.run_ticks(v, clock, dur - 0.1)
		self.assertTrue(v.is_speaking())
		self.assertAlmostEqual(v.speech_remaining_s(), 0.1, delta=0.03)
		self.run_ticks(v, clock, 0.2)
		self.assertFalse(v.is_speaking())
		self.assertEqual(sys_.state.in_speaking, 0.0)

	def test_music_position_and_stop(self):

		v, sys_, clock = self.make()
		song = np.zeros(24000 * 3, dtype=np.int16)
		self.assertAlmostEqual(v.play_music(song), 3.0)
		v.tick()
		self.run_ticks(v, clock, 1.5)
		self.assertTrue(v.is_music_playing())
		self.assertAlmostEqual(v.music_position_s(), 1.5, delta=0.03)
		self.assertFalse(v.is_speaking())
		v.say_pcm(np.ones(2400, dtype=np.int16))
		self.assertTrue(v.is_speaking())
		v.stop()
		self.assertFalse(v.is_music_playing())
		self.assertFalse(v.is_speaking())
		v.play_music(song[:24000], loop=True)
		self.run_ticks(v, clock, 2.5)
		self.assertTrue(v.is_music_playing())
		v.stop("music")
		self.assertFalse(v.is_music_playing())

	def test_no_sounddevice_falls_back(self):

		saved = sys.modules.get("sounddevice", "absent")
		sys.modules["sounddevice"] = None      # import raises ImportError
		try:
			v = node_voice.NodeVoice(make_sys(), clock=FakeClock(), open_device=True)
			self.assertTrue(v._silent)
			v.shutdown()
		finally:
			if saved == "absent":
				del sys.modules["sounddevice"]
			else:
				sys.modules["sounddevice"] = saved

	def test_miro_backend_pacing(self):

		v, sys_, clock = self.make({"HEYMIRO_SET": "voice.backend=miro"})
		self.assertEqual(v.backend, "miro")
		robot = au.SpeakerBuffer(8192)
		speech = np.full(24000 * 2, 1000, dtype=np.int16)    # 2 s
		v.say_pcm(speech)
		received = 0
		sizes = []
		speaking = []
		for tick in range(200):    # 4 s at 50 Hz
			clock.t += 0.02
			sys_.input.stream = [robot.space(), robot.total]
			v.tick()
			if sys_.output.stream:
				sizes.append(len(sys_.output.stream))
				received += robot.write(sys_.output.stream)
				sys_.output.stream = None
			robot.read(160)    # MiRo plays 8 kHz
			speaking.append(sys_.state.in_speaking)
		self.assertEqual(robot.overflows, 0)
		self.assertEqual(robot.oversize, 0)
		self.assertLessEqual(max(sizes), au.STREAM_MSG_MAX)
		self.assertAlmostEqual(received, 16000, delta=60)
		# speaking covers the robot's buffer, then stops: ~2.0 s + filter/tick slack
		n_speaking = int(sum(speaking))
		self.assertGreaterEqual(n_speaking, 100)
		self.assertLessEqual(n_speaking, 106)
		self.assertFalse(v.is_speaking())

	def test_miro_backend_without_feedback(self):

		v, sys_, clock = self.make({"HEYMIRO_SET": "voice.backend=miro"})
		dur = v.say_pcm(np.zeros(24000, dtype=np.int16))
		self.run_ticks(v, clock, dur + 1.5)
		# no sensors/stream: the silent clock finished the speech anyway
		self.assertFalse(v.is_speaking())
		self.assertIsNone(sys_.output.stream)

	def test_music_position_accounts_for_robot_latency(self):

		v, sys_, clock = self.make({"HEYMIRO_SET": "voice.backend=miro"})
		robot = au.SpeakerBuffer(8192)
		v.play_music(np.zeros(24000 * 5, dtype=np.int16))
		for tick in range(100):
			clock.t += 0.02
			sys_.input.stream = [robot.space(), robot.total]
			v.tick()
			if sys_.output.stream:
				robot.write(sys_.output.stream)
				sys_.output.stream = None
			robot.read(160)
		# 2 s of ticks: what the robot actually played, not what was sent
		self.assertAlmostEqual(v.music_position_s(), 2.0, delta=0.1)



if __name__ == "__main__":
	unittest.main()
