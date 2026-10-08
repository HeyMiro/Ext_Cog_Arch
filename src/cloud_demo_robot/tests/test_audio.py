#
#	Tests for core/audio_util.py, core/forebrain/clips.py and the pure
#	helpers of tools/host_audio_bridge.py (no ROS, no audio devices).
#

import _testpath

import glob
import os
import random
import tempfile
import unittest
import wave

import numpy as np

import audio_util as au
import heymiro_config
from forebrain.clips import ClipLibrary
import host_audio_bridge as bridge



def tone(freq, rate, seconds, amp=10000.0):

	t = np.arange(int(rate * seconds)) / float(rate)
	return amp * np.sin(2.0 * np.pi * freq * t)


def tone_gain_db(y, skip):

	# amplitude of a (steady) sine relative to 10000, from its RMS
	y = np.asarray(y[skip:], dtype=np.float64)
	return 20.0 * np.log10(np.sqrt(2.0) * np.sqrt(np.mean(y * y)) / 10000.0)



class TestResampler(unittest.TestCase):

	def test_ratios(self):

		for a, b, up, down in [(20000, 16000, 4, 5), (24000, 8000, 1, 3), (48000, 20000, 5, 12),
				(44100, 20000, 200, 441), (8000, 48000, 6, 1)]:
			r = au.StreamResampler(a, b)
			self.assertEqual((r.up, r.down), (up, down))

	def test_length_ratio_no_drift(self):

		rng = np.random.RandomState(1)
		for a, b in [(20000, 16000), (44100, 20000), (24000, 8000), (48000, 20000)]:
			r = au.StreamResampler(a, b)
			n_in = 0
			n_out = 0
			while n_in < 60 * a:
				n = int(rng.randint(1, 3000))
				n_out += r.process(np.zeros(n, dtype=np.int16)).size
				n_in += n
				# never more than one sample away from the ideal count
				self.assertLessEqual(abs(n_out - n_in * b / float(a)), 1.0)
			self.assertEqual(n_out, -(-n_in * b // a))

	def test_block_invariance(self):

		rng = np.random.RandomState(2)
		x = rng.randint(-20000, 20000, size=30000).astype(np.int16)
		for a, b in [(20000, 16000), (44100, 20000), (24000, 8000)]:
			one = au.resample(x, a, b)
			r = au.StreamResampler(a, b)
			parts = []
			k = 0
			while k < x.size:
				n = int(rng.randint(1, 1500))
				parts.append(r.process(x[k:k + n]))
				k += n
			blocks = np.concatenate(parts)
			self.assertEqual(blocks.shape, one.shape)
			self.assertLessEqual(np.max(np.abs(blocks - one)), 1.0)

	def test_stereo_matches_mono(self):

		rng = np.random.RandomState(3)
		x = rng.randn(5000, 2) * 5000
		y = au.resample(x, 48000, 20000)
		self.assertEqual(y.shape[1], 2)
		self.assertTrue(np.allclose(y[:, 1], au.resample(x[:, 1], 48000, 20000), atol=1e-2))

	def test_tones(self):

		rate = 20000
		y = au.resample(tone(1000, rate, 1.0), rate, 16000)
		self.assertAlmostEqual(tone_gain_db(y, 1000), 0.0, delta=0.1)
		y = au.resample(tone(9000, rate, 1.0), rate, 16000)
		self.assertLess(tone_gain_db(y, 1000), -40.0)
		# 24 kHz voice to the 8 kHz MiRo speaker: 1 kHz kept, 5 kHz removed
		y = au.resample(tone(1000, 24000, 1.0), 24000, 8000)
		self.assertAlmostEqual(tone_gain_db(y, 500), 0.0, delta=0.1)
		y = au.resample(tone(5000, 24000, 1.0), 24000, 8000)
		self.assertLess(tone_gain_db(y, 500), -40.0)

	def test_passthrough_and_dtype(self):

		x = np.arange(10, dtype=np.int16)
		y = au.StreamResampler(16000, 16000).process(x)
		self.assertEqual(y.dtype, np.float32)
		self.assertTrue(np.array_equal(y, x))
		self.assertEqual(au.resample(x, 20000, 16000).dtype, np.float32)



class TestFrames(unittest.TestCase):

	def test_chunker(self):

		c = au.FrameChunker(512)
		self.assertEqual(c.push(np.arange(500)), [])
		frames = c.push(np.arange(500, 1100))
		self.assertEqual(len(frames), 2)
		self.assertTrue(np.array_equal(frames[0], np.arange(512)))
		self.assertTrue(np.array_equal(frames[1], np.arange(512, 1024)))
		self.assertEqual(c.pending(), 1100 - 1024)

	def test_chunker_multichannel(self):

		c = au.FrameChunker(500, 4)
		frames = c.push(np.zeros((1200, 4)))
		self.assertEqual(len(frames), 2)
		self.assertEqual(frames[0].shape, (500, 4))

	def test_miro_frame_from_msg(self):

		data = list(range(2000))
		f = au.miro_frame_from_msg(data)
		self.assertEqual(f.shape, (4, 500))
		self.assertEqual(f.dtype, np.int16)
		self.assertEqual(f[1, 0], 500)
		self.assertIsNone(au.miro_frame_from_msg(list(range(1999))))
		self.assertIsNone(au.miro_frame_from_msg([]))

	def test_mix_channels_no_overflow(self):

		f = np.zeros((4, 500), dtype=np.int16)
		f[0] = 30000
		f[1] = 30000
		f[2] = -32768
		m = au.mix_channels(f, (0, 1))
		self.assertEqual(m.dtype, np.float32)
		self.assertTrue(np.all(m == 30000))
		self.assertTrue(np.all(au.mix_channels(f, (0, 2)) == -1384))

	def test_to_miro_frame(self):

		mono = np.arange(500) - 250
		out = au.to_miro_frame(mono)
		self.assertEqual(out.shape, (2000,))
		self.assertEqual(out.dtype, np.int16)
		f = au.miro_frame_from_msg(out)
		for ch in range(4):
			self.assertTrue(np.array_equal(f[ch], mono))
		stereo = np.stack([np.full(500, 100), np.full(500, 300)], axis=1)
		f = au.miro_frame_from_msg(au.to_miro_frame(stereo))
		self.assertEqual(list(f[:, 0]), [100, 300, 200, 200])
		# gain saturates instead of wrapping
		f = au.to_miro_frame(np.full(500, 20000), gain=4.0)
		self.assertTrue(np.all(f == 32767))
		self.assertRaises(ValueError, au.to_miro_frame, np.zeros(499))

	def test_levels_and_int16(self):

		self.assertEqual(au.level_dbfs(np.zeros(100)), -120.0)
		self.assertAlmostEqual(au.level_dbfs(np.full(100, 32768.0)), 0.0, places=3)
		self.assertTrue(np.array_equal(au.to_int16([1e9, -1e9, 1.6]), [32767, -32768, 2]))

	def test_wav_roundtrip(self):

		pcm = au.to_int16(tone(440, 16000, 0.1))
		data = au.wav_bytes(pcm, 16000)
		self.assertEqual(data[:4], b"RIFF")
		with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
			f.write(data)
			path = f.name
		try:
			back, rate = au.load_wav(path)
			self.assertEqual(rate, 16000)
			self.assertTrue(np.array_equal(back, pcm))
			# stereo is averaged
			with wave.open(path, "wb") as w:
				w.setnchannels(2)
				w.setsampwidth(2)
				w.setframerate(8000)
				w.writeframes(np.array([100, 300, -100, -300], dtype="<i2").tobytes())
			back, rate = au.load_wav(path)
			self.assertEqual(list(back), [200, -200])
		finally:
			os.remove(path)



class TestSpeakerBuffer(unittest.TestCase):

	def test_buffer(self):

		b = au.SpeakerBuffer(100)
		self.assertEqual(b.space(), 100)
		self.assertEqual(b.write(np.arange(60)), 60)
		self.assertEqual(b.write(np.arange(60)), 40)
		self.assertEqual(b.overflows, 1)
		self.assertEqual(b.occupancy(), 100)
		out = b.read(70)
		self.assertTrue(np.array_equal(out, np.arange(60).tolist() + list(range(10))))
		# ring wrap-around keeps order
		self.assertEqual(b.write(np.arange(1000, 1050)), 50)
		out = b.read(100)
		self.assertTrue(np.array_equal(out[:30], np.arange(10, 40)))
		self.assertTrue(np.array_equal(out[30:80], np.arange(1000, 1050)))
		self.assertTrue(np.all(out[80:] == 0))
		# oversize messages are dropped whole, as on the robot
		self.assertEqual(au.SpeakerBuffer(8192).write(np.zeros(au.STREAM_MSG_MAX + 1)), 0)

	def test_send_count(self):

		self.assertEqual(au.stream_send_count(8192, 8192), 4000)
		self.assertEqual(au.stream_send_count(8192 - 3000, 8192), 1000)
		self.assertEqual(au.stream_send_count(8192 - 5000, 8192), 0)
		self.assertEqual(au.stream_send_count(8192, 8192, stuff=9000), au.STREAM_MSG_MAX)

	def test_pacing_loop(self):

		# stock ActionSpecial back-pressure against an 8 kHz drain (160/tick at 50 Hz)
		b = au.SpeakerBuffer(8192)
		src = np.arange(500 * 160 + 10000) % 1000
		r = 0
		starved = 0
		for tick in range(500):
			space, total = b.space(), b.total
			n = au.stream_send_count(space, total)
			if n:
				r += b.write(src[r:r + n])
			before = b.occupancy()
			b.read(160)
			if tick > 5 and before < 160:
				starved += 1
		self.assertEqual(b.overflows, 0)
		self.assertEqual(b.oversize, 0)
		self.assertEqual(starved, 0)
		self.assertEqual(b.underruns, 0)
		self.assertGreater(b.occupancy(), 3000)



class TestClipLibrary(unittest.TestCase):

	@classmethod
	def setUpClass(cls):

		cls.hm = heymiro_config.load(environ={})
		cls.lib = ClipLibrary.from_config(cls.hm, rng=random.Random(0))

	def test_all_categories_resolve(self):

		self.assertEqual(self.lib.warnings, [])
		for cat in ["greeting", "goodbye", "thinking", "acknowledge", "loading", "pleased",
				"ouch_ear", "ouch_tail", "dance_start"]:
			self.assertTrue(self.lib.has(cat), cat)
			self.assertGreaterEqual(len(self.lib.files(cat)), 1)
		self.assertFalse(self.lib.has("no_such_category"))
		self.assertIsNone(self.lib.pick("no_such_category"))

	def test_every_wav_is_24k_mono(self):

		paths = glob.glob(os.path.join(self.hm.path("assets/clips"), "*", "*.wav"))
		self.assertGreater(len(paths), 20)
		for p in paths:
			with wave.open(p, "rb") as w:
				self.assertEqual((w.getframerate(), w.getnchannels(), w.getsampwidth()), (24000, 1, 2), p)

	def test_pick_no_immediate_repeat(self):

		cat = "thinking"
		self.assertGreater(len(self.lib.files(cat)), 1)
		last = None
		for _ in range(50):
			pcm = self.lib.pick(cat)
			self.assertEqual(pcm.dtype, np.int16)
			self.assertGreater(pcm.size, 1000)
			k = self.lib._last[cat]
			self.assertNotEqual(k, last)
			last = k

	def test_resamples_other_rates(self):

		d = tempfile.mkdtemp()
		os.mkdir(os.path.join(d, "beep"))
		with open(os.path.join(d, "beep", "0.wav"), "wb") as f:
			f.write(au.wav_bytes(au.to_int16(tone(440, 16000, 0.5)), 16000))
		manifest = os.path.join(d, "clips.yaml")
		with open(manifest, "w") as f:
			f.write("beep: [beep]\nempty: [missing_folder]\n")
		lib = ClipLibrary(manifest, d)
		self.assertEqual(lib.pick("beep").size, 12000)
		self.assertFalse(lib.has("empty"))
		self.assertEqual(len(lib.warnings), 1)



class TestBridgeHelpers(unittest.TestCase):

	def test_pacer_with_fake_clock(self):

		now = [100.0]
		slept = []

		def sleep(dt):
			slept.append(dt)
			now[0] += dt

		p = bridge.MonotonicPacer(0.025, clock=lambda: now[0], sleep=sleep)
		for _ in range(40):
			p.wait()
			now[0] += 0.005   # work per frame
		# 40 frames take 40 periods, not more (no drift from the work time)
		self.assertAlmostEqual(now[0], 100.0 + 40 * 0.025 + 0.005, places=6)
		# a long stall resyncs instead of bursting
		now[0] += 2.0
		p.wait()
		self.assertEqual(p.resyncs, 1)
		n = len(slept)
		p.wait()
		self.assertEqual(len(slept), n + 1)
		self.assertAlmostEqual(slept[-1], 0.025, places=6)

	def test_replay_frames(self):

		pcm = au.to_int16(tone(500, 16000, 1.0))
		frames = list(bridge.replay_frames(pcm, 16000))
		# one second at 20 kHz is 40 messages of 4 x 500
		self.assertIn(len(frames), (39, 40))
		for f in frames:
			self.assertEqual(f.shape, (2000,))
			self.assertEqual(f.dtype, np.int16)
		f = au.miro_frame_from_msg(frames[10])
		self.assertTrue(np.array_equal(f[0], f[3]))
		self.assertAlmostEqual(au.level_dbfs(f[0]), au.level_dbfs(pcm[4000:8000]), delta=0.5)
		looped = bridge.replay_frames(pcm, 16000, loop=True)
		self.assertEqual(len([next(looped) for _ in range(100)]), 100)

	def test_mic_framer(self):

		fr = bridge.MicFramer(48000, channels=2, gain_db=6.0)
		x = np.ones((1200, 2), dtype=np.float32) * 0.1
		out = []
		for _ in range(20):
			out += fr.push(x)
		# 24000 input samples at 48 kHz -> 10000 at 20 kHz -> 20 messages
		self.assertIn(len(out), (19, 20))
		f = au.miro_frame_from_msg(out[-1])
		self.assertAlmostEqual(float(f[0, 250]), 0.1 * 32768 * 10 ** (6 / 20.0), delta=20)

	def test_speaker_relay_feedback(self):

		relay = bridge.SpeakerRelay(8192)

		class Msg(object):
			pass

		m = Msg()
		m.data = list(range(1000))
		relay.on_stream(m)
		self.assertEqual(relay.feedback(), [7192, 8192])
		m.data = [0] * 5000
		relay.on_stream(m)
		self.assertEqual(relay.buffer.oversize, 1)
		self.assertEqual(relay.feedback(), [7192, 8192])



if __name__ == "__main__":
	unittest.main()
