#
#	Tests for core/choreography (moves, lights, dance) and the logic of
#	core/action/action_dance.py. The choreography is pure, so these
#	sweep tempos and times and check the fixes made to the HRI'25 code:
#	beat period 60 / bpm, joint limits, no NaN, beat-locked head bops,
#	8-16 beat sections, seeded determinism, continuous lights,
#	0xAARRGGBB packing, non-mutating min-jerk trajectories and a dance
#	that always ends. ActionDance runs against a fake ForebrainAction
#	base and fake nodes (no ROS, no audio).
#

import _testpath

import importlib
import math
import sys
import types
import unittest

import numpy as np

from choreography import moves
from choreography import lights
from choreography.dance import DanceController, DanceFrame, dance_affect, MOVES
from heymiro_config import Section


BPMS = list(range(60, 181, 10))
EPS = 1e-9


def within_limits(cfg):

	if len(cfg) != 4:
		return False
	for i, x in enumerate(cfg):
		if not math.isfinite(x):
			return False
		if x < moves.LOWER[i] - EPS or x > moves.UPPER[i] + EPS:
			return False
	return True



class TestMoves(unittest.TestCase):

	def test_beat_period(self):

		for bpm in BPMS:
			self.assertAlmostEqual(moves.beat_period(bpm), 60.0 / bpm)
			self.assertAlmostEqual(DanceController(bpm, "pop", 10.0, seed=1).beat_s, 60.0 / bpm)

	def test_beat_period_bad_tempo(self):

		# HRI divided by a tempo of 0 when Spotify had no analysis
		for bpm in (0, -5, None, float("nan"), "abc"):
			self.assertAlmostEqual(moves.beat_period(bpm), 0.5)

	def test_clip_config(self):

		cfg = moves.clip_config([1.0, 5.0, float("nan"), -3.0])
		self.assertTrue(within_limits(cfg))
		self.assertEqual(cfg[moves.LIFT_I], moves.LIFT_MAX)
		self.assertEqual(cfg[moves.YAW_I], moves.NEUTRAL[moves.YAW_I])
		self.assertEqual(cfg[moves.PITCH_I], moves.PITCH_MIN)
		self.assertEqual(cfg[moves.TILT_I], moves.TILT)

	def test_limits_match_contract(self):

		self.assertAlmostEqual(moves.LIFT_MIN, 0.1396, places=3)
		self.assertAlmostEqual(moves.LIFT_MAX, 1.0472, places=3)
		self.assertAlmostEqual(moves.YAW_MAX, 0.9599, places=3)
		self.assertAlmostEqual(moves.PITCH_MIN, -0.3840, places=3)
		self.assertAlmostEqual(moves.PITCH_MAX, 0.1396, places=3)
		self.assertTrue(within_limits(moves.NEUTRAL))

	def test_beat_moves_within_limits(self):

		starts = [moves.neutral(), [0.0, 0.14, 0.9, -0.38], [0.0, 1.04, -0.9, 0.13]]
		fns = [moves.head_bop, moves.head_bang, moves.head_spin, moves.party_nod]
		ranges = {"lift": (0.3, 0.9, 0.5, 1.0), "yaw": (-0.8, 0.8, 0.25, 0.0), "pitch": (-0.3, 0.1, 0.5, 2.0)}
		for bpm in BPMS:
			for t in np.arange(0.0, 120.0, 0.37):
				for start in starts:
					for fn in fns:
						cfg = fn(start, t, bpm)
						self.assertTrue(within_limits(cfg), (fn.__name__, bpm, t, cfg))
					self.assertTrue(within_limits(moves.free_dance(start, t, bpm, ranges)))

	def test_head_bop_tracks_beat(self):

		# pitch minus its mean crosses zero twice per bop, one bop per beat
		dt = 0.001
		for bpm in BPMS:
			beats = 16
			T = 60.0 / bpm
			ts = np.arange(0.0, beats * T, dt) + 0.5 * dt
			p = np.array([moves.head_bop(moves.NEUTRAL, t, bpm)[moves.PITCH_I] for t in ts])
			p = p - p.mean()
			crossings = int(np.sum(np.signbit(p[1:]) != np.signbit(p[:-1])))
			self.assertLessEqual(abs(crossings - 2 * beats), 1, (bpm, crossings))
			# and the head is down (max pitch) on the beat
			self.assertAlmostEqual(moves.head_bop(moves.NEUTRAL, 3 * T, bpm)[moves.PITCH_I], 0.06, places=6)

	def test_nod_no_nan(self):

		# HRI took arcsin() of the pitch: NaN outside the nominal range
		for p0 in (-1.0, -0.384, 0.0, 0.13, 0.5, float("nan")):
			start = [0.0, 0.6, 0.0, p0]
			for t in np.arange(0.0, 3.0, 0.05):
				cfg = moves.nod(start, t)
				self.assertTrue(within_limits(cfg), (p0, t, cfg))
		# a nod starts and ends at the current pitch
		start = [0.0, 0.6, 0.0, -0.1]
		self.assertAlmostEqual(moves.nod(start, 0.0)[moves.PITCH_I], -0.1)
		self.assertAlmostEqual(moves.nod(start, 1.0, period_s=1.0)[moves.PITCH_I], -0.1)
		self.assertAlmostEqual(moves.nod(start, 0.5, period_s=1.0, depth=0.12)[moves.PITCH_I], 0.02)

	def test_poses_within_limits_and_pure(self):

		start = [0.0, 0.6, 0.3, -0.1]
		keep = list(start)
		poses = [moves.look_up(start), moves.look_down(start), moves.lean_in(start), moves.lean_in(start, -1.0),
			moves.tilt(start, 1), moves.tilt(start, -1), moves.light_up(start), moves.light_up(), moves.neutral()]
		for t in np.arange(0.0, 2.0, 0.1):
			poses.append(moves.glance(start, t))
			poses.append(moves.glance(start, t, side=-1.0))
		for p in poses:
			self.assertTrue(within_limits(p), p)
		self.assertEqual(start, keep)
		# directions as in HRI: look_up raises the neck and tips the head up
		self.assertLess(moves.look_up(start)[moves.LIFT_I], start[moves.LIFT_I])
		self.assertLess(moves.look_up(start)[moves.PITCH_I], start[moves.PITCH_I])
		self.assertGreater(moves.look_down(start)[moves.LIFT_I], start[moves.LIFT_I])
		self.assertGreater(moves.look_down(start)[moves.PITCH_I], start[moves.PITCH_I])
		self.assertEqual(moves.light_up(), moves.clip_config([0.0, 0.399, 0.0, -0.21]))
		# neutral() returns a fresh list
		n = moves.neutral()
		n[1] = 99.0
		self.assertNotEqual(moves.neutral()[1], 99.0)
		# glance returns to the start
		self.assertEqual(moves.glance(start, 5.0), moves.clip_config(start))

	def test_ear_flap(self):

		self.assertEqual(moves.ear_flap(0.0), [1.0, 1.0])
		self.assertEqual(moves.ear_flap(0.3), [0.0, 0.0])
		self.assertEqual(moves.ear_flap(0.5), [1.0, 1.0])
		self.assertEqual(moves.ear_flap(0.1, rate_hz=4.0), [1.0, 1.0])
		self.assertEqual(moves.ear_flap(0.15, rate_hz=4.0), [0.0, 0.0])

	def test_trajectory(self):

		start = [0.0, 0.3, 0.5, -0.3]
		target = [0.0, 1.0, -0.5, 0.1]
		s0, t0 = list(start), list(target)
		traj = moves.Trajectory(start, target, 2.0)

		cfg, done = traj.sample(0.0)
		self.assertFalse(done)
		for a, b in zip(cfg[1:], moves.clip_config(start)[1:]):
			self.assertAlmostEqual(a, b)

		cfg, done = traj.sample(2.0)
		self.assertTrue(done)
		self.assertEqual(cfg, moves.clip_config(target))

		cfg, done = traj.sample(1.0)
		self.assertFalse(done)
		self.assertAlmostEqual(cfg[1], 0.65)

		# monotonic, within limits, and never touching the inputs
		last = -1.0
		for t in np.arange(0.0, 2.2, 0.02):
			cfg, _ = traj.sample(t)
			self.assertTrue(within_limits(cfg))
			self.assertGreaterEqual(cfg[1], last - EPS)
			last = cfg[1]
			cfg[1] = 123.0
		self.assertEqual(start, s0)
		self.assertEqual(target, t0)
		start[1] = 0.9
		self.assertAlmostEqual(traj.sample(0.0)[0][1], 0.3)

		# zero duration jumps straight to the target
		cfg, done = moves.Trajectory(start, target, 0.0).sample(0.0)
		self.assertTrue(done)
		self.assertEqual(cfg, moves.clip_config(target))

	def test_cycle_beats(self):

		self.assertEqual(moves.cycle_beats(60, 2, 1.0), 2)
		self.assertEqual(moves.cycle_beats(120, 2, 1.0), 2)
		self.assertEqual(moves.cycle_beats(180, 2, 1.0), 4)



class TestLights(unittest.TestCase):

	def test_pack_rgb(self):

		self.assertEqual(lights.pack_rgb(0x12, 0x34, 0x56), 0xFF123456)
		self.assertEqual(lights.pack_rgb(255, 0, 0), 0xFFFF0000)
		self.assertEqual(lights.pack_rgb(0, 0, 0, 0), 0)
		self.assertEqual(lights.pack_rgb(300, -4, 12.6), 0xFFFF000D)
		self.assertEqual(lights.unpack_rgb(0xFF123456), (0x12, 0x34, 0x56, 0xFF))
		for r, g, b in [(0, 153, 255), (255, 105, 180)]:
			self.assertEqual(lights.pack_rgb(r, g, b), int("0xFF%02x%02x%02x" % (r, g, b), 16))

	def test_palettes_and_emotions(self):

		for genre in ("pop", "disco", "soul", "rock", "dance", "default"):
			self.assertIn(genre, lights.PALETTES)
			self.assertGreaterEqual(len(lights.PALETTES[genre]), 2)
		self.assertEqual(lights.palette("unknown"), lights.PALETTES["default"])
		self.assertEqual(lights.EMOTION_COLOURS["happy"], (0, 255, 64))
		self.assertEqual(lights.EMOTION_COLOURS["sad"], (0, 153, 255))
		self.assertEqual(lights.EMOTION_COLOURS["angry"], (255, 0, 0))
		for e in ("happy", "fine", "angry", "worried", "sad", "scared"):
			self.assertIn(e, lights.EMOTION_COLOURS)

	def test_solid(self):

		illum = lights.solid("red")
		self.assertEqual(illum, [0xFFFF0000] * 6)
		self.assertEqual(lights.solid((0, 255, 64)), [0xFF00FF40] * 6)
		self.assertEqual(lights.solid((200, 100, 50), 0.5), [lights.pack_rgb(100, 50, 25)] * 6)

	def test_transition_continuous(self):

		pairs = [("red", "blue"), ((0, 0, 0), (255, 255, 255)), ((255, 105, 180), (0, 255, 64))]
		for c0, c1 in pairs:
			self.assertEqual(lights.transition(c0, c1, 0.0), list(lights.colour(c0)))
			self.assertEqual(lights.transition(c0, c1, 1.0), list(lights.colour(c1)))
			prev = None
			for x in np.linspace(-0.2, 1.2, 1401):
				rgb = lights.transition(c0, c1, x)
				for v in rgb:
					self.assertTrue(0 <= v <= 255)
				if prev is not None:
					# 1/1000 of the way can move a channel by at most one step
					self.assertLessEqual(max(abs(a - b) for a, b in zip(rgb, prev)), 1)
				prev = rgb

	def test_fade_and_pulse_continuous(self):

		cols = [lights.COLOURS["red"], lights.COLOURS["blue"], lights.COLOURS["yellow"]]
		for fn in (lambda t: lights.fade(cols, t, 120), lights.thinking_pulse):
			prev = None
			for t in np.arange(0.0, 12.0, 0.002):
				illum = fn(t)
				self.assertEqual(len(illum), 6)
				rgb = lights.unpack_rgb(illum[0])
				for v in rgb:
					self.assertTrue(0 <= v <= 255)
				if prev is not None:
					self.assertLessEqual(max(abs(a - b) for a, b in zip(rgb[:3], prev[:3])), 4, t)
				prev = rgb

	def test_flash_on_beats(self):

		cols = [(255, 0, 0), (0, 0, 255)]
		bpm = 100
		T = 0.6
		self.assertEqual(lights.flash(cols, 0.1, bpm), lights.solid(cols[0]))
		self.assertEqual(lights.flash(cols, T + 0.01, bpm), lights.solid(cols[1]))
		self.assertEqual(lights.flash(cols, 2 * T + 0.01, bpm), lights.solid(cols[0]))

	def test_scale(self):

		illum = lights.solid((200, 100, 50))
		self.assertEqual(lights.scale(illum, 0.0), [0xFF000000] * 6)
		self.assertEqual(lights.scale(illum, 1.0), illum)



class TestDance(unittest.TestCase):

	def test_frames_within_limits(self):

		for bpm in BPMS:
			for genre in ("pop", "rock", "soul", "dance", "disco", "default"):
				ctrl = DanceController(bpm, genre, 120.0, seed=bpm)
				for t in np.arange(0.0, 120.0, 0.11):
					f = ctrl.step(t)
					self.assertIsInstance(f, DanceFrame)
					self.assertTrue(within_limits(f.head), (bpm, genre, t, f.head))
					self.assertEqual(len(f.illum), 6)
					for v in f.illum:
						self.assertTrue(0 <= v <= 0xFFFFFFFF)
					for key, val in f.cosmetics.items():
						for x in val:
							self.assertTrue(0.0 <= x <= 1.0 and math.isfinite(x), (key, val))
					if f.wheels is not None:
						for x in f.wheels:
							self.assertTrue(math.isfinite(x) and abs(x) <= 0.5)

	def test_head_continuous(self):

		# moves cross-fade at section changes: no big jumps tick to tick
		for bpm in (60, 120, 180):
			ctrl = DanceController(bpm, "default", 60.0, seed=3)
			prev = ctrl.step(0.0).head
			self.assertLess(max(abs(a - b) for a, b in zip(prev, ctrl.home)), 1e-6)
			for t in np.arange(0.02, 60.0, 0.02):
				h = ctrl.step(t).head
				self.assertLess(max(abs(a - b) for a, b in zip(h, prev)), 0.25, (bpm, t))
				prev = h

	def test_sections(self):

		for bpm in BPMS:
			ctrl = DanceController(bpm, "pop", 90.0, seed=7)
			T = 60.0 / bpm
			self.assertGreater(len(ctrl.sections), 1)
			self.assertAlmostEqual(ctrl.sections[0].start_s, 0.0)
			self.assertGreaterEqual(ctrl.sections[-1].end_s, 90.0)
			for i, s in enumerate(ctrl.sections):
				self.assertTrue(8 <= s.beats <= 16)
				self.assertAlmostEqual(s.end_s - s.start_s, s.beats * T)
				if i > 0:
					self.assertAlmostEqual(s.start_s, ctrl.sections[i - 1].end_s)
				self.assertIn(s.move, MOVES + ("free",))
				self.assertEqual(s.move == "free", i % 2 == 1)
				self.assertEqual(len(s.colours), 2)

	def test_genre_moves(self):

		ctrl = DanceController(120, "soul", 120.0, seed=1)
		for s in ctrl.sections:
			self.assertIn(s.move, ("party", "head_bop", "free"))
		self.assertEqual(DanceController(120, "nonsense", 10.0, seed=1).genre, "default")

	def test_seeded(self):

		a = DanceController(118.6, "pop", 60.0, seed=42)
		b = DanceController(118.6, "pop", 60.0, seed=42)
		self.assertEqual([(s.beats, s.move, s.light_mode, s.colours, s.ranges) for s in a.sections],
			[(s.beats, s.move, s.light_mode, s.colours, s.ranges) for s in b.sections])
		for t in np.arange(0.0, 60.0, 0.5):
			fa, fb = a.step(t), b.step(t)
			self.assertEqual((fa.head, fa.wheels, fa.illum, fa.move), (fb.head, fb.wheels, fb.illum, fb.move))
		plans = set()
		for seed in range(8):
			c = DanceController(118.6, "pop", 60.0, seed=seed)
			plans.add(tuple((s.beats, s.move) for s in c.sections))
		self.assertGreater(len(plans), 1)

	def test_done_and_fade(self):

		ctrl = DanceController(120, "pop", 10.0, seed=1)
		self.assertFalse(ctrl.step(9.99).done)
		self.assertTrue(ctrl.step(10.0).done)
		self.assertTrue(ctrl.step(500.0).done)
		end = ctrl.step(10.0)
		self.assertIsNone(end.wheels)
		self.assertEqual(end.head, ctrl.home)
		# during the last second the head converges on home and the lights dim
		f = ctrl.step(9.999)
		self.assertLess(max(abs(a - b) for a, b in zip(f.head, ctrl.home)), 0.01)
		self.assertIsNone(f.wheels)
		self.assertTrue(all(sum(lights.unpack_rgb(v)[:3]) <= 3 for v in f.illum))
		# zero / bad durations end at once
		self.assertTrue(DanceController(120, "pop", 0.0).step(0.0).done)
		self.assertTrue(DanceController(120, "pop", None).step(0.0).done)
		self.assertTrue(DanceController(0, "pop", 5.0).step(0.0) is not None)

	def test_wheels(self):

		# off when disabled
		ctrl = DanceController(120, "dance", 120.0, seed=5, wheels=False)
		for t in np.arange(0.0, 120.0, 0.1):
			self.assertIsNone(ctrl.step(t).wheels)

		# only in spin / rock sections, constant within each beat, no
		# translation when not allowed
		for allow in (True, False):
			ctrl = DanceController(110, "dance", 120.0, seed=5, allow_translation=allow)
			T = ctrl.beat_s
			moved = set()
			for s in ctrl.sections:
				for k in range(s.beats):
					values = []
					for u in (0.05, 0.5, 0.95):
						t = s.start_s + (k + u) * T
						if t >= ctrl.duration_s - ctrl.fade_s:
							break
						w = ctrl.step(t).wheels
						values.append(None if w is None else tuple(w))
						if w is not None:
							moved.add(s.move)
							self.assertIn(s.move, ("spin", "rock"))
							if not allow:
								self.assertEqual(w[0], 0.0)
					self.assertTrue(all(v == values[0] for v in values), (s, k, values))
			self.assertTrue(moved)

		# rocking cancels out over each bar
		ctrl = DanceController(120, "disco", 120.0, seed=1)
		rock = [s for s in ctrl.sections if s.move == "rock" and s.end_s < 119.0]
		for s in rock:
			dx = dz = 0.0
			for t in np.arange(s.start_s + 0.005, s.end_s, 0.01):
				w = ctrl.step(t).wheels
				dx += w[0] * 0.01
				dz += w[1] * 0.01
			self.assertAlmostEqual(dx, 0.0, places=2)
			self.assertAlmostEqual(dz, 0.0, places=2)

	def test_cosmetics(self):

		f = DanceController(180, "pop", 30.0, seed=1).step(1.0)
		self.assertAlmostEqual(f.cosmetics["eyelids"][0], 0.15)
		f = DanceController(60, "pop", 30.0, seed=1).step(1.0)
		self.assertAlmostEqual(f.cosmetics["eyelids"][0], 0.5)
		f = DanceController(100, "pop", 30.0, seed=1).step(1.0)
		self.assertAlmostEqual(f.cosmetics["eyelids"][0], 0.3)
		self.assertEqual(len(f.cosmetics["ears"]), 2)
		self.assertEqual(len(f.cosmetics["tail"]), 2)

	def test_dance_affect(self):

		v, a_slow = dance_affect(60, "default")
		v2, a_fast = dance_affect(180, "default")
		self.assertGreater(v, 0.5)
		self.assertLess(a_slow, a_fast)
		self.assertLess(dance_affect(120, "soul")[1], dance_affect(120, "pop")[1])
		for bpm in (0, 60, 300):
			for g in ("soul", "pop", "x"):
				v, a = dance_affect(bpm, g)
				self.assertTrue(0.0 <= v <= 1.0 and 0.0 <= a <= 1.0)



# ------------------------------------------------------------- ActionDance

class FakeClock(object):

	def __init__(self):
		self.total = 0

	def start(self, n):
		self.total = n

	def reset(self):
		pass

	def stop(self):
		self.total = 0

	def isActive(self):
		return self.total > 0


class FakeForebrainAction(object):

	# the contract's ForebrainAction surface, recording what it is asked
	def __init__(self, parent):
		self.parent = parent
		self.pars = parent.pars
		self.kc = parent.kc_m
		self.clock = FakeClock()
		self.name = "unnamed"
		self.heads = []
		self.vels = []
		self.base_stops = 0
		self.finalize()

	def finalize(self):
		self.modulate_by_wakefulness = False

	def set_head(self, cfg):
		self.heads.append(list(cfg))
		self.kc.setConfig(moves.clip_config(cfg))

	def body_velocity(self, vx, wz):
		self.vels.append((vx, wz))

	def hold(self):
		self.clock.reset()

	def stop(self):
		self.base_stops += 1
		self.parent.nodes.express.release(self.name)


class FakeKc(object):

	def __init__(self):
		self.cfg = [moves.TILT, 0.9, 0.4, 0.05]

	def getConfig(self):
		return list(self.cfg)

	def setConfig(self, cfg):
		self.cfg = list(cfg)


class FakeForebrain(object):

	def __init__(self):
		self.pending = None
		self.active_task = None
		self.ended = []

	def request_task(self, name, params):
		self.pending = (name, params)

	def pending_task(self):
		return self.pending[0] if self.pending else None

	def claim_task(self, name):
		if self.pending and self.pending[0] == name:
			params = self.pending[1]
			self.pending = None
			self.active_task = name
			return params
		return None

	def end_task(self, name, reason):
		self.ended.append((name, reason))
		if self.active_task == name:
			self.active_task = None


class FakeVoice(object):

	DT = 0.02

	def __init__(self, clip_s=0.3):
		self.clip_s = clip_s
		self.speech_left = 0.0
		self.music_len = 0.0
		self.music_pos = 0.0
		self.music_on = False
		self.clips = []
		self.stops = []

	def play_clip(self, category, channel="speech"):
		self.clips.append(category)
		self.speech_left = self.clip_s
		return self.clip_s

	def play_music(self, pcm, loop=False):
		self.music_len = len(pcm) / 24000.0
		self.music_pos = 0.0
		self.music_on = True
		return self.music_len

	def stop(self, channel=None):
		self.stops.append(channel)
		if channel in (None, "music"):
			self.music_on = False
		if channel in (None, "speech"):
			self.speech_left = 0.0

	def is_speaking(self):
		return self.speech_left > 0.0

	def is_music_playing(self):
		return self.music_on

	def music_position_s(self):
		return self.music_pos

	def advance(self):
		self.speech_left = max(0.0, self.speech_left - self.DT)
		if self.music_on:
			self.music_pos += self.DT
			if self.music_pos >= self.music_len:
				self.music_on = False


class FakeExpress(object):

	def __init__(self):
		self.overrides = {}
		self.history = []
		self.released = []

	def override(self, channel, value, ttl_ticks=5, owner=""):
		self.overrides[channel] = (list(value), owner)
		self.history.append(channel)

	def release(self, owner):
		self.released.append(owner)
		for k in list(self.overrides):
			if self.overrides[k][1] == owner:
				del self.overrides[k]


class FakeCues(object):

	def __init__(self):
		self.modes = []
		self.lids = None

	def set_mode(self, mode):
		self.modes.append(mode)

	def set_dance_eyelids(self, lids):
		self.lids = list(lids)


class FakeAffect(object):

	def __init__(self):
		self.targets = []

	def set_forebrain_target(self, valence, arousal, gain=0.05, seconds=2.0):
		self.targets.append((valence, arousal, seconds))


def load_action_dance():

	# import action/action_dance.py on top of the fake base, then put any
	# real action.forebrain_action back so other tests are unaffected
	_testpath.install_stubs()
	import action
	fake = types.ModuleType("action.forebrain_action")
	fake.ForebrainAction = FakeForebrainAction
	saved_mod = sys.modules.get("action.forebrain_action")
	saved_attr = getattr(action, "forebrain_action", None)
	saved_dance = sys.modules.pop("action.action_dance", None)
	sys.modules["action.forebrain_action"] = fake
	action.forebrain_action = fake
	try:
		mod = importlib.import_module("action.action_dance")
	finally:
		if saved_mod is None:
			sys.modules.pop("action.forebrain_action", None)
		else:
			sys.modules["action.forebrain_action"] = saved_mod
		if saved_attr is None:
			del action.forebrain_action
		else:
			action.forebrain_action = saved_attr
		if saved_dance is None:
			sys.modules.pop("action.action_dance", None)
		else:
			sys.modules["action.action_dance"] = saved_dance
	return mod


class Song(object):

	def __init__(self, bpm=120.0, genre="pop"):
		self.id = "test"
		self.title = "Test Song"
		self.bpm = bpm
		self.genre = genre


class TestActionDance(unittest.TestCase):

	@classmethod
	def setUpClass(cls):

		cls.mod = load_action_dance()

	def make(self, max_duration_s=60.0, cues=True, wheels=True):

		cfg = Section({"dance": {"max_duration_s": max_duration_s, "default_bpm": 120.0,
			"wheels": wheels, "allow_translation": True}})
		pars = types.SimpleNamespace(
			forebrain=types.SimpleNamespace(cfg=cfg),
			timing=types.SimpleNamespace(tick_hz=50),
			action=types.SimpleNamespace(priority_forebrain=0.84))
		nodes = types.SimpleNamespace(
			forebrain=FakeForebrain(), voice=FakeVoice(), express=FakeExpress(),
			cues=FakeCues() if cues else None, affect=FakeAffect())
		parent = types.SimpleNamespace(pars=pars, kc_m=FakeKc(), nodes=nodes)
		return self.mod.ActionDance(parent), nodes

	def run_until_stopped(self, act, nodes, max_ticks=10000, on_tick=None):

		# what the stock action loop does: start once, then service while
		# the clock is held open
		act.start()
		n = 0
		while act.clock.isActive() and n < max_ticks:
			act.service()
			nodes.voice.advance()
			n += 1
			if on_tick is not None:
				on_tick(n)
		return n

	def test_wanted(self):

		act, nodes = self.make()
		self.assertEqual(act.name, "dance")
		self.assertFalse(act.modulate_by_wakefulness)
		self.assertFalse(act.wanted())
		nodes.forebrain.request_task("tricks", {})
		self.assertFalse(act.wanted())
		nodes.forebrain.request_task("dance", {"song": Song(), "pcm": np.zeros(24000, np.int16)})
		self.assertTrue(act.wanted())

	def test_full_song(self):

		act, nodes = self.make()
		pcm = np.zeros(int(3.0 * 24000), np.int16)
		nodes.forebrain.request_task("dance", {"song": Song(120, "pop"), "pcm": pcm, "intro": True})
		start_cfg = act.kc.getConfig()
		n = self.run_until_stopped(act, nodes)

		# intro clip, music, end of task
		self.assertEqual(nodes.voice.clips, ["dance_start"])
		self.assertAlmostEqual(nodes.voice.music_len, 3.0)
		self.assertEqual(nodes.forebrain.ended, [("dance", "done")])
		self.assertIsNone(nodes.forebrain.active_task)
		self.assertFalse(act.clock.isActive())
		self.assertIsNone(act.phase)
		# ~0.3 s intro + 3 s song + 1 s outro
		self.assertTrue(3.5 * 50 < n < 5.5 * 50, n)

		# head always in limits, ends at neutral, inputs untouched
		for h in act.heads:
			self.assertTrue(within_limits(h), h)
		self.assertEqual(act.heads[-1], moves.neutral())
		self.assertNotEqual(start_cfg, act.heads[-1])

		# lights / ears / tail driven, released at the end, cues restored
		for ch in ("illum", "ears", "tail"):
			self.assertIn(ch, nodes.express.history)
		self.assertNotIn("eyelids", nodes.express.history)
		self.assertIsNotNone(nodes.cues.lids)
		self.assertEqual(nodes.express.overrides, {})
		self.assertIn("dance", nodes.express.released)
		self.assertEqual(nodes.cues.modes[0], "dancing")
		self.assertEqual(nodes.cues.modes[-1], "")
		self.assertEqual(len(nodes.affect.targets), 1)
		self.assertGreaterEqual(act.base_stops, 1)

	def test_stop_word(self):

		act, nodes = self.make()
		pcm = np.zeros(int(20.0 * 24000), np.int16)
		nodes.forebrain.request_task("dance", {"song": Song(), "pcm": pcm, "intro": False})

		def on_tick(n):
			if n == 100:
				# forebrain.stop_all() clears the task (and the audio)
				nodes.forebrain.active_task = None
				nodes.voice.stop()

		n = self.run_until_stopped(act, nodes, on_tick=on_tick)
		self.assertEqual(n, 101)
		self.assertEqual(nodes.voice.clips, [])
		self.assertFalse(nodes.voice.music_on)
		self.assertEqual(nodes.express.overrides, {})
		self.assertEqual(nodes.cues.modes[-1], "")

	def test_max_duration(self):

		act, nodes = self.make(max_duration_s=2.0, cues=False)
		pcm = np.zeros(int(30.0 * 24000), np.int16)
		nodes.forebrain.request_task("dance", {"song": Song(150, "rock"), "pcm": pcm, "intro": False})
		n = self.run_until_stopped(act, nodes)
		self.assertTrue(2.5 * 50 < n < 3.5 * 50, n)
		self.assertIn("music", nodes.voice.stops)
		self.assertFalse(nodes.voice.music_on)
		self.assertEqual(nodes.forebrain.ended, [("dance", "done")])
		# without NodeCues the eyelids go through NodeExpress
		self.assertIn("eyelids", nodes.express.history)

	def test_preempted(self):

		act, nodes = self.make()
		pcm = np.zeros(int(20.0 * 24000), np.int16)
		nodes.forebrain.request_task("dance", {"song": Song(), "pcm": pcm, "intro": False})
		act.start()
		for _ in range(60):
			act.service()
			nodes.voice.advance()
		self.assertTrue(nodes.voice.music_on)
		# the stock loop calls stop() when another action wins
		act.stop()
		self.assertFalse(nodes.voice.music_on)
		self.assertEqual(nodes.forebrain.ended, [("dance", "preempted")])
		self.assertFalse(act.clock.isActive())

	def test_nothing_to_claim(self):

		act, nodes = self.make()
		act.start()
		self.assertFalse(act.clock.isActive())
		nodes.forebrain.request_task("dance", {"song": Song(), "pcm": np.zeros(0, np.int16)})
		act.start()
		self.assertFalse(act.clock.isActive())
		self.assertEqual(nodes.forebrain.ended, [("dance", "no music")])

	def test_no_wheels(self):

		act, nodes = self.make(wheels=False)
		pcm = np.zeros(int(15.0 * 24000), np.int16)
		nodes.forebrain.request_task("dance", {"song": Song(120, "dance"), "pcm": pcm, "intro": False})
		self.run_until_stopped(act, nodes)
		self.assertEqual(act.vels, [])


if __name__ == "__main__":
	unittest.main()
