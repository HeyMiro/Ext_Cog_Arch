#
#	Hey MiRo - head moves.
#
#	Every function here is pure: it takes a start configuration (and a
#	time) and returns a NEW kinematic configuration [tilt, lift, yaw,
#	pitch] in radians, clipped to MiRo's joint limits and free of NaN.
#	Inputs are never modified (HRI's slight_move() edited the caller's
#	list in place).
#
#	Ported from HRI'25 miro_movebank.py and dance_body.py, with fixes:
#	  * transitions use a min-jerk Trajectory that really goes from the
#	    start to the target (HRI's s-curve ran the wrong way and did
#	    not reach its end point at the stated duration)
#	  * nod() is a plain sinusoid from the current pitch (HRI used
#	    arcsin() of the pitch, which gave NaN outside the nominal range)
#	  * beat moves take bpm and use the beat period 60 / bpm (HRI passed
#	    the period where a frequency was expected)
#	  * moves keep the current lift / yaw instead of commanding 0 rad
#	    (HRI's head_bop sent lift = 0, below the joint minimum)
#
#	Joint conventions (MiRo): lift small = neck raised, large = neck
#	lowered forward; pitch negative = head tipped up, positive = down;
#	yaw positive = left.
#

import math

try:
	import miro2 as miro
	_c = miro.constants
	TILT = float(_c.TILT_RAD_CALIB)
	LIFT_MIN, LIFT_MAX, LIFT_CALIB = float(_c.LIFT_RAD_MIN), float(_c.LIFT_RAD_MAX), float(_c.LIFT_RAD_CALIB)
	YAW_MIN, YAW_MAX = float(_c.YAW_RAD_MIN), float(_c.YAW_RAD_MAX)
	PITCH_MIN, PITCH_MAX = float(_c.PITCH_RAD_MIN), float(_c.PITCH_RAD_MAX)
except Exception:
	# same numbers as miro2.constants, for machines without the MDK
	# (any failure importing it just means we use these)
	_DEG = math.pi / 180.0
	TILT = -6.0 * _DEG
	LIFT_MIN, LIFT_MAX, LIFT_CALIB = 8.0 * _DEG, 60.0 * _DEG, 34.0 * _DEG
	YAW_MIN, YAW_MAX = -55.0 * _DEG, 55.0 * _DEG
	PITCH_MIN, PITCH_MAX = -22.0 * _DEG, 8.0 * _DEG

# indices into a kinematic configuration
TILT_I, LIFT_I, YAW_I, PITCH_I = 0, 1, 2, 3

LOWER = [TILT, LIFT_MIN, YAW_MIN, PITCH_MIN]
UPPER = [TILT, LIFT_MAX, YAW_MAX, PITCH_MAX]

# HRI miro_movebank.reset(): lift 0.64, pitch -0.02. HRI also used yaw
# 0.19, which compensated for one robot's yaw offset; we centre it.
NEUTRAL = [TILT, 0.64, 0.0, -0.02]



# ------------------------------------------------------------------ helpers

def clip_config(cfg):

	# copy, replace NaN / inf by the neutral value and clip to limits;
	# tilt is not actuated on MiRo so it is always the fixed value
	out = []
	for i in range(4):
		try:
			x = float(cfg[i])
		except (TypeError, ValueError, IndexError):
			x = NEUTRAL[i]
		if not math.isfinite(x):
			x = NEUTRAL[i]
		out.append(min(max(x, LOWER[i]), UPPER[i]))
	return out


def _cfg(start, lift=None, yaw=None, pitch=None):

	# build a new clipped config from start, replacing the given joints
	out = clip_config(start if start is not None else NEUTRAL)
	if lift is not None:
		out[LIFT_I] = lift
	if yaw is not None:
		out[YAW_I] = yaw
	if pitch is not None:
		out[PITCH_I] = pitch
	return clip_config(out)


def min_jerk(x):

	# 0..1 -> 0..1 with zero velocity and acceleration at both ends
	x = min(max(float(x), 0.0), 1.0)
	return x * x * x * (10.0 + x * (-15.0 + 6.0 * x))


def beat_period(bpm):

	# seconds per beat; a missing / silly tempo falls back to 120 bpm
	# (HRI divided by a tempo of 0 when Spotify returned nothing)
	try:
		bpm = float(bpm)
	except (TypeError, ValueError):
		bpm = 120.0
	if not math.isfinite(bpm) or bpm <= 0.0:
		bpm = 120.0
	bpm = min(max(bpm, 40.0), 240.0)
	return 60.0 / bpm


def cycle_beats(bpm, beats, max_hz):

	# MiRo's neck is slow: double the cycle length (in beats) until the
	# joint would move at or below max_hz, so moves stay on the beat
	# grid but never ask the servos for more than they can do
	T = beat_period(bpm)
	while 1.0 / (beats * T) > max_hz:
		beats *= 2
	return beats


def _phase(t, bpm, beats):

	# phase in radians of a cycle lasting `beats` beats, 0 at t = 0
	return 2.0 * math.pi * float(t) / (beats * beat_period(bpm))


def _swing(lo, hi, phase):

	# sinusoid between lo and hi (HRI new_sine_generator, without the
	# frequency / period mix-up), at the midpoint when phase = 0
	return 0.5 * (lo + hi) + 0.5 * (hi - lo) * math.sin(phase)



# --------------------------------------------------------------- trajectory

class Trajectory(object):

	"""
	Min-jerk move from start_cfg to target_cfg over duration_s.
	sample(t) -> (cfg, done) with t in seconds since the move began;
	at t >= duration_s it returns the (clipped) target and done = True.
	"""

	def __init__(self, start_cfg, target_cfg, duration_s):

		# private copies: callers may keep editing their lists
		self.start = clip_config(start_cfg)
		self.target = clip_config(target_cfg)
		self.duration_s = max(0.0, float(duration_s))

	def sample(self, t):

		if self.duration_s <= 0.0 or t >= self.duration_s:
			return list(self.target), True
		s = min_jerk(max(0.0, float(t)) / self.duration_s)
		cfg = [a + s * (b - a) for a, b in zip(self.start, self.target)]
		return clip_config(cfg), False



# ------------------------------------------------------- conversation poses

def neutral():

	return list(NEUTRAL)


def look_up(start):

	# neck up, head tipped back: looking up at a person (HRI look_up)
	return _cfg(start, lift=0.14, pitch=-0.38)


def look_down(start):

	# neck forward and low, head down: sad / ashamed (HRI look_down)
	return _cfg(start, lift=1.04, pitch=0.12)


def lean_in(start, side=0.0):

	# lean towards the listener, looking up at them (HRI lean_forward);
	# side in -1..1 turns the head (HRI picked +-0.75 rad at random)
	yaw = clip_config(start)[YAW_I] + 0.75 * float(side)
	return _cfg(start, lift=0.75, yaw=yaw, pitch=-0.35)


def tilt(start, side=1.0):

	# "thinking" head cock: MiRo has no roll joint, so turn a little to
	# one side and look up (HRI turn_head)
	side = 1.0 if float(side) >= 0.0 else -1.0
	return _cfg(start, lift=0.5, yaw=0.35 * side, pitch=-0.25)


def light_up(start=None):

	# HRI "happy" pose: head up and centred
	return _cfg(start, lift=0.399, yaw=0.0, pitch=-0.21)



# ----------------------------------------------------------- periodic moves

def nod(start, t, period_s=1.0, depth=0.12):

	# yes-nod: pitch dips by `depth` and comes back once per period,
	# starting (and ending) at the current pitch so there is no jump;
	# if there is no room below the pitch limit it nods upwards instead
	p0 = clip_config(start)[PITCH_I]
	period_s = max(0.1, float(period_s))
	offset = float(depth) * 0.5 * (1.0 - math.cos(2.0 * math.pi * float(t) / period_s))
	if p0 + depth > PITCH_MAX:
		offset = -offset
	return _cfg(start, pitch=p0 + offset)


def glance(start, t, side=1.0, duration_s=1.2, amplitude=0.35):

	# quick look to one side and back (sin^2 bump: smooth at both ends);
	# after duration_s it returns the start configuration
	x = float(t) / max(0.1, float(duration_s))
	if x <= 0.0 or x >= 1.0:
		return clip_config(start)
	bump = math.sin(math.pi * x) ** 2
	c = clip_config(start)
	return _cfg(c, yaw=c[YAW_I] + float(side) * amplitude * bump, pitch=c[PITCH_I] - 0.08 * bump)


def ear_flap(t, rate_hz=2.0):

	# HRI flourish: ears fully forward then back, rate_hz times a second
	on = (float(t) * float(rate_hz)) % 1.0 < 0.5
	v = 1.0 if on else 0.0
	return [v, v]


def head_bop(start, t, bpm):

	# one bop per beat (HRI head_bop): the head snaps down ON the beat
	# (the cusp of |sin|) and floats up between beats
	x = abs(math.sin(math.pi * float(t) / beat_period(bpm)))
	return _cfg(start, pitch=0.06 - 0.26 * x)


def head_bang(start, t, bpm):

	# big neck + head swing together (HRI head_Banging), one cycle per
	# two beats, slowed to every 4 / 8 beats for fast songs
	ph = _phase(t, bpm, cycle_beats(bpm, 2, 1.0)) - 0.5 * math.pi
	return _cfg(start, lift=_swing(0.3, 0.95, ph), yaw=0.0, pitch=_swing(-0.3, 0.12, ph))


def head_spin(start, t, bpm):

	# head rolls round in a circle (HRI full_head_spin): yaw swings side
	# to side while pitch runs a quarter cycle ahead
	ph = _phase(t, bpm, cycle_beats(bpm, 2, 0.5))
	return _cfg(start, yaw=_swing(-0.9, 0.9, ph), pitch=_swing(-0.36, 0.12, ph + 0.5 * math.pi))


def party_nod(start, t, bpm):

	# HRI soul head bounce / party nod: neck bounces once per beat while
	# the head sweeps side to side over four (or more) beats
	bounce = abs(math.sin(math.pi * float(t) / beat_period(bpm)))
	ph = _phase(t, bpm, cycle_beats(bpm, 4, 0.5))
	return _cfg(start, lift=0.45 + 0.3 * bounce, yaw=_swing(-0.8, 0.8, ph), pitch=0.0)


def free_dance(start, t, bpm, ranges):

	# HRI "general dancing": each joint swings within its own range at
	# its own rate. ranges = {"lift"|"yaw"|"pitch": (lo, hi, cycles_per_beat,
	# phase)}; a joint that is missing keeps its start value
	T = beat_period(bpm)
	vals = {}
	for name, (lo, hi, cpb, phase) in ranges.items():
		vals[name] = _swing(lo, hi, 2.0 * math.pi * float(cpb) * float(t) / T + phase)
	return _cfg(start, lift=vals.get("lift"), yaw=vals.get("yaw"), pitch=vals.get("pitch"))


def blend(cfg_a, cfg_b, w):

	# w = 0 -> a, w = 1 -> b (both clipped); used to cross-fade moves
	w = min(max(float(w), 0.0), 1.0)
	a = clip_config(cfg_a)
	b = clip_config(cfg_b)
	return clip_config([x + w * (y - x) for x, y in zip(a, b)])
