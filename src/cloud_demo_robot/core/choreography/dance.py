#
#	Hey MiRo - dance controller.
#
#	DanceController turns "time into the song" into a DanceFrame: head
#	configuration, wheel velocity, cosmetic joints and LEDs. It is a
#	pure function of t (plus a plan made once from the seed), so it can
#	be driven by the music position reported by NodeVoice and replayed
#	in tests.
#
#	Structure (as HRI'25 action_dance / dance_body / dance_wheels /
#	dance_illum, fixed):
#	  * a beat clock with period 60 / bpm (HRI mixed beat period and
#	    beat frequency between modules)
#	  * the song is cut into sections of 8, 12 or 16 beats; each section
#	    has one move. Like HRI, sections alternate between a programmed
#	    move chosen for the genre and "free" dancing where each joint
#	    swings in its own randomly chosen range
#	  * moves cross-fade over one beat at section changes so the head
#	    never jumps
#	  * wheels only move in "spin" and "rock" sections, in whole-beat
#	    segments that cancel out (spins alternate direction, rocking
#	    goes forward and back), so MiRo stays where it started
#	  * every section changes the LED pattern (beat flash or fade
#	    between two colours from the genre palette)
#	  * the dance has a fixed duration and fades out over its last
#	    second (HRI could dance forever if the song never ended)
#

import bisect
import math
import random

from . import moves
from . import lights


# programmed moves, and the ones that suit each genre (HRI change_dance)
MOVES = ("head_bop", "head_bang", "spin", "rock", "party")

GENRE_MOVES = {
	"pop": ("party", "spin", "head_bop"),
	"disco": ("party", "spin", "rock"),
	"soul": ("party", "head_bop"),
	"rock": ("head_bang", "spin"),
	"dance": ("head_bang", "spin", "rock", "head_bop"),
	"default": MOVES,
}

# wheel pushes, in the units of ForebrainAction.body_velocity (HRI
# dance_wheels push values: full spin 0.5, small spin 0.1, step 0.1)
SPIN_SPEED = 0.5
ROCK_TURN = 0.15
ROCK_STEP = 0.1

# cosmetic joint calibration (ears / tail rest positions)
EAR_REST = 0.3333
WAG_REST = 0.5



class DanceFrame(object):

	"""
	What to do on one tick of the dance.
	head       [tilt, lift, yaw, pitch] (clipped)
	wheels     None (do not drive) or [vx, wz] body velocity push
	cosmetics  dict with optional "ears" [l, r], "tail" [droop, wag],
	           "eyelids" [l, r] (0 open .. 1 closed)
	illum      six uint32 0xAARRGGBB
	done       True once t >= duration_s
	"""

	def __init__(self, head, wheels, cosmetics, illum, done, move="", section=-1, beat=0.0):

		self.head = head
		self.wheels = wheels
		self.cosmetics = cosmetics
		self.illum = illum
		self.done = done
		self.move = move
		self.section = section
		self.beat = beat

	def __repr__(self):

		return "DanceFrame(%s, section %d, beat %.2f, done %s)" % (self.move, self.section, self.beat, self.done)



class Section(object):

	def __init__(self, index, start_s, end_s, beats, move, light_mode, colours, ranges):

		self.index = index
		self.start_s = start_s
		self.end_s = end_s
		self.beats = beats
		self.move = move
		self.light_mode = light_mode    # "flash" | "fade"
		self.colours = colours          # two (r, g, b)
		self.ranges = ranges            # free dance joint ranges, or None

	def __repr__(self):

		return "Section(%d, %.2f-%.2f s, %d beats, %s, %s)" % (
			self.index, self.start_s, self.end_s, self.beats, self.move, self.light_mode)



def dance_affect(bpm, genre):

	# affect target while dancing: always pleased; faster songs are more
	# arousing, and the genre nudges it (HRI node_affect.drive_by_dance:
	# soul calms, pop and dance music excite)
	T = moves.beat_period(bpm)
	arousal = 0.3 + 0.6 * min(max((60.0 / T - 60.0) / 120.0, 0.0), 1.0)
	arousal += {"soul": -0.1, "pop": 0.1, "dance": 0.05, "disco": 0.05, "rock": 0.05}.get(str(genre).lower(), 0.0)
	return 0.85, min(max(arousal, 0.0), 1.0)



class DanceController(object):

	def __init__(self, bpm, genre, duration_s, seed=None, wheels=True, allow_translation=True,
			home=None, fade_s=1.0):

		self.beat_s = moves.beat_period(bpm)
		self.bpm = 60.0 / self.beat_s
		self.genre = str(genre or "default").lower()
		if self.genre not in GENRE_MOVES:
			self.genre = "default"
		try:
			duration_s = float(duration_s)
		except (TypeError, ValueError):
			duration_s = 0.0
		self.duration_s = max(0.0, duration_s) if math.isfinite(duration_s) else 0.0
		self.wheels = bool(wheels)
		self.allow_translation = bool(allow_translation)
		self.home = moves.clip_config(home if home is not None else moves.NEUTRAL)
		self.fade_s = max(0.0, float(fade_s))
		self.palette = lights.palette(self.genre)

		# the whole plan is drawn up front from one RNG, so the same
		# seed always gives the same dance
		self.rng = random.Random(seed)
		self.sections = self._plan()
		self.starts = [s.start_s for s in self.sections]

	# ------------------------------------------------------------- planning

	def _plan(self):

		rng = self.rng
		sections = []
		start_s = 0.0
		last_move = None
		while start_s < self.duration_s or not sections:
			beats = rng.choice((8, 12, 16))
			index = len(sections)

			# alternate programmed and free dancing (HRI autoState)
			if index % 2 == 1:
				move = "free"
				ranges = self._free_ranges(rng)
			else:
				options = [m for m in GENRE_MOVES[self.genre] if m != last_move] or list(GENRE_MOVES[self.genre])
				move = rng.choice(options)
				last_move = move
				ranges = None

			light_mode = rng.choice(("flash", "fade"))
			if len(self.palette) >= 2:
				colours = rng.sample(self.palette, 2)
			else:
				colours = list(self.palette) * 2

			end_s = start_s + beats * self.beat_s
			sections.append(Section(index, start_s, end_s, beats, move, light_mode, colours, ranges))
			start_s = end_s
		return sections

	def _free_ranges(self, rng):

		# port of HRI change_{yaw,lift,pitch}_values, fixed: fast songs use
		# a third of each joint's dance range at double rate, medium songs
		# half at normal rate, slow songs the full range at half rate (HRI
		# set min = max for slow pitch, so it never moved); a joint rests
		# for the section with probability 1/4
		if self.beat_s <= 0.45:
			frac, rate = 1.0 / 3.0, 2.0
		elif self.beat_s < 0.75:
			frac, rate = 0.5, 1.0
		else:
			frac, rate = 1.0, 0.5

		# (joint, low, high, cycles per beat at rate 1)
		joints = (("yaw", -0.8, 0.8, 0.25), ("lift", 0.3, 0.9, 0.25), ("pitch", -0.3, 0.1, 0.5))
		ranges = {}
		for name, lo, hi, cpb in joints:
			span = (hi - lo) * frac
			a = rng.uniform(lo, hi - span)
			phase = rng.uniform(0.0, math.pi)
			rest = rng.randint(1, 4) == 1
			if not rest:
				# pitch keeps its rate: it carries the beat
				ranges[name] = (a, a + span, cpb if name == "pitch" else cpb * rate, phase)
		return ranges

	# ---------------------------------------------------------------- frame

	def section_at(self, t):

		i = bisect.bisect_right(self.starts, max(0.0, t)) - 1
		return self.sections[min(max(i, 0), len(self.sections) - 1)]

	def _head(self, sec, t):

		home = self.home
		if sec.move == "head_bop" or sec.move == "rock":
			return moves.head_bop(home, t, self.bpm)
		if sec.move == "head_bang":
			return moves.head_bang(home, t, self.bpm)
		if sec.move == "spin":
			return moves.head_spin(home, t, self.bpm)
		if sec.move == "party":
			return moves.party_nod(home, t, self.bpm)
		return moves.free_dance(home, t, self.bpm, sec.ranges or {})

	def _wheels(self, sec, t):

		if not self.wheels:
			return None
		b = (t - sec.start_s) / self.beat_s
		if sec.move == "spin":
			# spin for the first half of each 8-beat phrase, alternating
			# direction phrase by phrase
			phrase = int(b // 8)
			if b - 8 * phrase < 4.0:
				sign = 1.0 if (sec.index // 2 + phrase) % 2 == 0 else -1.0
				return [0.0, sign * SPIN_SPEED]
			return None
		if sec.move == "rock":
			# twist left / right each beat; step forward / back every two
			# beats (sections are whole bars, so the steps cancel out)
			n = int(b)
			wz = ROCK_TURN if n % 2 == 0 else -ROCK_TURN
			vx = 0.0
			if self.allow_translation:
				vx = ROCK_STEP if (n // 2) % 2 == 0 else -ROCK_STEP
			return [vx, wz]
		return None

	def _lights(self, sec, t):

		if sec.light_mode == "flash":
			return lights.flash(sec.colours, t, self.bpm)
		return lights.fade(sec.colours, t, self.bpm, beats=2.0)

	def _cosmetics(self, t, k):

		# ears wiggle over two beats, tail wags once per beat (HRI
		# move_ears / wag_tail); k scales them down during the fade-out.
		# eyelids: wide open for fast songs, sleepy for slow ones (HRI
		# set_eyes: beat period - 0.3, limited to 0.15..0.5)
		T = self.beat_s
		ear = 0.5 + 0.5 * math.sin(math.pi * t / T)
		wag = WAG_REST + 0.45 * math.sin(2.0 * math.pi * t / T)
		ear = EAR_REST + k * (ear - EAR_REST)
		wag = WAG_REST + k * (wag - WAG_REST)
		lid = min(max(T - 0.3, 0.15), 0.5)
		return {"ears": [ear, ear], "tail": [0.0, wag], "eyelids": [lid, lid]}

	def step(self, t):

		try:
			t = float(t)
		except (TypeError, ValueError):
			t = 0.0
		if not math.isfinite(t) or t < 0.0:
			t = 0.0

		if t >= self.duration_s:
			return DanceFrame(list(self.home), None, {}, lights.solid((0, 0, 0)), True, "end",
				len(self.sections) - 1, t / self.beat_s)

		sec = self.section_at(t)
		head = self._head(sec, t)

		# cross-fade from the previous move (or from home at the start)
		# over the first beat of each section
		into = t - sec.start_s
		if into < self.beat_s:
			if sec.index > 0:
				prev = self._head(self.sections[sec.index - 1], t)
			else:
				prev = self.home
			head = moves.blend(prev, head, moves.min_jerk(into / self.beat_s))

		wheels = self._wheels(sec, t)
		illum = self._lights(sec, t)

		# fade out over the last fade_s seconds: back towards home, lights
		# dimming, wheels stopped
		k = 1.0
		if self.fade_s > 0.0 and t > self.duration_s - self.fade_s:
			k = (self.duration_s - t) / self.fade_s
			head = moves.blend(self.home, head, k)
			illum = lights.scale(illum, k)
			wheels = None

		return DanceFrame(head, wheels, self._cosmetics(t, k), illum, False, sec.move, sec.index, t / self.beat_s)
