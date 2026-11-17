#
#	Hey MiRo - ActionTricks: show MiRo a hand sign and it does a trick.
#
#	Runs the forebrain "tricks" task (asked for in conversation, with the
#	"tricks" command, or - if features.spontaneous_gestures is on - when
#	someone makes a hand sign while MiRo is idle). The scene client
#	recognises gestures (mediapipe); this action
#
#	  wait     looks up at the user and waits for a fresh, confident gesture
#	  perform  says a line and does the trick mapped to it (tricks.map):
#	             greet  nod along         (Open_Palm)
#	             spin   spin on the spot  (Pointing_Up)
#	             ears   lean in, flap ears (Victory)
#	             party  rainbow LEDs and a party nod (ILoveYou)
#	  cooldown a short pause, so one sign is one trick
#
#	and ends after tricks.duration_s or tricks.max_tricks tricks. HRI'25
#	synthesized every line with a blocking ElevenLabs call inside the
#	tick and ended after 2-3 s; here lines go through the forebrain's
#	cached, non-blocking say().
#

import random

from . import forebrain_action
from choreography import moves, lights

# HRI'25 lines for each trick (one picked at random)
LINES = {
	"greet": ["Are you waving at me?", "Hello there!", "I'm sorry, I don't have arms so I can't wave back!", "HI HI HI!"],
	"spin": ["Here I go!", "Look at me go!", "WOOHOO!", "Woah woah woah!"],
	"ears": ["Heck yeah, I can do a peace sign too!", "Yeahhh, peace and love!", "Hee hee, I can do that too!"],
	"party": ["ROCK ON, YEAH!", "Woooo!", "Heck yeah!", "Untz untz untz!"],
}

INTRO_LINE = "Okay! Do some hand signs and I'll show you some tricks!"

# head pose while watching for hand signs (HRI'25 [0, 0.4, 0, 0.1])
WATCH_LIFT = 0.4
WATCH_PITCH = 0.1
WATCH_POSE_S = 1.0

# trick timings (HRI'25 values)
GREET_MIN_S = 1.5
GREET_NOD_PERIOD_S = 0.8
SPIN_S = 1.9
SPIN_SPEED = 0.5
EARS_CYCLES = 4
EARS_CYCLE_S = 1.0
PARTY_S = 4.0
PARTY_BPM = 120.0
RAINBOW_STEP_S = 0.1
RAINBOW = ["purple", "blue", "green", "yellow", "orange", "red"]

# never wait longer than this for the line to finish
TRICK_MAX_S = 8.0
COOLDOWN_S = 1.0



class ActionTricks(forebrain_action.ForebrainAction):

	NAME = "tricks"
	TASK = "tricks"

	def init(self):

		cfg = self.pars.forebrain.cfg
		self.duration_s = float(cfg.get("tricks.duration_s", 20.0))
		self.max_tricks = int(cfg.get("tricks.max_tricks", 5))
		trick_map = cfg.get("tricks.map")
		self.trick_map = trick_map.to_dict() if trick_map is not None else {}
		self.rng = random.Random()
		self._reset()

	def _reset(self):

		self.phase = None
		self.t_phase = 0.0
		self.trick = None
		self.done_tricks = 0
		self.last_gesture_t = None
		self.traj = None
		self.start_cfg = None

	def wanted(self):

		return self.task_wanted()

	# ------------------------------------------------------------- lifecycle

	def on_start(self):

		fb = self.forebrain
		params = fb.claim_task(self.TASK)
		if params is None and fb.active_task != self.TASK:
			return False
		params = params or {}
		self._reset()
		print("[tricks] start")
		start = list(self.kc.getConfig())
		target = list(start)
		target[moves.LIFT_I] = WATCH_LIFT
		target[moves.PITCH_I] = WATCH_PITCH
		self.traj = moves.Trajectory(start, target, WATCH_POSE_S)
		if params.get("intro"):
			fb.say(INTRO_LINE)
		self._phase("wait")

	def on_stop(self, reason):

		print("[tricks] end (" + reason + ", " + str(self.done_tricks) + " tricks)")
		self._reset()

	def _phase(self, phase):

		self.phase = phase
		self.t_phase = self.now()
		self.start_cfg = list(self.kc.getConfig())

	def on_service(self):

		t = self.now() - self.t_phase

		if self.phase == "wait":
			if self.traj is not None:
				cfg, done = self.traj.sample(self.elapsed())
				self.set_head(cfg)
				if done:
					self.traj = None
			if self.elapsed() > self.duration_s or self.done_tricks >= self.max_tricks:
				self.finish("done")
				return
			self._look_for_gesture()

		elif self.phase == "perform":
			if self._perform(t) or t > TRICK_MAX_S:
				self.done_tricks += 1
				self._phase("cooldown")

		elif self.phase == "cooldown":
			if t > COOLDOWN_S:
				self._phase("wait")

	# ---------------------------------------------------------------- tricks

	def _look_for_gesture(self):

		gesture = self.forebrain.fresh_gesture()
		if gesture is None:
			return
		# one sign is one trick: ignore the detection we already used
		stamp = gesture.get("t")
		if stamp is not None and stamp == self.last_gesture_t:
			return
		trick = self.trick_map.get(gesture.get("gesture"))
		if trick not in LINES:
			return
		self.last_gesture_t = stamp
		self.trick = trick
		print("[tricks] " + str(gesture.get("gesture")) + " -> " + trick)
		self.forebrain.say(self.rng.choice(LINES[trick]))
		self._phase("perform")

	def _speaking(self):

		voice = self.voice
		return self.forebrain.is_saying() or (voice is not None and voice.is_speaking())

	def _perform(self, t):

		# one tick of the current trick; True when it is complete
		if self.trick == "greet":
			self.set_head(moves.nod(self.start_cfg, t, GREET_NOD_PERIOD_S, 0.12))
			return t > GREET_MIN_S and not self._speaking()

		if self.trick == "spin":
			if t < SPIN_S:
				self.body_velocity(0.0, SPIN_SPEED)
				return False
			return not self._speaking()

		if self.trick == "ears":
			if t < EARS_CYCLES * EARS_CYCLE_S:
				phase = (t % EARS_CYCLE_S) / EARS_CYCLE_S
				lean = moves.Trajectory(self.start_cfg, moves.lean_in(self.start_cfg), 0.5 * EARS_CYCLE_S)
				self.set_head(lean.sample(min(t, 0.5 * EARS_CYCLE_S))[0])
				ears = [1.0, 1.0] if phase < 0.5 else [0.0, 0.0]
				self.override("ears", ears)
				return False
			return not self._speaking()

		if self.trick == "party":
			step = int(t / RAINBOW_STEP_S) % len(RAINBOW)
			self.override("illum", lights.solid(RAINBOW[step]))
			self.set_head(moves.party_nod(self.start_cfg, t, PARTY_BPM))
			return t > PARTY_S and not self._speaking()

		return True
