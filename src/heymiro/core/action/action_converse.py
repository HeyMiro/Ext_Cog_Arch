#
#	Hey MiRo - ActionConverse: the body language of a conversation.
#
#	NodeDialogue decides what is said; this action moves the robot to
#	match, phase by phase (HRI'25 action_llm.py states in brackets):
#
#	  facing      turn to the user [face_user]: rotate on the spot until an
#	              AprilTag is seen in 3 of the last 5 ticks (converse.face_user
#	              = apriltag; HRI'25 had no timeout), or turn the head
#	              toward the last sound (= sound), or skip (= none)
#	  greeting    lean in toward the user [look_up / hey_miro]
#	  listening   hold still, nod when the user starts speaking [listening]
#	  thinking    cock the head, LEDs pulse while the cloud works
#	  responding  emotion pose and LED colour [responding]: happy looks up
#	              (green), sad / angry / worried / scared look down (blue,
#	              red, orange, purple); gentle nodding while speaking;
#	              flourishes the user asked for: spin (time-bounded by
#	              converse.spin_s), ear flapping (10 flaps), impression
#	              (the voice is chosen by the dialogue)
#	  goodbye     back to the neutral pose
#
#	Every move is a trajectory sampled per tick (HRI'25 used a blocking
#	while-loop on the kinematic chain). If a higher-priority action
#	(halt, avert) pre-empts us the conversation carries on verbally; if
#	that lasts longer than converse.preempt_abort_s it is ended.
#

import collections

from . import forebrain_action
from choreography import moves, lights

# HRI'25 head pose while looking for the tag (lift, pitch)
FACE_LIFT = 0.64
FACE_PITCH = -0.02

# a tag must be seen in this many of the last FACE_WINDOW ticks
FACE_HITS = 3
FACE_WINDOW = 5

# an audio event older than this is no guide to where the user is
SOUND_FRESH_S = 3.0

# move durations (s)
FACE_POSE_S = 0.5
TURN_HEAD_S = 1.0
GREET_S = 1.0
THINK_S = 0.8
EMOTION_POSE_S = 1.0
EMOTION_HOLD_S = 2.5
GOODBYE_S = 1.5

# nods (s, rad)
LISTEN_NOD_PERIOD_S = 1.0
LISTEN_NOD_DEPTH = 0.12
TALK_NOD_PERIOD_S = 2.0
TALK_NOD_DEPTH = 0.06

# HRI'25 flapped the ears 10 times at 2 Hz
EAR_FLAPS = 10
EAR_FLAP_HZ = 2.0

# reply emotions that look down (HRI'25: sad and angry)
LOOK_DOWN = ("sad", "angry", "worried", "scared")



class ActionConverse(forebrain_action.ForebrainAction):

	NAME = "converse"

	def init(self):

		cfg = self.pars.forebrain.cfg
		self.face_user = str(cfg.get("converse.face_user", "none"))
		self.face_rotate_speed = float(cfg.get("converse.face_rotate_speed", 0.08))
		self.face_timeout_s = float(cfg.get("converse.face_timeout_s", 8.0))
		self.spin_speed = float(cfg.get("converse.spin_speed", 0.4))
		self.spin_s = float(cfg.get("converse.spin_s", 3.0))
		self.preempt_abort_s = float(cfg.get("converse.preempt_abort_s", 10.0))

		self.phase = None
		self.t_phase = 0.0
		self.traj = None
		self.base = None
		self.tags = collections.deque(maxlen=FACE_WINDOW)
		self.nod_t0 = None
		self.side = 1.0

		# the last sound heard (for face_user = sound), kept across ticks
		# because the stock audio events only live for one tick
		self.last_azim = None
		self.t_last_azim = -1e9

		# time we started wanting the body but not getting it
		self.t_preempted = None

	@property
	def dialogue(self):

		return self.node("dialogue")

	def wanted(self):

		d = self.dialogue
		return d is not None and d.engaged()

	# ------------------------------------------------------------- selection

	def ascending(self):

		self._track_sound()
		forebrain_action.ForebrainAction.ascending(self)

		# wanted but pre-empted (halt, avert, being picked up): keep
		# talking for a while, but do not hold a conversation forever
		# (being stroked hands the body to mull for a while: that is not a
		# reason to end the conversation, so touch does not count)
		if self.wanted() and self.interface.inhibition > 0 and not self.input.user_touch > 0:
			now = self.now()
			if self.t_preempted is None:
				self.t_preempted = now
			elif now - self.t_preempted > self.preempt_abort_s:
				print("[converse] pre-empted too long: ending the conversation")
				self.t_preempted = None
				self.dialogue.on_stop()
		else:
			self.t_preempted = None

	def _track_sound(self):

		events = getattr(self.system_state, "audio_events_for_50Hz", None) or []
		if events:
			self.last_azim = float(events[-1].azim)
			self.t_last_azim = self.now()

	# ------------------------------------------------------------- lifecycle

	def on_start(self):

		self.phase = None

	def on_stop(self, reason):

		self.phase = None

	def on_service(self):

		d = self.dialogue
		state = d.state_name()
		if state != self.phase:
			self._enter(state)
		t = self.now() - self.t_phase

		if state == "facing":
			self._service_facing(d, t)
		elif state == "greeting":
			self._follow(t)
		elif state == "listening":
			self._service_listening(d, t)
		elif state == "thinking":
			self._follow(t)
			self.override("illum", lights.thinking_pulse(t))
		elif state == "responding":
			self._service_responding(d, t)
		elif state == "goodbye":
			self._follow(t)

	def _enter(self, state):

		# new phase: start its move from where the head is now
		self.phase = state
		self.t_phase = self.now()
		start = list(self.kc.getConfig())
		self.base = start
		self.nod_t0 = None
		self.traj = None

		if state == "facing":
			self.tags.clear()
			if self.face_user == "sound" and self._sound_fresh():
				# turn the head toward the sound (yaw is clipped to its range)
				target = list(start)
				target[moves.YAW_I] = start[moves.YAW_I] + self.last_azim
				self.traj = moves.Trajectory(start, target, TURN_HEAD_S)
			else:
				target = list(start)
				target[moves.LIFT_I] = FACE_LIFT
				target[moves.PITCH_I] = FACE_PITCH
				self.traj = moves.Trajectory(start, target, FACE_POSE_S)

		elif state == "greeting":
			self.traj = moves.Trajectory(start, moves.lean_in(start), GREET_S)

		elif state == "thinking":
			self.side = -self.side
			self.traj = moves.Trajectory(start, moves.tilt(start, self.side), THINK_S)

		elif state == "responding":
			reply = self.dialogue.current_reply()
			emotion = getattr(reply, "emotion", None) or "fine"
			if emotion == "happy":
				target = moves.look_up(start)
			elif emotion in LOOK_DOWN:
				target = moves.look_down(start)
			else:
				target = start
			self.traj = moves.Trajectory(start, target, EMOTION_POSE_S)

		elif state == "goodbye":
			self.traj = moves.Trajectory(start, moves.neutral(), GOODBYE_S)

	def _follow(self, t):

		# play the phase's move; returns True when it has arrived
		if self.traj is None:
			return True
		cfg, done = self.traj.sample(t)
		self.set_head(cfg)
		if done:
			self.base = cfg
		return done

	def _sound_fresh(self):

		return self.last_azim is not None and self.now() - self.t_last_azim < SOUND_FRESH_S

	# ---------------------------------------------------------------- phases

	def _service_facing(self, d, t):

		arrived = self._follow(t)

		if self.face_user == "apriltag":
			# rotate on the spot until the tag is seen steadily
			self.tags.append(1 if self._tag_seen() else 0)
			if sum(self.tags) >= FACE_HITS:
				print("[converse] found the user (AprilTag)")
				d.face_done()
				return
			self.body_velocity(0.0, self.face_rotate_speed)
		elif self.face_user == "sound":
			if arrived:
				d.face_done()
				return
		else:
			d.face_done()
			return

		if t > self.face_timeout_s:
			print("[converse] could not find the user; carrying on")
			d.face_done()

	def _tag_seen(self):

		# stock AprilTag detection: miro.msg.objects per camera, .tags list
		for msg in getattr(self.system_state, "detect_objects_for_50Hz", None) or []:
			if msg is not None and len(getattr(msg, "tags", None) or []) > 0:
				return True
		return False

	def _service_listening(self, d, t):

		self._follow(t)

		# nod once each time the user starts speaking
		if d.take_nods() > 0 and self.nod_t0 is None:
			self.nod_t0 = self.now()
		if self.nod_t0 is not None:
			tn = self.now() - self.nod_t0
			if tn >= LISTEN_NOD_PERIOD_S:
				self.nod_t0 = None
				self.set_head(self.base)
			else:
				self.set_head(moves.nod(self.base, tn, LISTEN_NOD_PERIOD_S, LISTEN_NOD_DEPTH))

	def _service_responding(self, d, t):

		reply = d.current_reply()
		emotion = getattr(reply, "emotion", None) or "fine"
		commands = getattr(reply, "commands", None) or {}

		# emotion pose, then gentle nodding while MiRo talks
		if t < EMOTION_POSE_S + EMOTION_HOLD_S or self.traj is None:
			self._follow(t)
		else:
			tn = t - EMOTION_POSE_S - EMOTION_HOLD_S
			self.set_head(moves.nod(self.base, tn, TALK_NOD_PERIOD_S, TALK_NOD_DEPTH))

		# LED colour of the emotion (HRI set_colour_info)
		rgb = lights.EMOTION_COLOURS.get(emotion, lights.EMOTION_COLOURS["fine"])
		self.override("illum", lights.solid(rgb))

		# flourishes
		if commands.get("spin") and t < self.spin_s:
			self.body_velocity(0.0, self.spin_speed)
		if commands.get("move_ears") and t < EAR_FLAPS / EAR_FLAP_HZ:
			self.override("ears", moves.ear_flap(t, EAR_FLAP_HZ))
		elif commands.get("impression"):
			# ears up and forward while doing the voice
			self.override("ears", [0.0, 0.0])
