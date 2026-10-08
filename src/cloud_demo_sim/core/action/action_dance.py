#
#	Hey MiRo - ActionDance: dance to a song.
#
#	The forebrain resolves the song and decodes it on its worker thread,
#	then posts a "dance" task ({"song": Song, "pcm": int16 24 kHz,
#	"intro": bool}). This action wants to run while that task is
#	pending, claims it on start, and then:
#
#	  intro   play a "dance_start" clip while the head rises to a happy
#	          pose (skipped silently if there is no clip)
#	  dance   play the music and, every tick, ask the DanceController
#	          for the frame at the current music position (so moves stay
#	          in sync with what is actually heard) and apply it: head,
#	          wheels, LEDs / ears / tail through NodeExpress overrides
#	  outro   head back to neutral, then end the task
#
#	It ends on the end of the song, after dance.max_duration_s (with a
#	1 s fade-out), when the forebrain clears the task ("Stop MiRo"), or
#	when a higher-priority action (e.g. halt at a cliff) pre-empts it.
#
#	Replaces HRI'25 action_dance.py: no Shazam recording loop, no
#	blocking audio calls, no Spotify calls or keys, no os.walk() for
#	the song folder; all of that lives in the forebrain / cloud modules.
#

from . import forebrain_action

from choreography import moves
from choreography.dance import DanceController, dance_affect

# NodeVoice music rate
AUDIO_RATE = 24000.0

# never wait longer than this for the intro clip to finish
INTRO_MAX_S = 6.0

# time to move the head into / out of the dance
POSE_S = 1.0

# music that has not started after this long counts as finished
MUSIC_START_GRACE_S = 1.0



def _task_name(pending):

	# NodeForebrain.pending_task() may return a name, a (name, params)
	# pair or an object with .name; accept all three
	if pending is None:
		return None
	if isinstance(pending, str):
		return pending
	if isinstance(pending, (tuple, list)):
		return pending[0] if pending else None
	if isinstance(pending, dict):
		return pending.get("name")
	return getattr(pending, "name", None)



class ActionDance(forebrain_action.ForebrainAction):

	NAME = "dance"

	def finalize(self):

		forebrain_action.ForebrainAction.finalize(self)
		self.name = self.NAME
		self.modulate_by_wakefulness = False

		# settings
		cfg = self.pars.forebrain.cfg
		self.max_duration_s = float(cfg.get("dance.max_duration_s", 60.0))
		self.default_bpm = float(cfg.get("dance.default_bpm", 120.0))
		self.use_wheels = bool(cfg.get("dance.wheels", True))
		self.allow_translation = bool(cfg.get("dance.allow_translation", True))
		try:
			self.tick_hz = float(self.pars.timing.tick_hz)
		except AttributeError:
			self.tick_hz = 50.0

		self._reset()

	def _reset(self):

		self.phase = None           # None | "intro" | "dance" | "outro"
		self.ticks = 0              # ticks spent in the current phase
		self.ctrl = None
		self.traj = None
		self.pcm = None
		self.song = None
		self.music_started = False
		self.task_tracked = False   # forebrain.active_task was "dance" at start
		self.end_reason = None

	# ---------------------------------------------------------------- nodes

	def _node(self, name):

		return getattr(self.parent.nodes, name, None)

	# ------------------------------------------------------------ selection

	def wanted(self):

		# keep wanting to run for the whole episode; otherwise only when
		# the forebrain has a dance waiting for us
		if self.phase is not None:
			return True
		fb = self._node("forebrain")
		if fb is None:
			return False
		return _task_name(fb.pending_task()) == self.NAME

	# ------------------------------------------------------------ lifecycle

	def start(self):

		fb = self._node("forebrain")
		params = fb.claim_task(self.NAME) if fb is not None else None
		if not params:
			# nothing to dance to (the task was cancelled or taken); do
			# not start the clock, so the action simply lapses
			return

		pcm = params.get("pcm")
		if pcm is None or len(pcm) == 0:
			print("[dance] no music to dance to")
			fb.end_task(self.NAME, "no music")
			return

		self._reset()
		self.song = params.get("song")
		self.pcm = pcm
		self.task_tracked = getattr(fb, "active_task", None) == self.NAME

		bpm = getattr(self.song, "bpm", None) or self.default_bpm
		genre = getattr(self.song, "genre", None) or "default"
		duration_s = min(len(pcm) / AUDIO_RATE, self.max_duration_s)

		# the dance is centred on the happy "light up" pose; the intro
		# moves the head there so the first beat starts without a jump
		home = moves.light_up()
		self.ctrl = DanceController(bpm, genre, duration_s, seed=None,
			wheels=self.use_wheels, allow_translation=self.allow_translation, home=home)
		self.traj = moves.Trajectory(self.kc.getConfig(), home, POSE_S)

		title = getattr(self.song, "title", "?")
		print("[dance] start: %s (%.0f bpm, %s, %.1f s)" % (title, self.ctrl.bpm, self.ctrl.genre, duration_s))

		# pleased, and more excited the faster the song
		affect = self._node("affect")
		if affect is not None:
			valence, arousal = dance_affect(bpm, genre)
			affect.set_forebrain_target(valence, arousal, seconds=min(10.0, max(2.0, duration_s)))

		cues = self._node("cues")
		if cues is not None:
			cues.set_mode("dancing")

		voice = self._node("voice")
		if params.get("intro", True) and voice is not None:
			voice.play_clip("dance_start")

		self.phase = "intro"
		self.ticks = 0
		self.clock.start(int(self.tick_hz))

	def service(self):

		# hold the clock open: the episode ends when we say so
		self.hold()
		self.ticks += 1
		t_phase = self.ticks / self.tick_hz

		# "Stop MiRo" (forebrain.stop_all) clears the active task
		fb = self._node("forebrain")
		if self.task_tracked and fb is not None and getattr(fb, "active_task", None) != self.NAME:
			self._finish("stopped")
			return

		voice = self._node("voice")

		if self.phase == "intro":
			cfg, _ = self.traj.sample(t_phase)
			self.set_head(cfg)
			speaking = voice is not None and voice.is_speaking()
			if not speaking or t_phase > INTRO_MAX_S:
				if voice is not None:
					voice.play_music(self.pcm)
					self.music_started = True
				self.phase = "dance"
				self.ticks = 0

		elif self.phase == "dance":
			t_music = voice.music_position_s() if voice is not None else t_phase
			frame = self.ctrl.step(t_music)
			music_over = self.music_started and t_phase > MUSIC_START_GRACE_S \
				and not voice.is_music_playing()
			if frame.done or music_over:
				self._begin_outro()
				return
			self._apply(frame)

		elif self.phase == "outro":
			cfg, done = self.traj.sample(t_phase)
			self.set_head(cfg)
			if done:
				self._finish("done")

	def _apply(self, frame):

		self.set_head(frame.head)
		if frame.wheels is not None:
			self.body_velocity(frame.wheels[0], frame.wheels[1])

		express = self._node("express")
		if express is not None:
			express.override("illum", frame.illum, owner=self.name)
			if "ears" in frame.cosmetics:
				express.override("ears", frame.cosmetics["ears"], owner=self.name)
			if "tail" in frame.cosmetics:
				express.override("tail", frame.cosmetics["tail"], owner=self.name)

		# eyelids belong to NodeCues ("dancing" mode); hand it the
		# openness for this song if it takes it, else drive them directly
		lids = frame.cosmetics.get("eyelids")
		if lids is not None:
			cues = self._node("cues")
			if cues is None:
				if express is not None:
					express.override("eyelids", lids, owner=self.name)
			elif hasattr(cues, "set_dance_eyelids"):
				cues.set_dance_eyelids(lids)

	def _begin_outro(self):

		# stop the music (if the song outlasted max_duration_s) and let the
		# stock express routine have the LEDs back while the head settles
		voice = self._node("voice")
		if voice is not None and voice.is_music_playing():
			voice.stop("music")
		self.music_started = False
		express = self._node("express")
		if express is not None:
			express.release(self.name)
		self.traj = moves.Trajectory(self.kc.getConfig(), moves.neutral(), POSE_S)
		self.phase = "outro"
		self.ticks = 0

	def _finish(self, reason):

		self.end_reason = reason
		self.stop()

	def stop(self):

		# called by us when the dance ends, and by the stock action loop
		# when we are pre-empted
		reason = self.end_reason or "preempted"
		if self.phase is not None:
			print("[dance] end (" + reason + ")")
			voice = self._node("voice")
			if voice is not None and self.music_started and voice.is_music_playing():
				voice.stop("music")
			cues = self._node("cues")
			if cues is not None:
				cues.set_mode("")
		self._reset()

		# base: releases our express overrides and ends the task
		forebrain_action.ForebrainAction.stop(self)
		self.clock.stop()

		# make sure the task is closed even if the base did not do it
		fb = self._node("forebrain")
		if fb is not None and getattr(fb, "active_task", None) == self.NAME:
			fb.end_task(self.NAME, reason)
