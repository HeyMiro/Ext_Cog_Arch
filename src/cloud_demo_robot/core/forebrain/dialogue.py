#
#	Hey MiRo - the conversation state machine (forebrain, pure).
#
#	One conversation ("episode") runs from the wake word to the goodbye:
#
#	  IDLE --wake--> FACING --> GREETING --> LISTENING --utterance--> THINKING
#	                 (turn to                (segmenter   (hmm + loading tune,
#	                  the user)               armed)       one cloud job)
#	                                              ^            |
#	                                              |            v
#	                                              +------- RESPONDING (speech)
#
#	  LISTENING --no_speech--> GOODBYE --> IDLE
#	  any engaged state --stop word--> GOODBYE
#	  THINKING/RESPONDING --dance/tricks/identify--> SUSPENDED --resume--> GREETING
#
#	The HRI'25 code did all of this inside a ROS mic callback and the
#	50 Hz action tick, blocking on Whisper / GPT / ElevenLabs. Here the
#	machine never blocks: the cloud turn (wav -> speech-to-text -> chat ->
#	parse -> text-to-speech) is ONE job on the CloudWorker and the tick
#	only polls its Future. Everything the machine needs from the outside
#	(clock, hearing, voice, job submission) is injected, so it can be
#	unit tested with fakes; it reports what the node should do through an
#	outbox of (name, payload) events:
#
#	  ("started", episode)      a conversation began
#	  ("nod", None)             the user started speaking (nod to show we listen)
#	  ("reply", LlmReply)       a reply was accepted (affect target, LEDs, flourishes)
#	  ("remember", (user, raw)) store this exchange in the chat memory
#	  ("task", (name, params))  hand over to dance / tricks / identify
#	  ("sleep", None)           the user asked MiRo to go to sleep
#	  ("ended", episode)        the conversation is over (clear the memory)
#
#	Stale results: every job carries a token (episode, turn); a Future
#	whose token is no longer current (the user said "stop", a task took
#	over, the conversation ended and a new one began) is ignored.
#

from cloud.llm import parse_llm_reply, context_message
from cloud.worker import describe_error
import audio_util


# states (strings, so they read well in logs and in core/forebrain/state)
IDLE = "idle"
FACING = "facing"
GREETING = "greeting"
LISTENING = "listening"
THINKING = "thinking"
RESPONDING = "responding"
GOODBYE = "goodbye"
SUSPENDED = "suspended"

STATES = (IDLE, FACING, GREETING, LISTENING, THINKING, RESPONDING, GOODBYE, SUSPENDED)

# reply commands that hand the robot over to another behaviour, and the
# forebrain task each one becomes
TASK_COMMANDS = (("dance", "dance"), ("tricks", "tricks"), ("identify_object", "identify"))

# extra time allowed after a clip's nominal length before we stop
# waiting for voice.is_speaking() to clear (robot speaker latency)
SPEECH_SLACK_S = 5.0

# how long a greeting / resume prompt waits for MiRo to finish saying
# something else (an "ouch", the end of a task) before talking over it
QUIET_WAIT_S = 8.0

# a speech channel needs a moment to report busy after we start it
MIN_SPEECH_S = 0.2

# FACING normally ends when the action says so; this is the backstop
# if the action never runs (pre-empted, no cameras)
FACE_SLACK_S = 2.0

# Whisper sometimes "hears" these in near-silence; treat them as nothing
NON_SPEECH = ("", ".", "you", "thank you.", "thanks for watching!", "bye.")



class TurnResult(object):

	"""What one cloud turn produced (built on the worker thread)."""

	def __init__(self, transcript="", reply=None, speech=None, raw="", tts_error=None):

		self.transcript = transcript      # what the user said ("" = nothing)
		self.reply = reply                # LlmReply or None
		self.speech = speech              # int16 24 kHz ndarray or None (no TTS)
		self.raw = raw                    # the model's raw answer (for the memory)
		self.tts_error = tts_error        # sanitized reason TTS failed, if it did



class TurnRunner(object):

	"""
	One conversation turn, run on the CloudWorker:
	pcm (16 kHz) -> wav -> speech-to-text -> chat -> parse -> text-to-speech.

	clients is a cloud.llm.OpenAIClients, tts a cloud.tts.TextToSpeech (or
	None: the reply is then printed and a clip played instead), memory a
	cloud.llm.ChatMemory (read here, written on the tick when the result
	is accepted, so a stale turn can never pollute it).
	"""

	def __init__(self, clients, tts, memory, system_prompt, cfg):

		self.clients = clients
		self.tts = tts
		self.memory = memory
		self.system_prompt = system_prompt
		self.stt_model = cfg.get("stt.model", "whisper-1")
		self.language = cfg.get("stt.language", None)
		self.llm_model = cfg.get("llm.model", "gpt-4o")
		self.temperature = cfg.get("llm.temperature", None)
		self.max_tokens = cfg.get("llm.max_tokens", None)
		self.json_mode = bool(cfg.get("llm.json_mode", True))
		self.impression_voice_id = cfg.get("tts.impression_voice_id", None)

	def __call__(self, pcm16k, valence=None, arousal=None, now_str=None):

		if self.clients is None:
			raise RuntimeError("conversation is not configured (no OpenAI route)")

		# speech to text
		wav = audio_util.wav_bytes(audio_util.to_int16(pcm16k), audio_util.HEARING_RATE)
		text = (self.clients.transcribe(wav, self.stt_model, self.language) or "").strip()
		if text.lower() in NON_SPEECH:
			return TurnResult(transcript="")

		# chat (the context message is ephemeral: never stored)
		context = context_message(valence, arousal, now_str)
		messages = self.memory.messages(self.system_prompt, context, text)
		raw = self.clients.chat(messages, self.llm_model, temperature=self.temperature,
			max_tokens=self.max_tokens, json_mode=self.json_mode)
		reply = parse_llm_reply(raw)

		# a dance hands straight over to the dance intro (as in HRI'25),
		# so there is nothing to synthesize
		speech = None
		tts_error = None
		if self.tts is not None and reply.reply and not reply.commands.get("dance"):
			voice_id = self.impression_voice_id if reply.commands.get("impression") else None
			try:
				speech = self.tts.synthesize(reply.reply, voice_id)
			except Exception as e:
				# the reply is still worth having: it is printed and a clip played
				tts_error = describe_error(e)
		return TurnResult(text, reply, speech, raw, tts_error)



class DialogueMachine(object):

	"""
	The conversation state machine. Injected collaborators:

	  clock()                     seconds (monotonic)
	  hearing.arm_utterance()     start waiting for the user to speak
	  hearing.cancel_utterance()
	  voice.play_clip(category, channel="speech") -> duration s
	  voice.say_pcm(pcm24k) -> duration s
	  voice.stop(channel=None)
	  voice.is_speaking()
	  submit_turn(pcm16k) -> concurrent.futures.Future of TurnResult

	Feed it on_wake(), on_hearing(name, payload), on_stop(), face_done(),
	suspend(task) and resume(); call tick() every control tick; drain
	take_outbox() after each tick.
	"""

	def __init__(self, clock, hearing, voice, submit_turn,
			face_user="none", face_timeout_s=8.0, thinking_music=True,
			turn_timeout_s=65.0, listen_timeout_s=40.0, greet_min_s=1.0,
			log=print, log_transcripts=False):

		self.clock = clock
		self.hearing = hearing
		self.voice = voice
		self.submit_turn = submit_turn
		self.face_user = str(face_user or "none")
		self.face_timeout_s = float(face_timeout_s)
		self.thinking_music = bool(thinking_music)
		self.turn_timeout_s = float(turn_timeout_s)
		self.listen_timeout_s = float(listen_timeout_s)
		self.greet_min_s = float(greet_min_s)
		self.log = log
		self.log_transcripts = bool(log_transcripts)

		self.state = IDLE
		self.t_state = clock()
		self.episode = 0
		self.turn = 0
		self.reply = None            # last accepted LlmReply (for the motor side)
		self.outbox = []

		# per-state working variables
		self.job = None              # (token, Future) while THINKING
		self.deadline = None
		self.music_started = False
		self.faced = False
		self.prompt = None           # pcm for GREETING (None = greeting clip)
		self.prompt_started = False
		self.t_prompt = 0.0
		self.speech_dur = 0.0
		self.task_after = None       # (name, params) to hand over after speaking
		self.sleep_after = False

	# ---------------------------------------------------------------- queries

	def engaged(self):

		# in a conversation that wants the robot's body (ActionConverse)
		return self.state not in (IDLE, SUSPENDED)

	def active(self):

		# in a conversation at all (including while a task has the body)
		return self.state != IDLE

	def state_age(self):

		return self.clock() - self.t_state

	def take_outbox(self):

		out = self.outbox
		self.outbox = []
		return out

	# ----------------------------------------------------------------- events

	def on_wake(self):

		# "Hey MiRo": start a conversation (ignored if one is running)
		if self.state != IDLE:
			return False
		self.episode += 1
		self.reply = None
		self.sleep_after = False
		self.task_after = None
		self._emit("started", self.episode)
		if self.face_user in ("apriltag", "sound"):
			self._enter(FACING)
		else:
			self._enter_greeting(None)
		return True

	def face_done(self):

		# ActionConverse found the user (or gave up looking)
		if self.state == FACING:
			self.faced = True

	def on_hearing(self, name, payload=None):

		# events from NodeHearing; only meaningful while listening
		if self.state != LISTENING:
			return
		if name == "voiced":
			self._emit("nod", None)
		elif name == "utterance":
			self._enter_thinking(payload)
		elif name == "no_speech":
			self.log("[dialogue] nobody spoke: goodbye")
			self._enter_goodbye()

	def on_stop(self):

		# "Stop MiRo" (or the forebrain ending everything): say goodbye
		if self.state in (IDLE, GOODBYE):
			return
		self.log("[dialogue] stop")
		self._enter_goodbye()

	def suspend(self, task=None):

		# a task (dance, tricks, identify) takes the body; keep the
		# conversation (and its memory) to come back to afterwards
		if self.state in (IDLE, SUSPENDED):
			return
		self.log("[dialogue] suspended for " + str(task))
		self.hearing.cancel_utterance()
		self.voice.stop("music")
		self._drop_job()
		self._enter(SUSPENDED)

	def resume(self, prompt_pcm=None):

		# back from a task: "that was fun, do you want to keep talking?"
		if self.state != SUSPENDED:
			return
		self.log("[dialogue] resumed")
		self._enter_greeting(prompt_pcm)

	def abort(self):

		# end silently (shutdown)
		if self.state != IDLE:
			self.hearing.cancel_utterance()
			self._drop_job()
			self._end()

	# ------------------------------------------------------------------- tick

	def tick(self):

		now = self.clock()
		age = now - self.t_state

		if self.state == FACING:
			if self.faced or age > self.face_timeout_s + FACE_SLACK_S:
				self._enter_greeting(None)

		elif self.state == GREETING:
			self._tick_greeting(now, age)

		elif self.state == LISTENING:
			# backstop for a hearing pipeline that never answers (no mic data)
			if age > self.listen_timeout_s:
				self.log("[dialogue] no answer from hearing: goodbye")
				self._enter_goodbye()

		elif self.state == THINKING:
			self._tick_thinking(now)

		elif self.state == RESPONDING:
			if self._speech_finished(age):
				if self.task_after is not None:
					name, params = self.task_after
					self.task_after = None
					self._emit("task", (name, params))
					self._enter(SUSPENDED)
				elif self.sleep_after:
					self._enter_goodbye()
				else:
					self._enter_listening()

		elif self.state == GOODBYE:
			if self._speech_finished(age):
				self._end()

	def _tick_greeting(self, now, age):

		if not self.prompt_started:
			# do not talk over something MiRo is still saying
			if self.voice.is_speaking() and age < QUIET_WAIT_S:
				return
			if self.prompt is not None:
				self.speech_dur = self.voice.say_pcm(self.prompt)
			else:
				self.speech_dur = self.voice.play_clip("greeting")
			self.prompt_started = True
			self.t_prompt = now
			return
		t = now - self.t_prompt
		# at least greet_min_s, so the lean-in move can finish
		if (t >= self.greet_min_s and not self.voice.is_speaking()) or t > self.speech_dur + SPEECH_SLACK_S:
			self._enter_listening()

	def _tick_thinking(self, now):

		# "hmm" first, then the loading tune while the cloud works
		speaking = self.voice.is_speaking()
		if self.thinking_music and not self.music_started and not speaking:
			self.voice.play_clip("loading", channel="music")
			self.music_started = True

		if self.job is None:
			return
		token, future = self.job

		if not future.done():
			if now > self.deadline:
				self.log("[dialogue] the cloud took too long (> %.0f s)" % self.turn_timeout_s)
				self._drop_job()
				self._acknowledge_and_listen()
			return

		# let the "hmm" finish before answering
		if speaking:
			return
		self.job = None
		if token != (self.episode, self.turn):
			return
		try:
			result = future.result()
		except Exception as e:
			self.log("[dialogue] turn failed: " + describe_error(e))
			self._acknowledge_and_listen()
			return
		self._accept(result)

	def _accept(self, result):

		if not result.transcript:
			self.log("[dialogue] heard nothing intelligible; listening again")
			self.voice.stop("music")
			self._enter_listening()
			return

		reply = result.reply
		if self.log_transcripts:
			self.log("[dialogue] user: " + result.transcript)
		self.reply = reply
		self._emit("remember", (result.transcript, result.raw or (reply.reply if reply else "")))
		if reply is None:
			self._acknowledge_and_listen()
			return
		self._emit("reply", reply)
		if result.tts_error:
			self.log("[dialogue] text-to-speech failed (" + result.tts_error + ")")

		commands = reply.commands or {}
		if commands.get("sleep"):
			self.sleep_after = True

		task = None
		for command, name in TASK_COMMANDS:
			if commands.get(command):
				params = {"intro": False}
				if name == "dance":
					params["query"] = commands.get("song")
				task = (name, params)
				break

		if task is not None and task[0] == "dance":
			# straight into the dance intro clip, as in HRI'25
			self.voice.stop("music")
			self._emit("task", task)
			self._enter(SUSPENDED)
			return

		self.task_after = task
		self._enter_responding(result.speech, reply)

	# ------------------------------------------------------------ transitions

	def _emit(self, name, payload):

		self.outbox.append((name, payload))

	def _enter(self, state):

		self.state = state
		self.t_state = self.clock()
		self.faced = False

	def _enter_greeting(self, prompt_pcm):

		self.prompt = prompt_pcm
		self.prompt_started = False
		self.speech_dur = 0.0
		self._enter(GREETING)

	def _enter_listening(self):

		self.hearing.arm_utterance()
		self._enter(LISTENING)

	def _enter_thinking(self, pcm):

		self.hearing.cancel_utterance()
		self.voice.play_clip("thinking")
		self.music_started = False
		self.turn += 1
		token = (self.episode, self.turn)
		self.job = (token, self.submit_turn(pcm))
		self.deadline = self.clock() + self.turn_timeout_s
		self._enter(THINKING)

	def _enter_responding(self, speech, reply):

		self.voice.stop("music")
		if speech is not None and len(speech) > 0:
			self.speech_dur = self.voice.say_pcm(speech)
		else:
			# no voice for the reply (TTS off or failed): print it and
			# make a friendly noise so the user knows MiRo answered
			self.log("[dialogue] MiRo says: " + (reply.reply if reply else ""))
			self.speech_dur = self.voice.play_clip("acknowledge")
		self._enter(RESPONDING)

	def _enter_goodbye(self):

		self.hearing.cancel_utterance()
		self._drop_job()
		self.voice.stop("music")
		self.task_after = None
		self.speech_dur = self.voice.play_clip("goodbye")
		self._enter(GOODBYE)

	def _acknowledge_and_listen(self):

		# a short "hmm" instead of an answer, then listen again
		self.voice.stop("music")
		self.voice.play_clip("acknowledge")
		self._enter_listening()

	def _drop_job(self):

		# the Future keeps running on the worker; its result is ignored
		self.job = None
		self.turn += 1

	def _speech_finished(self, age):

		if age < MIN_SPEECH_S:
			return False
		return not self.voice.is_speaking() or age > self.speech_dur + SPEECH_SLACK_S

	def _end(self):

		episode = self.episode
		sleep = self.sleep_after
		self.sleep_after = False
		self.task_after = None
		self._enter(IDLE)
		if sleep:
			self._emit("sleep", None)
		self._emit("ended", episode)
