#
#	Hey MiRo - NodeDialogue: the conversation manager.
#
#	Wraps the pure state machine in forebrain/dialogue.py and connects it
#	to the rest of the robot:
#
#	  - hearing events arrive through NodeForebrain (on_hearing)
#	  - each user utterance becomes one CloudWorker job (TurnRunner):
#	    speech-to-text -> chat (persona + history + current affect) ->
#	    reply parser -> text-to-speech
#	  - an accepted reply pulls MiRo's affect toward the reply's valence
#	    and arousal (HRI'25 wrote these to attributes nobody read), may
#	    send MiRo to sleep (HRI'25 parsed but ignored this), and may hand
#	    over to a task (dance / tricks / identify) through the forebrain
#	  - ActionConverse reads the state, the reply and the nod requests to
#	    move the body; it reports back with face_done()
#
#	The chat memory holds the user/assistant turns of the current
#	conversation only; it is cleared when the conversation ends.
#

import datetime
import time

import node
from cloud.llm import ChatMemory, EMOTION_CENTRES
from forebrain import dialogue as dlg
from node_forebrain import RESUME_PROMPT

# HRI'25 used the time in the prompt; this is the format we send
TIME_FORMAT = "%A %d %B %Y, %H:%M"



class NodeDialogue(node.Node):

	def __init__(self, sys, clock=None):

		node.Node.__init__(self, sys, "dialogue")

		heymiro = self.pars.forebrain
		cfg = heymiro.cfg
		self.cfg = cfg
		self.features = heymiro.features
		self.clock = clock if clock is not None else time.monotonic
		self.log_transcripts = bool(self.features.get("log_transcripts", False))

		# cloud: the forebrain node (created before us) owns the clients
		forebrain = self.nodes.forebrain
		cloud = forebrain.cloud
		self.memory = ChatMemory(cfg.get("llm.max_history_turns", 10))
		self.runner = dlg.TurnRunner(cloud.clients, cloud.tts, self.memory, cloud.persona, cfg)

		# a turn may take as long as its three cloud calls together
		turn_timeout_s = float(cfg.get("stt.timeout_s", 20)) + float(cfg.get("llm.timeout_s", 25)) \
			+ float(cfg.get("tts.timeout_s", 20))

		# listening normally ends with an utterance or no_speech from the
		# segmenter; this backstop only matters if no mic data arrives
		listen_timeout_s = float(cfg.get("hearing.no_speech_timeout_s", 3.0)) \
			+ float(cfg.get("hearing.max_utterance_s", 20.0)) \
			+ float(cfg.get("hearing.end_silence_s", 3.0)) + 10.0

		self.machine = dlg.DialogueMachine(
			clock=self.clock,
			hearing=self.nodes.hearing,
			voice=self.nodes.voice,
			submit_turn=self._submit_turn,
			face_user=cfg.get("converse.face_user", "none"),
			face_timeout_s=cfg.get("converse.face_timeout_s", 8.0),
			thinking_music=cfg.get("voice.thinking_music", True),
			turn_timeout_s=turn_timeout_s,
			listen_timeout_s=listen_timeout_s,
			log_transcripts=self.log_transcripts)

		# voiced onsets not yet nodded to (ActionConverse takes them)
		self.nods = 0

	# --------------------------------------------------------------- queries

	def engaged(self):

		# talking with someone (the conversation wants the body)
		return self.machine.engaged()

	def active(self):

		# in a conversation, including while a task has the body
		return self.machine.active()

	def suspended(self):

		return self.machine.state == dlg.SUSPENDED

	def state_name(self):

		return self.machine.state

	def state_age(self):

		return self.machine.state_age()

	def current_reply(self):

		return self.machine.reply

	def take_nods(self):

		n = self.nods
		self.nods = 0
		return n

	# ---------------------------------------------------------------- events

	def on_wake(self):

		return self.machine.on_wake()

	def on_hearing(self, name, payload=None):

		self.machine.on_hearing(name, payload)

	def on_stop(self):

		self.machine.on_stop()

	def face_done(self):

		self.machine.face_done()

	def suspend_for(self, task):

		self.machine.suspend(task)

	def resume(self):

		# the "keep talking?" line if it has been synthesized, else a
		# greeting clip
		self.machine.resume(self.nodes.forebrain.cached_speech(RESUME_PROMPT))

	# ------------------------------------------------------------------ cloud

	def _submit_turn(self, pcm):

		# snapshot everything the worker needs now, on the tick
		valence = arousal = None
		affect = getattr(self.nodes, "affect", None)
		emotion = getattr(affect, "emotion", None)
		if emotion is not None:
			valence = float(emotion.valence)
			arousal = float(emotion.arousal)
		now_str = datetime.datetime.now().strftime(TIME_FORMAT)
		return self.nodes.forebrain.worker.submit(self.runner, pcm, valence, arousal, now_str)

	# ------------------------------------------------------------------- tick

	def tick(self):

		self.machine.tick()
		for name, payload in self.machine.take_outbox():
			self._handle(name, payload)

	def _handle(self, name, payload):

		forebrain = self.nodes.forebrain
		affect = getattr(self.nodes, "affect", None)

		if name == "started":
			print("[dialogue] conversation " + str(payload) + " started")

		elif name == "nod":
			self.nods += 1

		elif name == "remember":
			user, assistant = payload
			self.memory.add(user, assistant)

		elif name == "reply":
			reply = payload
			print("[dialogue] reply (" + str(reply.emotion) + "): " + reply.reply)
			valence, arousal = reply.valence, reply.arousal
			if (valence is None or arousal is None) and reply.emotion in EMOTION_CENTRES:
				centre = EMOTION_CENTRES[reply.emotion]
				valence = centre[0] if valence is None else valence
				arousal = centre[1] if arousal is None else arousal
			if affect is not None and (valence is not None or arousal is not None):
				affect.set_forebrain_target(valence, arousal)

		elif name == "task":
			task, params = payload
			if task == "dance":
				forebrain.prepare_dance(params.get("query"))
			elif not forebrain.request_task(task, params):
				self.resume()
			forebrain.prefetch(RESUME_PROMPT)

		elif name == "sleep":
			if self.features.get("sleep_command", False) and affect is not None:
				print("[dialogue] going to sleep")
				affect.request_sleep()

		elif name == "ended":
			print("[dialogue] conversation " + str(payload) + " ended")
			self.memory.clear()

	def shutdown(self):

		self.machine.abort()
		self.machine.take_outbox()

