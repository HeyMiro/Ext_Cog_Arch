#
#	Hey MiRo - NodeForebrain: top-down modulation and the task board.
#
#	The forebrain (P3) sits above the stock brainstem demo. Each tick it
#
#	  - reads what was heard (NodeHearing events) and what was asked on
#	    the core/forebrain/command topic, and maps it to behaviour:
#	      "Hey MiRo"   -> start a conversation (NodeDialogue), wake up
#	      "Dance MiRo" -> pick a song and dance
#	      "Stop MiRo"  -> stop everything
#	      ear / tail scratching (mic noise) -> "Ouch!" + a negative affect target
#	      petting (head touch)              -> pleased noises, now and then
#	  - keeps ONE pending task slot (dance / tricks / identify): a task is
#	    requested here (or by the dialogue), claimed by its action when the
#	    basal ganglia select it, and ended by that action; a conversation
#	    that handed over to the task is resumed afterwards
#	  - modulates the lower layers: which wake words are listened for,
#	    what the scene client should look for, how the eyelids behave
#
#	It owns the CloudWorker (no blocking work ever runs in tick()) and the
#	cloud clients, shared by the dialogue and the actions via self.cloud.
#
#	Command topic (std_msgs/String on core/forebrain/command), useful in
#	the simulator and without a Picovoice key:
#	  converse | dance [song] | tricks | identify | stop | say <text>
#

import collections
import json
import queue
import random
import time

import node
import audio_util
from cloud.worker import CloudWorker, describe_error


# a requested task nobody claims within this long is dropped (e.g. the
# robot was picked up and halt kept the action from being selected)
TASK_CLAIM_TIMEOUT_S = 5.0

# pleased noise while being petted: at most this often, and only with
# this probability per petted tick (HRI'25: 1 in 20, but with no limit)
PURR_COOLDOWN_S = 6.0
PURR_PROB = 0.05

# speech synthesized for say() is played if it arrives within this long
SAY_MAX_WAIT_S = 8.0
SPEECH_CACHE_SIZE = 32

# a gesture older than this is not "being made now"
GESTURE_FRESH_S = 0.5

# what the dialogue says when a task hands the conversation back
RESUME_PROMPT = "That was fun! Do you want to keep talking?"

COMMANDS = ("converse", "dance", "tricks", "identify", "stop", "say")

# feature switch that has to be on for each task
TASK_FEATURES = {"dance": "dance", "tricks": "tricks", "identify": "identify"}



def parse_command(text):

	"""
	"dance abba" -> ("dance", "abba"); "STOP" -> ("stop", "");
	anything unknown -> (None, text). Arguments keep their case
	("say Hello" says "Hello").
	"""

	text = (text or "").strip()
	if not text:
		return None, ""
	parts = text.split(None, 1)
	verb = parts[0].lower()
	arg = parts[1].strip() if len(parts) > 1 else ""
	if verb in ("hey", "hello", "talk", "chat"):
		verb = "converse"
	if verb not in COMMANDS:
		return None, text
	return verb, arg



class CloudServices(object):

	"""The off-board (P4) clients, shared by the forebrain nodes and actions."""

	def __init__(self):

		self.clients = None     # cloud.llm.OpenAIClients (speech-to-text + chat)
		self.tts = None         # cloud.tts.TextToSpeech
		self.songs = None       # cloud.music.SongLibrary
		self.spotify = None     # cloud.music.SpotifyClient
		self.persona = ""       # system prompt for the conversation



def make_cloud_services(heymiro):

	# build what the configuration and the available keys allow; a
	# failure only turns that part off (never print exception text: it
	# can contain URLs or keys)
	cloud = CloudServices()
	cfg = heymiro.cfg
	features = heymiro.features

	try:
		from cloud.llm import OpenAIClients, build_system_prompt
		persona = ""
		try:
			persona = heymiro.persona()
		except (IOError, OSError):
			print("[forebrain] WARN persona file not found; using a plain persona")
		cloud.persona = build_system_prompt(persona, cfg.get("llm.event_name", ""))
		if features.get("conversation", False):
			cloud.clients = OpenAIClients(heymiro.route("stt"), heymiro.route("llm"))
	except Exception as e:
		print("[forebrain] conversation unavailable (" + describe_error(e) + ")")

	if features.get("tts", False):
		try:
			from cloud.tts import TextToSpeech
			cloud.tts = TextToSpeech(heymiro.route("tts"), cfg.get("tts"))
		except Exception as e:
			print("[forebrain] text-to-speech unavailable (" + describe_error(e) + ")")

	if features.get("dance", False):
		try:
			from cloud.music import SongLibrary
			cloud.songs = SongLibrary(heymiro)
		except Exception as e:
			print("[forebrain] song library unavailable (" + describe_error(e) + ")")

	if features.get("spotify", False):
		try:
			from cloud.music import SpotifyClient
			cloud.spotify = SpotifyClient(heymiro.route("spotify"))
		except Exception as e:
			print("[forebrain] Spotify unavailable (" + describe_error(e) + ")")

	return cloud



class NodeForebrain(node.Node):

	def __init__(self, sys, clock=None, wall=None, worker=None, cloud=None, rng=None):

		node.Node.__init__(self, sys, "forebrain")

		heymiro = self.pars.forebrain
		self.heymiro = heymiro
		self.cfg = heymiro.cfg
		self.features = heymiro.features

		# time: monotonic for our own timers, wall clock to compare with
		# the receive stamps of the scene client messages
		self.clock = clock if clock is not None else time.monotonic
		self.wall = wall if wall is not None else time.time
		self.rng = rng if rng is not None else random.Random()

		# off-board
		self.worker = worker if worker is not None else CloudWorker(2)
		self.cloud = cloud if cloud is not None else make_cloud_services(heymiro)

		# commands from the ROS callback thread
		self.commands = queue.Queue()

		# task board
		self._pending = None             # (name, params, t_request)
		self.active_task = None
		self._dance_job = None           # (token, Future) while resolving / decoding a song
		self._dance_token = 0
		self._resume_reason = None       # resume a suspended conversation (see _poll_resume)

		# speech synthesized on demand (say(), tricks, identify, resume prompt)
		self.speech_cache = collections.OrderedDict()
		self._tts_jobs = {}
		self._say_pending = None         # (text, t_request)

		# touch reactions (ear / tail scratching heard by the mics)
		tn = self.cfg.get("hearing.touch_noise")
		self.hits_to_react = int(tn.get("hits_to_react", 4)) if tn else 4
		self.touch_window_s = float(tn.get("window_s", 2.0)) if tn else 2.0
		self.touch_cooldown_s = float(tn.get("cooldown_s", 3.0)) if tn else 3.0
		self.touch_hits = {"ear": collections.deque(), "tail": collections.deque()}
		self.touch_ready_at = 0.0
		self.purr_ready_at = 0.0

		# tricks without being asked (off by default: HRI'25 reacted to
		# every hand it saw, even mid-conversation)
		self.min_gesture_score = float(self.cfg.get("tricks.min_score", 0.6))
		self.trick_map = self.cfg.get("tricks.map")
		self.trick_map = self.trick_map.to_dict() if self.trick_map is not None else {}
		self._last_gesture_t = None

		# outputs of the last tick (only re-sent on change)
		self.keyword_mask = "unset"
		self.scene_mode = "off"
		self.cue_mode = ""
		self._last_state_json = None

	# ------------------------------------------------------------- helpers

	def _node(self, name):

		# nodes are created in order (dialogue and cues after us), so look
		# them up when needed; tests may leave some out
		return getattr(self.nodes, name, None)

	def _feature(self, name):

		return bool(self.features.get(name, False))

	# ---------------------------------------------------- ROS callback thread

	def on_command(self, text):

		# thread-safe: only enqueue, the tick acts on it
		self.commands.put(str(text))

	# ------------------------------------------------------------ task board

	def request_task(self, name, params=None):

		feature = TASK_FEATURES.get(name)
		if feature is None or not self._feature(feature):
			print("[forebrain] task '" + str(name) + "' is not enabled")
			return False
		if self._pending is not None:
			print("[forebrain] task '" + self._pending[0] + "' replaced by '" + name + "'")
		self._pending = (name, dict(params or {}), self.clock())
		print("[forebrain] task requested: " + name)

		# a running conversation yields the body to the task
		dialogue = self._node("dialogue")
		if dialogue is not None and dialogue.engaged():
			dialogue.suspend_for(name)
			self.prefetch(RESUME_PROMPT)
		return True

	def pending_task(self):

		return self._pending[0] if self._pending is not None else None

	def claim_task(self, name):

		# the action that won selection takes the task (once)
		if self._pending is None or self._pending[0] != name:
			return None
		params = self._pending[1]
		self._pending = None
		self.active_task = name
		print("[forebrain] task started: " + name)
		return params

	def end_task(self, name, reason="done"):

		ended = False
		if self.active_task == name:
			self.active_task = None
			ended = True
		if self._pending is not None and self._pending[0] == name:
			self._pending = None
			ended = True
		if ended:
			print("[forebrain] task ended: " + str(name) + " (" + str(reason) + ")")
		self._maybe_resume_dialogue(reason)

	def _maybe_resume_dialogue(self, reason):

		# hand the body back to a conversation that was waiting for it;
		# done from the tick, once any closing line of the task has been
		# synthesized (else the resume prompt and that line would collide)
		self._resume_reason = reason

	def _poll_resume(self):

		reason = self._resume_reason
		if reason is None:
			return
		if self.active_task is not None or self._pending is not None or self._dance_job is not None:
			# another task is on its way; the conversation keeps waiting
			self._resume_reason = None
			return
		if self.is_saying():
			return
		self._resume_reason = None
		dialogue = self._node("dialogue")
		if dialogue is None or not dialogue.suspended():
			return
		if reason == "stop":
			dialogue.on_stop()
		else:
			dialogue.resume()

	def engaged(self):

		dialogue = self._node("dialogue")
		return (dialogue is not None and dialogue.active()) or self.active_task is not None \
			or self._pending is not None or self._dance_job is not None

	# ----------------------------------------------------------------- dance

	def prepare_dance(self, query=None):

		# resolve the song and decode it on the worker; the "dance" task is
		# posted only when the audio is ready, so ActionDance never waits
		if not self._feature("dance"):
			print("[forebrain] dancing is not enabled")
			self._maybe_resume_dialogue("disabled")
			return False
		songs = self.cloud.songs
		if songs is None or not songs.songs:
			print("[forebrain] no songs to dance to (see config/songs.yaml)")
			self._maybe_resume_dialogue("no songs")
			return False
		self._dance_token += 1
		self._dance_job = (self._dance_token, self.worker.submit(self._resolve_song, query))
		dialogue = self._node("dialogue")
		if dialogue is not None and dialogue.engaged():
			dialogue.suspend_for("dance")
			self.prefetch(RESUME_PROMPT)
		return True

	def _resolve_song(self, query):

		# worker thread: find the song (manifest, then optionally Spotify
		# to turn a vague request into a title we have) and decode it
		from cloud.music import decode_mp3
		songs = self.cloud.songs
		song = songs.find(query) if query else None
		if song is None and query and self.cloud.spotify is not None:
			info = self.cloud.spotify.lookup(query)
			if info:
				song = songs.find(str(info.get("title", "")) + " " + str(info.get("artist", "")))
		found = song is not None
		if song is None:
			song = songs.random()
		max_s = float(self.cfg.get("dance.max_duration_s", 60.0)) + 2.0
		pcm = decode_mp3(song.path, audio_util.VOICE_RATE, max_s=max_s)
		return {"song": song, "pcm": pcm, "intro": True, "query": query, "found": found or not query}

	def _poll_dance_job(self):

		if self._dance_job is None:
			return
		token, future = self._dance_job
		if not future.done():
			return
		self._dance_job = None
		if token != self._dance_token:
			return
		try:
			params = future.result()
		except Exception as e:
			print("[forebrain] cannot prepare the song (" + describe_error(e) + ")")
			self._maybe_resume_dialogue("error")
			return
		if params.get("query") and not params.get("found"):
			print("[forebrain] song '" + params["query"] + "' not found; dancing to " + params["song"].title)
		self.request_task("dance", params)

	# ---------------------------------------------------------------- speech

	def prefetch(self, text, voice_id=None):

		# start synthesizing a phrase we will probably need soon
		text = (text or "").strip()
		if not text or self.cloud.tts is None:
			return
		if text in self.speech_cache or text in self._tts_jobs:
			return
		self._tts_jobs[text] = self.worker.submit(self.cloud.tts.synthesize, text, voice_id)

	def cached_speech(self, text):

		# pcm for text if it has been synthesized, else None
		self._poll_tts()
		return self.speech_cache.get((text or "").strip())

	def say(self, text):

		"""
		Say text with the TTS voice: at once if it is cached, else as soon
		as it has been synthesized. Without TTS the text is printed.
		Returns True if speech will be heard.
		"""

		text = (text or "").strip()
		if not text:
			return False
		pcm = self.cached_speech(text)
		voice = self._node("voice")
		if pcm is not None and voice is not None:
			voice.say_pcm(pcm)
			self._say_pending = None
			return True
		if self.cloud.tts is None:
			print("[forebrain] MiRo says (no TTS): " + text)
			return False
		self.prefetch(text)
		self._say_pending = (text, self.clock())
		return True

	def is_saying(self):

		# speech requested with say() that is still being synthesized
		return self._say_pending is not None

	def _poll_tts(self):

		for text in list(self._tts_jobs.keys()):
			future = self._tts_jobs[text]
			if not future.done():
				continue
			del self._tts_jobs[text]
			try:
				pcm = future.result()
			except Exception as e:
				print("[forebrain] text-to-speech failed (" + describe_error(e) + ")")
				continue
			if pcm is not None and len(pcm) > 0:
				self.speech_cache[text] = pcm
				while len(self.speech_cache) > SPEECH_CACHE_SIZE:
					self.speech_cache.popitem(last=False)

	def _poll_say(self, now):

		if self._say_pending is None:
			return
		text, t_request = self._say_pending
		pcm = self.speech_cache.get(text)
		if pcm is not None:
			self._say_pending = None
			if now - t_request <= SAY_MAX_WAIT_S:
				voice = self._node("voice")
				if voice is not None:
					voice.say_pcm(pcm)
		elif text not in self._tts_jobs:
			# synthesis failed (already reported)
			self._say_pending = None
			print("[forebrain] MiRo says: " + text)

	# -------------------------------------------------------------- behaviour

	def wake(self):

		# "Hey MiRo" (or the converse command)
		affect = self._node("affect")
		if affect is not None:
			affect.wake()
		cues = self._node("cues")
		if cues is not None:
			cues.perk()
		dialogue = self._node("dialogue")
		if self._feature("conversation") and dialogue is not None:
			dialogue.on_wake()
		elif not self.engaged():
			# no cloud: still show that we heard
			voice = self._node("voice")
			if voice is not None:
				voice.play_clip("greeting")

	def stop_all(self):

		# "Stop MiRo": silence, end the task (ActionDance & co notice the
		# cleared active_task), and say goodbye if we were talking
		print("[forebrain] stop")
		voice = self._node("voice")
		if voice is not None:
			voice.stop()
		self._say_pending = None
		self._dance_token += 1
		self._dance_job = None
		dialogue = self._node("dialogue")
		if dialogue is not None:
			dialogue.on_stop()
		self._pending = None
		if self.active_task is not None:
			print("[forebrain] task ended: " + self.active_task + " (stop)")
			self.active_task = None

	def _handle_command(self, text):

		verb, arg = parse_command(text)
		if verb is None:
			if arg:
				print("[forebrain] unknown command (use: " + " | ".join(COMMANDS) + ")")
			return
		print("[forebrain] command: " + verb)
		if verb == "converse":
			self.wake()
		elif verb == "dance":
			self.prepare_dance(arg or None)
		elif verb == "tricks":
			self.request_task("tricks", {"intro": True})
		elif verb == "identify":
			self.request_task("identify", {"intro": True})
		elif verb == "stop":
			self.stop_all()
		elif verb == "say":
			self.say(arg)

	def _on_wake_word(self, keyword):

		print("[forebrain] heard: " + str(keyword))
		if keyword == "hey_miro":
			self.wake()
		elif keyword == "dance_miro":
			if not self.engaged():
				affect = self._node("affect")
				if affect is not None:
					affect.wake()
				self.prepare_dance(None)
		elif keyword == "stop_miro":
			self.stop_all()

	def _on_touch(self, where, now):

		# HRI'25: four scratches within a short time -> "Ouch!"; only when
		# idle, because motion and speech also make noise on the mics
		hits = self.touch_hits.get(where)
		if hits is None:
			return
		hits.append(now)
		while hits and now - hits[0] > self.touch_window_s:
			hits.popleft()
		if len(hits) < self.hits_to_react or now < self.touch_ready_at or self.engaged():
			return
		hits.clear()
		self.touch_ready_at = now + self.touch_cooldown_s
		print("[forebrain] ouch (" + where + ")")
		voice = self._node("voice")
		if voice is not None and not voice.is_speaking():
			voice.play_clip("ouch_ear" if where == "ear" else "ouch_tail")
		affect = self._node("affect")
		if affect is not None:
			affect.set_forebrain_target(0.2, 0.8)

	def _petting(self, now):

		if not self._feature("purr") or now < self.purr_ready_at or self.engaged():
			return
		pet = getattr(self.state, "pet", 0) or 0
		stroke = getattr(self.state, "stroke", 0.0) or 0.0
		if pet <= 0 and abs(stroke) <= 0.0:
			return
		voice = self._node("voice")
		if voice is None or voice.is_speaking():
			return
		if self.rng.random() >= PURR_PROB:
			return
		self.purr_ready_at = now + PURR_COOLDOWN_S
		voice.play_clip("pleased")

	def _spontaneous_tricks(self):

		# optional: a hand sign while idle starts the tricks game
		if not (self._feature("spontaneous_gestures") and self._feature("tricks")):
			return
		if self.engaged():
			return
		gesture = self.fresh_gesture()
		if gesture is None or gesture.get("gesture") not in self.trick_map:
			return
		self.request_task("tricks", {"intro": False, "spontaneous": True})

	def fresh_gesture(self):

		# the latest gesture from the scene client if it is recent and
		# confident, else None (shared with ActionTricks)
		sg = getattr(self.state, "scene_gesture", None)
		if not sg:
			return None
		recv_time, data = sg
		if not isinstance(data, dict) or self.wall() - recv_time > GESTURE_FRESH_S:
			return None
		try:
			score = float(data.get("score", 0.0) or 0.0)
		except (TypeError, ValueError):
			return None
		if not data.get("gesture") or score < self.min_gesture_score:
			return None
		return data

	# ------------------------------------------------------------ modulation

	def _task_now(self):

		if self.active_task is not None:
			return self.active_task
		return self.pending_task()

	def compute_scene_mode(self):

		task = self._task_now()
		if task == "tricks":
			return "gestures"
		if task == "identify":
			return "objects"
		if self._feature("spontaneous_gestures") and self._feature("tricks") and not self.engaged():
			return "gestures"
		return "off"

	def compute_cue_mode(self):

		task = self.active_task
		if task == "dance":
			return "dancing"
		if task == "tricks":
			return "tricks"
		if task == "identify":
			return "listening"
		dialogue = self._node("dialogue")
		state = dialogue.state_name() if dialogue is not None else "idle"
		if state in ("facing", "greeting", "listening"):
			return "listening"
		if state == "thinking":
			return "thinking"
		if state in ("responding", "goodbye"):
			return "responding"
		return ""

	def compute_keyword_mask(self):

		# while busy only "Stop MiRo" may interrupt (HRI'25 turned wake
		# words off entirely, so a dance could not be stopped by voice)
		return frozenset(["stop_miro"]) if self.engaged() else None

	def _modulate(self):

		hearing = self._node("hearing")
		mask = self.compute_keyword_mask()
		if hearing is not None and mask != self.keyword_mask:
			hearing.set_keyword_mask(mask)
			self.keyword_mask = mask

		self.scene_mode = self.compute_scene_mode()
		self.output.scene_mode = self.scene_mode

		self.cue_mode = self.compute_cue_mode()
		cues = self._node("cues")
		if cues is not None:
			cues.set_mode(self.cue_mode)

		dialogue = self._node("dialogue")
		state = {
			"dialogue": dialogue.state_name() if dialogue is not None else "idle",
			"task": self.active_task,
			"pending": self.pending_task(),
			"scene_mode": self.scene_mode,
		}
		text = json.dumps(state, sort_keys=True)
		if text != self._last_state_json:
			self._last_state_json = text
			self.output.forebrain_state = text

	# ------------------------------------------------------------------ tick

	def tick(self):

		now = self.clock()

		# commands from the topic
		while True:
			try:
				text = self.commands.get_nowait()
			except queue.Empty:
				break
			self._handle_command(text)

		# what was heard
		hearing = self._node("hearing")
		dialogue = self._node("dialogue")
		if hearing is not None:
			for name, payload in hearing.poll_events():
				if name == "wake":
					self._on_wake_word(payload)
				elif name == "touch":
					self._on_touch(payload, now)
				elif dialogue is not None:
					dialogue.on_hearing(name, payload)

		# touch and gestures
		self._petting(now)
		self._spontaneous_tricks()

		# worker results
		self._poll_dance_job()
		self._poll_tts()
		self._poll_say(now)

		# a task nobody took
		if self._pending is not None and now - self._pending[2] > TASK_CLAIM_TIMEOUT_S:
			name = self._pending[0]
			print("[forebrain] task '" + name + "' was not taken up; dropped")
			self._pending = None
			self._maybe_resume_dialogue("not started")
		self._poll_resume()

		self._modulate()

	def shutdown(self):

		self.worker.shutdown()
