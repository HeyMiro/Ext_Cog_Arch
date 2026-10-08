#
#	Hey MiRo - ActionIdentify: "what am I holding?" (the object game).
#
#	Runs the forebrain "identify" task. The scene client runs YOLO on the
#	left camera while this task is active (scene mode "objects") and
#	publishes what it sees; this action
#
#	  look     holds a look-up pose with the loading tune playing, and
#	           counts each label over the last detections: a label seen
#	           in identify.confirm = [k, n] (4 of the last 10) is confirmed
#	  comment  for a newly confirmed object, one CloudWorker job asks the
#	           LLM (llm.object_comment_model) for a one-line comment and
#	           synthesizes it; without a key it uses a template line
#
#	and ends after identify.duration_s spent looking. If nothing was
#	recognised it says so (HRI'25's closing line). HRI'25 called GPT and
#	ElevenLabs inside the tick and blocked on wait_done().
#

import collections
import random

from . import forebrain_action
from choreography import moves
from cloud.llm import object_comment_prompt, clean_text
from cloud.worker import describe_error

INTRO_LINE = "Hmmm, let me see if I can figure out what it is!"
NOTHING_LINE = ("I'm sorry, I couldn't see anything. I can't recognise as much as you humans "
	"because I'm still learning!")
TEMPLATES = ["Ooh, a {}!", "Is that a {}? Cool!", "I can see a {}!"]

# head pose while looking (HRI'25 [0, 0.4, 0, 0.1])
LOOK_LIFT = 0.4
LOOK_PITCH = 0.1
LOOK_POSE_S = 1.0

# scene results older than this are not "now"
OBJECTS_FRESH_S = 1.0

# give up on a comment job after this long
COMMENT_TIMEOUT_S = 30.0



class ActionIdentify(forebrain_action.ForebrainAction):

	NAME = "identify"
	TASK = "identify"

	def init(self):

		cfg = self.pars.forebrain.cfg
		self.duration_s = float(cfg.get("identify.duration_s", 15.0))
		confirm = cfg.get("identify.confirm", [4, 10]) or [4, 10]
		self.confirm_k = int(confirm[0])
		self.confirm_n = int(confirm[1])
		self.comment_model = cfg.get("llm.object_comment_model", "gpt-4o-mini")
		self.rng = random.Random()
		self._reset()

	def _reset(self):

		self.phase = None
		self.history = collections.deque(maxlen=max(1, getattr(self, "confirm_n", 10)))
		self.last_stamp = None
		self.seen = []
		self.look_time = 0.0
		self.t_tick = None
		self.job = None
		self.t_job = 0.0
		self.traj = None
		self.t_traj = 0.0
		self.music = False

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
		print("[identify] start")
		start = list(self.kc.getConfig())
		target = list(start)
		target[moves.LIFT_I] = LOOK_LIFT
		target[moves.PITCH_I] = LOOK_PITCH
		self.traj = moves.Trajectory(start, target, LOOK_POSE_S)
		self.t_traj = self.now()
		if params.get("intro"):
			fb.say(INTRO_LINE)
		self.phase = "look"

	def on_stop(self, reason):

		print("[identify] end (" + reason + "; saw " + (", ".join(self.seen) or "nothing") + ")")
		if self.music and self.voice is not None:
			self.voice.stop("music")
		# closing line only if the game ran its course (not after "stop",
		# which clears the task before we get here)
		fb = self.forebrain
		if reason == "done" and not self.seen and fb.active_task == self.TASK:
			fb.say(NOTHING_LINE)
		self._reset()

	def on_service(self):

		now = self.now()
		dt = 0.0 if self.t_tick is None else now - self.t_tick
		self.t_tick = now

		if self.traj is not None:
			cfg, done = self.traj.sample(now - self.t_traj)
			self.set_head(cfg)
			if done:
				self.traj = None

		if self.phase == "look":
			self._service_look(dt)
		elif self.phase == "comment":
			self._service_comment(now)

	# ----------------------------------------------------------------- look

	def _speaking(self):

		voice = self.voice
		return self.forebrain.is_saying() or (voice is not None and voice.is_speaking())

	def _service_look(self, dt):

		# the loading tune plays while MiRo is looking (not under speech)
		voice = self.voice
		if not self.music and voice is not None and not self._speaking():
			voice.play_clip("loading", channel="music")
			self.music = True

		self.look_time += dt
		if self.look_time > self.duration_s:
			self.finish("done")
			return

		label = self._confirmed_object()
		if label is not None and label not in self.seen:
			self.seen.append(label)
			print("[identify] I see a " + label)
			self.job = self.forebrain.worker.submit(self._comment, label)
			self.t_job = self.now()
			self.phase = "comment"

	def _confirmed_object(self):

		# count each new scene result once; k of the last n confirms
		so = getattr(self.system_state, "scene_objects", None)
		if not so:
			return None
		recv_time, data = so
		if not isinstance(data, dict):
			return None
		stamp = data.get("t", recv_time)
		if stamp == self.last_stamp or self.forebrain.wall() - recv_time > OBJECTS_FRESH_S:
			return None
		self.last_stamp = stamp
		labels = set()
		for obj in data.get("objects") or []:
			if isinstance(obj, dict) and obj.get("label"):
				labels.add(str(obj["label"]))
		self.history.append(labels)
		counts = collections.Counter()
		for frame in self.history:
			counts.update(frame)
		best = None
		for label, n in counts.most_common():
			if n >= self.confirm_k and label not in self.seen:
				best = label
				break
		return best

	# -------------------------------------------------------------- comment

	def _comment(self, label):

		# worker thread: an LLM one-liner (or a template), then TTS
		cloud = self.forebrain.cloud
		text = None
		if cloud.clients is not None:
			try:
				text = clean_text(cloud.clients.chat(object_comment_prompt(label), self.comment_model,
					temperature=0.8, max_tokens=60))
			except Exception as e:
				print("[identify] comment failed (" + describe_error(e) + "); using a template")
		if not text:
			text = self.rng.choice(TEMPLATES).format(label.replace("_", " "))
		pcm = None
		if cloud.tts is not None:
			try:
				pcm = cloud.tts.synthesize(text)
			except Exception as e:
				print("[identify] text-to-speech failed (" + describe_error(e) + ")")
		return text, pcm

	def _service_comment(self, now):

		voice = self.voice
		if self.job is not None:
			if not self.job.done():
				if now - self.t_job > COMMENT_TIMEOUT_S:
					self.job = None
					self.phase = "look"
				return
			job = self.job
			self.job = None
			try:
				text, pcm = job.result()
			except Exception as e:
				print("[identify] comment failed (" + describe_error(e) + ")")
				self.phase = "look"
				return
			if voice is not None:
				voice.stop("music")
			self.music = False
			if pcm is not None and len(pcm) > 0 and voice is not None:
				voice.say_pcm(pcm)
			else:
				print("[identify] MiRo says: " + text)
			return

		# wait for the comment to be said, then look again
		if not self._speaking():
			self.history.clear()
			self.phase = "look"
