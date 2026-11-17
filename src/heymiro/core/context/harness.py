#
#	Hey MiRo - ContextHarness: what the small language model is told on
#	every turn (offline pipeline).
#
#	A 4B model cannot be asked to "remember" or to "know what it is
#	doing": it only knows what is in its context. The harness assembles
#	that context from the robot's own state, in one bracketed note in
#	front of the person's words:
#
#	  [Now: Monday 17 November 2026, 14:32. You are awake and feeling
#	   happy (valence 0.80, arousal 0.70).
#	   Recently: you danced to Dancing Queen by ABBA (2 minutes ago).
#	   Your camera sees: a person in front of you.
#	   You remember Sam (you last talked 1 hour ago): likes ABBA.
#	   Choose actions from the upcoming message only, ...]
#	  Hi MiRo, it's Sam again!
#
#	  self-state  time, awake/asleep, emotion, what the body is doing now
#	  events      context/events.py: what the body did recently
#	  scene       forebrain/scene_context.py: what the camera sees
#	  memory      context/memory.py: the person, if they said a stored name
#
#	Order and budget: the system prompt and the examples stay first and
#	never change (the llama.cpp prompt cache keeps them); the note comes
#	last, just before the person's words, so it costs only its own
#	tokens. Each section has a character cap, and empty sections are left
#	out, so the note stays around 100 tokens.
#
#	Split in two because of threads: snapshot() runs on the 50 Hz tick
#	(it reads the robot's state), message() on the worker after the
#	speech-to-text (it needs the person's words to recall them).
#

import threading
import time

from context import memory as mem

EVENTS_MAX_CHARS = 300

NOTHING_HAPPENED = "Recently: nothing; you have just been listening."

# sections given up first when the note is over its token reserve; the
# time, the self-state and the reminder always stay
DROP_ORDER = ("scene", "events", "memory")
SCENE_MAX_CHARS = 160
MEMORY_MAX_CHARS = 260

# what the active forebrain task looks like from inside
DOING = {
	"dance": "you are dancing",
	"tricks": "you are playing the tricks game",
	"identify": "you are playing the object game",
}


def _setting(cfg, key, default):

	# dotted key in a config Section (heymiro_config) or a plain dict
	if cfg is None:
		return default
	if not isinstance(cfg, dict):
		return cfg.get(key, default)
	node = cfg
	for part in key.split("."):
		if not isinstance(node, dict) or part not in node:
			return default
		node = node[part]
	return node


def _cap(text, n):

	text = " ".join(str(text or "").split())
	if len(text) <= n:
		return text
	return text[:n - 3].rsplit(" ", 1)[0] + "..."


class Snapshot(object):

	"""The robot's state at the moment a turn was submitted (taken on the tick)."""

	def __init__(self, now_str=None, valence=None, arousal=None, awake=True, doing=None,
			events=None, scene=None, t=None):

		self.now_str = now_str
		self.valence = valence
		self.arousal = arousal
		self.awake = awake
		self.doing = doing
		self.events = events
		self.scene = scene
		self.t = t


class ContextHarness(object):

	def __init__(self, events=None, memory=None, cfg=None, chat=None, clock=None):

		"""
		events: context.events.EventLog or None; memory: MemoryStore or
		None; cfg: the ai.local.context settings (a dict or a config
		node); chat(messages, response_format) -> text is used to
		consolidate a finished conversation into the memory.
		"""

		get = lambda k, d: _setting(cfg, k, d)
		self.events = events
		self.memory = memory
		self.chat = chat
		self.clock = clock if clock is not None else time.time
		self.self_state = bool(get("self_state", True))
		self.events_max_age_s = float(get("events.max_age_s", 600.0))
		self.events_max_items = int(get("events.max_items", 4))
		self.max_facts = int(get("memory.max_facts", 5))
		self.budget = None  # context.budget.TokenBudget (set by the forebrain once measured)
		self.lock = threading.Lock()
		self._name = None   # the person recalled in the current conversation

	# ---------------------------------------------------------- the tick

	def snapshot(self, valence=None, arousal=None, now_str=None, awake=True, task=None, scene=None):

		events = None
		if self.events is not None:
			events = self.events.describe(self.clock(), self.events_max_age_s, self.events_max_items)
		return Snapshot(now_str, valence, arousal, awake, DOING.get(task), events, scene, self.clock())

	def start_conversation(self):

		with self.lock:
			self._name = None

	def end_conversation(self):

		"""Forget who this conversation was with; returns that name (for consolidate)."""

		with self.lock:
			name, self._name = self._name, None
			return name

	# -------------------------------------------------------- the worker

	def recall(self, transcript):

		"""The person in this conversation, if they said a stored name (sticky)."""

		if self.memory is None:
			return None
		name = self.memory.find_name(transcript)
		with self.lock:
			if name is not None:
				self._name = name
			return self._name

	def note(self, snap, transcript):

		"""The bracketed context for one turn (without the brackets)."""

		from cloud.llm import infer_emotion, ACTION_REMINDER
		parts = []      # (section, text)
		add = lambda key, text: parts.append((key, text))
		if snap.now_str:
			add("now", "Now: " + str(snap.now_str) + ".")
		if self.self_state:
			state = "You are " + ("awake" if snap.awake else "sleepy")
			if snap.valence is not None and snap.arousal is not None:
				emotion = infer_emotion(snap.valence, snap.arousal) or "fine"
				state += " and feeling %s (valence %.2f, arousal %.2f)" % (emotion, snap.valence, snap.arousal)
			if snap.doing:
				state += "; " + snap.doing
			add("self", state + ".")
		elif snap.valence is not None and snap.arousal is not None:
			add("self", "Your current valence is %.2f and your current arousal is %.2f." % (snap.valence, snap.arousal))
		if snap.events:
			add("events", _cap(snap.events, EVENTS_MAX_CHARS))
		elif self.events is not None:
			# say it: an SLM with no events tends to make one up
			add("events", NOTHING_HAPPENED)
		if snap.scene:
			add("scene", _cap(snap.scene, SCENE_MAX_CHARS))
		name = self.recall(transcript)
		if name is not None:
			remembered = self.memory.describe(name, self.clock(), MEMORY_MAX_CHARS)
			if remembered:
				add("memory", remembered)
		add("reminder", ACTION_REMINDER)
		if self.budget is not None:
			# over its token reserve: drop the least needed sections first
			for key in DROP_ORDER:
				if self.budget.estimate(" ".join(t for _, t in parts)) <= self.budget.reserve_note:
					break
				parts = [(k, t) for k, t in parts if k != key]
		return " ".join(t for _, t in parts)

	def message(self, snap, transcript):

		return {"role": "system", "content": self.note(snap, transcript)}

	def consolidate(self, turns, known=None):

		"""
		Worker job when a conversation ends: the SLM turns the transcript
		into a memory record. known: the name recalled during the
		conversation (end_conversation()), used if the model gives none.
		Returns the stored name (or None); never
		raises (a failure only means nothing is remembered).
		"""

		if self.memory is None or self.chat is None or not turns:
			return None
		try:
			text = self.chat(mem.consolidation_messages(turns), mem.consolidation_format(self.max_facts))
		except Exception as e:
			from cloud.worker import describe_error
			print("[context] memory not updated (" + describe_error(e) + ")")
			return None
		record = mem.parse_consolidation(text, self.max_facts)
		if record is None:
			print("[context] memory not updated (unreadable answer)")
			return None
		if record.get("name") is None and known is not None:
			record["name"] = known
		name = self.memory.remember(record)
		print("[context] remembered " + (name if name else "an anonymous conversation")
			+ " (" + str(len(record.get("facts") or [])) + " facts)")
		return name
