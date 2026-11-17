#
#	Hey MiRo - MemoryStore: who MiRo has talked to, across conversations.
#
#	When a conversation ends, the SLM itself turns the transcript into a
#	small record (consolidation: one extra, schema-constrained call on the
#	worker, off the critical path):
#
#	  {"name": "Sam", "facts": ["likes ABBA", "has a dog called Max"],
#	   "summary": "talked about dancing and dogs"}
#
#	Records with a name are kept per person; the rest only as anonymous
#	episodes. In a later conversation, a stored name heard in what the
#	person says brings that person's record back into the context:
#
#	  "You remember Sam (from 1 hour ago): likes ABBA; has a dog called
#	   Max. Last time you talked about dancing and dogs."
#
#	Recall is by name only: MiRo cannot tell people apart by face or
#	voice, so without a name it never claims to know someone.
#
#	Privacy: the file lives in the heymiro-cache volume on this computer,
#	records expire after ai.local.context.memory.ttl_hours (default 24 h,
#	one event day), and the "forget" command clears everything.
#

import json
import os
import re
import tempfile
import threading
import time

from context.events import ago

VERSION = 1
MAX_EPISODES = 20
MAX_FACT_CHARS = 80
MAX_SUMMARY_CHARS = 120
MAX_TURNS_IN_TRANSCRIPT = 12
HEAD_TURNS = 4

# "we talked about dancing" -> "dancing" (the sentence adds its own verb)
TALKED_RE = re.compile(r"^(?:(?:we|you|they|miro|the person)\s+)?(?:talked|chatted|spoke)\s+about\s+", re.I)

# a name: one word of letters (plus ' and -), as speech-to-text writes it
NAME_RE = re.compile(r"^[A-Za-z][A-Za-z'\-]{1,19}$")

# words a model or a regex might take for a name
NOT_NAMES = frozenset("""
	miro mirror robot null none unknown person user visitor child kid someone nobody anonymous
	name friend human boy girl man woman
""".split())

CONSOLIDATE_PROMPT = (
	"You keep the memory of MiRo, a robot pet, about the people it talks to. Read the conversation and answer in "
	"JSON with: "
	"\"name\": the person's first name if they said it, else an empty string; "
	"\"facts\": up to 5 things the person TOLD MiRo about themselves in a statement, each under 10 words and "
	"written about them, e.g. \"has a dog called Max\" or \"likes ABBA\". A question the person asks (\"Do you "
	"like music?\") tells you nothing about them: never turn it into a fact. What MiRo says about itself is never "
	"a fact about the person. "
	"\"summary\": the topics as a short phrase, e.g. \"dogs and ABBA\". Never invent anything; if the person "
	"told nothing about themselves, leave facts empty."
)

# a short answer needs MiRo's question to make sense ("Yes! I have
# twenty of them."); longer lines and the person's own questions do not
SHORT_ANSWER_WORDS = 8

# "facts" that are really the person's questions
QUESTION_FACT_RE = re.compile(r"^(asks|asked|wants to know|is curious|wonders|questions)\b", re.I)

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def consolidation_schema(max_facts=5):

	return {
		"type": "object",
		"properties": {
			"name": {"type": "string", "maxLength": 20},
			"facts": {"type": "array", "items": {"type": "string", "maxLength": MAX_FACT_CHARS},
				"maxItems": int(max_facts)},
			"summary": {"type": "string", "maxLength": MAX_SUMMARY_CHARS},
		},
		"required": ["name", "facts", "summary"],
		"additionalProperties": False,
	}


def consolidation_format(max_facts=5):

	return {"type": "json_schema", "json_schema": {"name": "miro_memory", "schema": consolidation_schema(max_facts)}}


def consolidation_messages(turns):

	"""
	turns: [(user text, MiRo's raw reply)] of the WHOLE conversation (not
	the trimmed chat history). For a long conversation only its beginning
	(introductions, the name) and its latest turns are used.

	Only the PERSON's lines are shown, so MiRo's own likes ("I love ABBA!")
	can never become facts about the person. The one exception: before a
	short answer, MiRo's last question is shown, so that "Yes! I have
	twenty of them." still means dinosaurs. (Measured on a 4B model: with
	all of MiRo's lines, its favourite colour and songs were stored as
	the person's; see docs/TECHNICAL.md section 20.)
	"""

	from cloud.llm import parse_llm_reply
	from context.budget import pick_turns
	lines = []
	asked = None
	for user, raw in pick_turns(turns, HEAD_TURNS, MAX_TURNS_IN_TRANSCRIPT - HEAD_TURNS):
		user = " ".join(str(user).split())
		short = len(user.split()) <= SHORT_ANSWER_WORDS and not user.endswith("?")
		if short and asked:
			lines.append("(MiRo asked: " + asked + ")")
		lines.append("Person: " + user)
		said = parse_llm_reply(raw).reply if raw else ""
		questions = [q.strip() for q in SENTENCE_RE.split(said or "") if q.strip().endswith("?")]
		asked = questions[-1] if questions else None
	return [
		{"role": "system", "content": CONSOLIDATE_PROMPT},
		{"role": "user", "content": "\n".join(lines)},
	]


def clean_name(value):

	name = str(value or "").strip().strip(".,!?\"'")
	if not NAME_RE.match(name) or name.lower() in NOT_NAMES:
		return None
	return name[:1].upper() + name[1:]


def parse_consolidation(text, max_facts=5):

	"""
	The model's answer -> {"name": str or None, "facts": [...], "summary": str},
	or None if it is not usable. Lenient: code fences and text around
	the object are ignored.
	"""

	text = str(text or "")
	start, end = text.find("{"), text.rfind("}")
	if start < 0 or end <= start:
		return None
	try:
		data = json.loads(text[start:end + 1])
	except ValueError:
		return None
	if not isinstance(data, dict):
		return None
	facts = []
	for f in data.get("facts") or []:
		f = " ".join(str(f).split()).strip(" .")[:MAX_FACT_CHARS]
		if QUESTION_FACT_RE.match(f):
			continue
		if f and f.lower() not in (x.lower() for x in facts):
			facts.append(f)
	summary = " ".join(str(data.get("summary") or "").split()).strip(" .")[:MAX_SUMMARY_CHARS]
	return {"name": clean_name(data.get("name")), "facts": facts[:max_facts], "summary": summary}


class MemoryStore(object):

	def __init__(self, path, ttl_hours=24.0, max_people=50, max_facts=5, clock=None):

		self.path = path
		self.ttl_s = max(0.0, float(ttl_hours)) * 3600.0
		self.max_people = int(max_people)
		self.max_facts = int(max_facts)
		self.clock = clock if clock is not None else time.time
		self.lock = threading.Lock()
		self.people = {}       # lower-case name -> record
		self.episodes = []     # [{"t", "summary"}] of conversations without a name
		self._load()

	# ------------------------------------------------------------ storage

	def _load(self):

		try:
			with open(self.path, "r") as f:
				data = json.load(f)
		except (IOError, OSError, ValueError):
			return
		if not isinstance(data, dict) or data.get("version") != VERSION:
			return
		people = data.get("people") or {}
		self.people = dict((k, v) for k, v in people.items() if isinstance(v, dict) and v.get("name"))
		self.episodes = [e for e in data.get("episodes") or [] if isinstance(e, dict)]
		self._purge(self.clock())

	def _save(self):

		# atomic: write a temporary file next to it, then rename
		folder = os.path.dirname(self.path) or "."
		os.makedirs(folder, exist_ok=True)
		data = {"version": VERSION, "people": self.people, "episodes": self.episodes}
		fd, tmp = tempfile.mkstemp(prefix=".memory-", dir=folder)
		try:
			with os.fdopen(fd, "w") as f:
				json.dump(data, f, indent=1, sort_keys=True)
			os.replace(tmp, self.path)
		except Exception:
			if os.path.exists(tmp):
				os.remove(tmp)
			raise

	def _purge(self, now):

		if self.ttl_s <= 0:
			return
		self.people = dict((k, v) for k, v in self.people.items()
			if now - float(v.get("last_seen", 0)) <= self.ttl_s)
		self.episodes = [e for e in self.episodes if now - float(e.get("t", 0)) <= self.ttl_s]

	# ------------------------------------------------------------- writing

	def remember(self, record, now=None):

		"""Merge one consolidated conversation; returns the person's name or None."""

		if not record:
			return None
		now = self.clock() if now is None else float(now)
		name = clean_name(record.get("name"))
		facts = list(record.get("facts") or [])
		summary = str(record.get("summary") or "")
		with self.lock:
			self._purge(now)
			if name is None:
				if summary:
					self.episodes.append({"t": now, "summary": summary})
					self.episodes = self.episodes[-MAX_EPISODES:]
					self._save()
				return None
			key = name.lower()
			person = self.people.get(key) or {"name": name, "facts": [], "first_seen": now, "times": 0}
			old = list(person.get("facts") or [])
			# newest facts first, no duplicates, at most max_facts
			merged = []
			for f in facts + old:
				if f.lower() not in (x.lower() for x in merged):
					merged.append(f)
			person["facts"] = merged[:self.max_facts]
			if summary:
				person["summary"] = summary
			person["last_seen"] = now
			person["times"] = int(person.get("times", 0)) + 1
			self.people[key] = person
			if len(self.people) > self.max_people:
				oldest = sorted(self.people, key=lambda k: self.people[k].get("last_seen", 0))
				for k in oldest[:len(self.people) - self.max_people]:
					del self.people[k]
			self._save()
			return name

	def forget_all(self):

		with self.lock:
			self.people = {}
			self.episodes = []
			self._save()

	# ------------------------------------------------------------- reading

	def names(self):

		with self.lock:
			return [p["name"] for p in self.people.values()]

	def person(self, name):

		with self.lock:
			p = self.people.get(str(name or "").lower())
			return dict(p) if p else None

	def find_name(self, transcript):

		"""A stored person's name said (as a whole word) in transcript, or None."""

		words = set(w.lower() for w in re.findall(r"[A-Za-z][A-Za-z'\-]*", str(transcript or "")))
		with self.lock:
			for key, p in sorted(self.people.items(), key=lambda kv: -float(kv[1].get("last_seen", 0))):
				if key in words:
					return p["name"]
		return None

	def describe(self, name, now=None, max_chars=260):

		"""The "You remember ..." sentence for a stored person, or None."""

		p = self.person(name)
		if p is None:
			return None
		now = self.clock() if now is None else float(now)
		text = "You remember " + p["name"] + " (you last talked " + ago(now - float(p.get("last_seen", now))) + ")"
		facts = p.get("facts") or []
		text += (": " + "; ".join(facts) + "." if facts else ".")
		summary = TALKED_RE.sub("", str(p.get("summary") or "")).rstrip(".")
		if summary:
			text += " Last time you talked about " + summary + "."
		if len(text) > max_chars:
			text = text[:max_chars - 3].rsplit(" ", 1)[0] + "..."
		return text

	def stats(self):

		with self.lock:
			return {"people": len(self.people), "episodes": len(self.episodes)}
