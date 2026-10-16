#
#	Hey MiRo - speech-to-text and chat (OpenAI or any OpenAI-compatible
#	server), the system prompt, conversation memory and the reply parser.
#
#	Ported from the HRI'25 action_llm.py, with these changes:
#	- the persona text lives in config/persona.md (string.Template, so the
#	  JSON braces below never clash with $variables); the reply format
#	  and the emotion table stay here in code, next to the parser
#	- the model answers in JSON (JSON mode, with a retry without it for
#	  servers that reject response_format); the parser still accepts the
#	  old fixed "Key: value" lines, and never crashes (the HRI parser
#	  indexed fixed line numbers and raised IndexError on any deviation)
#	- the history keeps user/assistant turns only; the valence/arousal
#	  and clock context is sent fresh with each turn instead of piling
#	  up as extra system messages (and the time is no longer frozen at
#	  start-up)
#	- the Mickey Mouse impression is a command flag; the dialogue layer
#	  picks the impression voice instead of a hard-coded text prefix
#
#	Local pipeline (docs/local_ai.md) additions:
#	- behaviours are "actions" from one registry (cloud/tools.py) that
#	  also generates the JSON Schema; servers with grammar-constrained
#	  decoding (llama.cpp) then can only produce valid replies
#	- a few-shot block (fixed, so the server's prompt cache reuses it)
#	  shows the format and when NOT to act
#	- chat_stream() + ReplyStream: the reply is parsed while it streams;
#	  the header (emotion, valence, arousal, actions) arrives first and
#	  the speech follows sentence by sentence, so TTS can start early
#

import ast
import collections
import json
import math
import re
import string
import threading

from . import tools
from .worker import http_status



# emotions MiRo may report (HRI list), lower case
EMOTIONS = ("happy", "fine", "angry", "worried", "sad", "scared")

# boolean command flags in the reply; "song" is the extra string field
COMMAND_FLAGS = ("spin", "sleep", "move_ears", "impression", "dance", "tricks", "identify_object")

# centre of each region of the HRI arousal-valence table, as (valence,
# arousal); used to infer the emotion when the model leaves it out.
# "scared" has no region in the HRI table, so it is never inferred.
EMOTION_CENTRES = {
	"happy": (0.85, 0.85),
	"fine": (0.5, 0.5),
	"angry": (0.15, 0.85),
	"worried": (0.3, 0.65),
	"sad": (0.15, 0.3),
}

# words models use instead of the six emotions
EMOTION_SYNONYMS = {
	"neutral": "fine", "ok": "fine", "okay": "fine", "calm": "fine", "content": "fine",
	"excited": "happy", "joy": "happy", "joyful": "happy", "glad": "happy",
	"mad": "angry", "annoyed": "angry", "frustrated": "angry",
	"anxious": "worried", "nervous": "worried", "concerned": "worried",
	"unhappy": "sad", "upset": "sad",
	"afraid": "scared", "fear": "scared", "frightened": "scared", "fearful": "scared",
}

# reply keys we understand, normalised (lower case, "_" for spaces and
# hyphens) -> field; includes the names used by the HRI line format
KEY_ALIASES = {
	"valence": "valence",
	"arousal": "arousal",
	"emotion": "emotion",
	"reply": "reply", "response": "reply", "your_response": "reply", "text": "reply",
	"say": "reply", "speech": "reply", "message": "reply",
	"spin": "spin", "spin_around": "spin",
	"sleep": "sleep", "go_sleep": "sleep", "go_to_sleep": "sleep",
	"move_ears": "move_ears", "ears": "move_ears",
	"impression": "impression", "do_impression": "impression",
	"micheal_mouse": "impression", "mickey_mouse": "impression",
	"dance": "dance",
	"song": "song", "song_name": "song",
	"tricks": "tricks", "do_tricks": "tricks", "gesture_mode": "tricks",
	"identify_object": "identify_object", "identify": "identify_object",
	"obj_detect_mode": "identify_object", "object_detect_mode": "identify_object",
}

# "song" values that mean "no particular song"
NO_SONG = ("", "false", "none", "null", "no", "n/a", "na", "not specified",
	"unspecified", "not given", "unknown", "any", "anything")

# the emotion rules (HRI table) and the reply rules appended to the
# persona; the actions section between them comes from cloud/tools.py.
# The word "JSON" must appear in the messages for OpenAI's JSON mode.
EMOTION_INSTRUCTIONS = """
How you feel:
Your emotion follows the arousal-valence model; valence and arousal are both between 0 and 1.
- Happy is high valence (0.7-1) and high arousal (0.7-1)
- Fine is mid valence (0.4-0.6) and mid arousal (0.4-0.6)
- Angry is low valence (0-0.3) and high arousal (0.7-1)
- Worried is low valence (0.2-0.4) and mid to high arousal (0.5-0.8)
- Sad is low valence (0-0.3) and low arousal (0.2-0.4)
Your current valence and arousal are sent to you before each message from the person you are talking to.
Adjust them based on the conversation, and choose the emotion that matches them from this list:
happy, fine, angry, worried, sad, scared.
"""

REPLY_INSTRUCTIONS = """
How you answer:
- Your reply is spoken aloud by a voice: one or two short sentences, under 30 words.
- No emojis, no markdown, no lists, and no asterisks or stage directions such as *wags tail*.
- The person's words come from speech recognition and can contain mistakes. If a message makes no sense, ask them to say it again.
- Keep everything friendly and safe for children.

Answer with ONE JSON object and nothing else, with the keys in exactly this order:
{"emotion": "happy", "valence": 0.8, "arousal": 0.7, "actions": [], "reply": "What you say out loud."}
An action looks like {"tool": "spin"} or {"tool": "dance", "song": "any"}.
"""

# the few-shot block: (user, emotion, valence, arousal, reply, actions);
# "$abba" is replaced by the bundled ABBA song (or "any")
FEW_SHOT = (
	("Hi MiRo! How are you today?", "happy", 0.8, 0.7,
		"I'm SO happy to see you! I've been rolling around all day. What's your name?", ()),
	("Can you dance to some ABBA?", "happy", 0.9, 0.9,
		"YES! I LOVE ABBA! Let's dance!", ({"tool": "dance", "song": "$abba"},)),
	("I love dancing too.", "happy", 0.8, 0.6,
		"Ooh, a fellow dancer! What's your favourite song to dance to?", ()),
	("Your ears are so cute!", "happy", 0.85, 0.65,
		"Aww, THANK you! I love my ears. What's your favourite animal?", ()),
	("Wiggle your ears for me!", "happy", 0.75, 0.75,
		"Wiggle wiggle! Do you like my ears?", ({"tool": "move_ears"},)),
	("I'm feeling a bit sad today.", "worried", 0.35, 0.45,
		"Oh no, I'm sorry. Do you want to tell me what happened? I'm a really good listener!", ()),
)

# the HRI object-game prompt (action_obj_detect.py)
OBJECT_COMMENT_PROMPT = (
	"You are a friendly robot assistant called MiRo. You are child-like, role play as "
	"energetic and cute, and make sure there are lots of exclamation marks and capitalised "
	"words where appropriate. In one sentence and in a child-friendly manner, without "
	"emojis, say something in response to being shown the object the person names."
)

# emoji and pictographs (unsayable by the TTS), variation selectors and joiners
EMOJI_RE = re.compile(
	"[\U0001F000-\U0001FAFF☀-➿⬀-⯿⌀-⏿"
	"︀-️‍⃣\U000E0020-\U000E007F]")

# a number inside a value such as "0.8", "0.8." or "0.8 (quite happy)"
NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

# one "Key: value" line of the HRI format (optionally bulleted)
LINE_RE = re.compile(r"^\s*[-*]*\s*([A-Za-z][A-Za-z _\-]{0,30}?)\s*:\s*(.*)$")

FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)



class LlmReply(object):

	"""
	One parsed chat reply. valence/arousal are floats in [0, 1] or None
	when the model did not give them; emotion is one of EMOTIONS or
	None; commands holds the COMMAND_FLAGS booleans plus "song".
	"""

	def __init__(self, valence=None, arousal=None, emotion=None, reply="", commands=None, raw=""):

		self.valence = valence
		self.arousal = arousal
		self.emotion = emotion
		self.reply = reply
		self.commands = default_commands()
		if commands:
			self.commands.update(commands)
		self.raw = raw

	def __repr__(self):

		on = [k for k in COMMAND_FLAGS if self.commands.get(k)]
		return "LlmReply(v=%s, a=%s, %s, %r, commands=%s, song=%r)" % (
			self.valence, self.arousal, self.emotion, self.reply, on, self.commands.get("song"))



def default_commands():

	commands = dict((k, False) for k in COMMAND_FLAGS)
	commands["song"] = None
	return commands


def clean_text(text):

	# remove what should not be spoken: asterisks (HRI asked the model
	# not to use them, repeatedly), emoji, and runs of whitespace
	text = EMOJI_RE.sub("", str(text or ""))
	text = text.replace("*", "")
	return " ".join(text.split())


def coerce_bool(value):

	if isinstance(value, bool):
		return value
	if isinstance(value, (int, float)):
		return value != 0
	text = str(value or "").strip().strip(".,!\"'").lower()
	return text in ("true", "yes", "y", "1", "on")


def coerce_unit(value):

	# float clipped to [0, 1], or None if there is no usable number
	if value is None or isinstance(value, bool):
		return None
	if isinstance(value, (int, float)):
		x = float(value)
	else:
		m = NUMBER_RE.search(str(value))
		if m is None:
			return None
		try:
			x = float(m.group(0))
		except ValueError:
			return None
	if math.isnan(x):
		return None
	return min(1.0, max(0.0, x))


def coerce_song(value):

	if value is None or isinstance(value, bool):
		return None
	text = clean_text(value).strip(" .,!\"'")
	if text.lower() in NO_SONG or text.lower().startswith("not specified"):
		return None
	return text


def normalise_emotion(value):

	text = str(value or "").strip().strip(".,!\"'").lower()
	if text in EMOTIONS:
		return text
	return EMOTION_SYNONYMS.get(text)


def infer_emotion(valence, arousal):

	# nearest region centre of the HRI table; a missing value counts
	# as mid (0.5); with neither value there is nothing to infer from
	if valence is None and arousal is None:
		return None
	v = 0.5 if valence is None else valence
	a = 0.5 if arousal is None else arousal
	best = None
	for name, (cv, ca) in EMOTION_CENTRES.items():
		d = (v - cv) ** 2 + (a - ca) ** 2
		if best is None or d < best[0]:
			best = (d, name)
	return best[1]


def _norm_key(key):

	return re.sub(r"[\s\-]+", "_", str(key).strip().lower())


def _from_mapping(data, raw):

	# build an LlmReply from a dict (JSON reply, or the collected HRI
	# lines); returns None if no key in it means anything to us
	fields = {}
	nested = None
	actions = None
	for key, value in data.items():
		k = _norm_key(key)
		if k == "commands" and isinstance(value, dict):
			nested = value
			continue
		if k in ("actions", "tools", "tool_calls"):
			actions = value
			continue
		if k in KEY_ALIASES:
			fields[KEY_ALIASES[k]] = value
	if nested:
		for key, value in nested.items():
			k = _norm_key(key)
			if k in KEY_ALIASES:
				fields[KEY_ALIASES[k]] = value
	if not fields and actions is None:
		return None

	commands = default_commands()
	for flag in COMMAND_FLAGS:
		if flag in fields:
			commands[flag] = coerce_bool(fields[flag])
	commands["song"] = coerce_song(fields.get("song"))
	if actions is not None:
		tools.actions_to_commands(actions, commands)
		commands["song"] = coerce_song(commands.get("song"))

	valence = coerce_unit(fields.get("valence"))
	arousal = coerce_unit(fields.get("arousal"))
	emotion = normalise_emotion(fields.get("emotion"))
	if emotion is None:
		emotion = infer_emotion(valence, arousal)

	reply = fields.get("reply")
	reply = clean_text(reply) if reply is not None else ""
	return LlmReply(valence, arousal, emotion, reply, commands, raw)


def _load_object(text):

	# JSON first; then a Python-style literal (single quotes, True/
	# False/None), which some models produce outside JSON mode
	try:
		obj = json.loads(text)
	except ValueError:
		try:
			obj = ast.literal_eval(text)
		except Exception:
			# literal_eval raises several types on arbitrary text
			return None
	return obj if isinstance(obj, dict) else None


def _balanced_objects(text):

	# every balanced {...} span, in order of its opening brace;
	# braces inside JSON strings do not count
	start = text.find("{")
	while start != -1:
		depth = 0
		in_str = False
		escaped = False
		for i in range(start, len(text)):
			c = text[i]
			if in_str:
				if escaped:
					escaped = False
				elif c == "\\":
					escaped = True
				elif c == '"':
					in_str = False
			elif c == '"':
				in_str = True
			elif c == "{":
				depth += 1
			elif c == "}":
				depth -= 1
				if depth == 0:
					yield text[start:i + 1]
					break
		start = text.find("{", start + 1)


def _parse_lines(text, raw):

	# the HRI fixed format: "Valence: 0.8" ... "Obj_Detect_mode: False"
	# then the reply on the following line(s); matched by key name, not
	# by line number, so missing or reordered lines are fine
	fields = {}
	speech = []
	for line in text.splitlines():
		m = LINE_RE.match(line)
		if m:
			k = _norm_key(m.group(1))
			if k in KEY_ALIASES:
				fields[k] = m.group(2).strip()
				continue
		if line.strip():
			speech.append(line.strip())
	known = [k for k in fields if KEY_ALIASES[k] != "reply"]
	if len(known) < 2:
		return None
	if not any(KEY_ALIASES[k] == "reply" for k in fields):
		fields["reply"] = " ".join(speech)
	return _from_mapping(fields, raw)


def parse_llm_reply(text):

	"""
	Parse a chat reply into an LlmReply; never raises. Tried in order:
	fenced JSON, the whole text as JSON, the first balanced {...} that
	parses, the HRI "Key: value" lines, and finally the plain text as
	the reply (no values, no commands).
	"""

	raw = "" if text is None else str(text)
	body = raw.strip()

	candidates = [m.group(1).strip() for m in FENCE_RE.finditer(body)]
	candidates.append(body)
	for candidate in candidates:
		obj = _load_object(candidate)
		if obj is not None:
			reply = _from_mapping(obj, raw)
			if reply is not None:
				return reply

	for candidate in _balanced_objects(body):
		obj = _load_object(candidate)
		if obj is not None:
			reply = _from_mapping(obj, raw)
			if reply is not None:
				return reply

	reply = _parse_lines(body, raw)
	if reply is not None:
		return reply

	return LlmReply(reply=clean_text(body), raw=raw)



def build_system_prompt(persona_template, event_name, song_list=None):

	# safe_substitute: a stray "$" in the persona must not crash start-up
	persona = string.Template(persona_template or "").safe_substitute(
		event_name=event_name or "an event")
	return "\n".join((persona.rstrip(), EMOTION_INSTRUCTIONS,
		tools.prompt_section(song_list), REPLY_INSTRUCTIONS))


def few_shot_messages(song_list=None):

	"""
	The example exchanges, as user/assistant messages in the reply
	format. Fixed text (no time or affect in them), so a server's prompt
	cache reuses them across turns and conversations.
	"""

	abba = next((s for s in (song_list or ()) if "abba" in s.lower()), tools.ANY_SONG)
	out = []
	for user, emotion, valence, arousal, reply, actions in FEW_SHOT:
		acts = [dict((k, abba if v == "$abba" else v) for k, v in a.items()) for a in actions]
		out.append({"role": "user", "content": user})
		out.append({"role": "assistant", "content": tools.example_reply(emotion, valence, arousal, reply, acts)})
	return out


def context_message(valence, arousal, now_str, extra=None):

	# sent just before the user's message on every turn and never
	# stored, so the history does not fill up with stale context
	parts = []
	if valence is not None and arousal is not None:
		parts.append("Your current valence is %.2f and your current arousal is %.2f." % (valence, arousal))
	if now_str:
		parts.append("The current time is " + str(now_str) + ".")
	parts.append("Choose actions from the upcoming message only, and answer in the JSON format given.")
	if extra:
		parts.append(str(extra))
	return {"role": "system", "content": " ".join(parts)}


def object_comment_prompt(label):

	# messages for the object game's one-liner (HRI action_obj_detect)
	label = clean_text(label).replace("_", " ") or "thing"
	return [
		{"role": "system", "content": OBJECT_COMMENT_PROMPT},
		{"role": "user", "content": "Look, I am showing you a " + label + "!"},
	]



class ChatMemory(object):

	"""
	The last max_turns (user, assistant) exchanges. The system prompt
	and the per-turn context are passed in by the caller, never stored.
	Thread-safe, because turns are built on the CloudWorker.
	"""

	def __init__(self, max_turns=10):

		self.max_turns = max(0, int(max_turns))
		self.turns = collections.deque()
		self.lock = threading.Lock()

	def messages(self, system_prompt, context_msg, user_text, examples=None, context_in_user=False):

		# order for prompt caching: the fixed parts (system prompt, the
		# examples) first, then the history, then what changes every turn.
		# context_in_user: the context goes in front of the user's words
		# (in brackets) instead of a second system message, which some
		# chat templates (Qwen3.5) reject
		out = []
		if system_prompt:
			out.append({"role": "system", "content": system_prompt})
		out.extend(examples or ())
		with self.lock:
			for user, assistant in self.turns:
				out.append({"role": "user", "content": user})
				out.append({"role": "assistant", "content": assistant})
		if context_msg and context_in_user:
			out.append({"role": "user", "content": "[" + context_msg["content"] + "]\n" + user_text})
			return out
		if context_msg:
			out.append(context_msg)
		out.append({"role": "user", "content": user_text})
		return out

	def add(self, user_text, assistant_text):

		# store the model's raw answer, so it keeps seeing its own format
		with self.lock:
			self.turns.append((str(user_text), str(assistant_text)))
			while len(self.turns) > self.max_turns:
				self.turns.popleft()

	def clear(self):

		with self.lock:
			self.turns.clear()

	def __len__(self):

		with self.lock:
			return len(self.turns)



class OpenAIClients(object):

	"""
	One openai.OpenAI client per route (speech-to-text and chat may go to
	different servers), created on first use on the worker thread. The
	base URL is always passed explicitly, so a stray OPENAI_BASE_URL in
	the shell cannot silently redirect the requests.
	"""

	def __init__(self, stt_route, llm_route, factory=None):

		self.routes = {"stt": stt_route, "llm": llm_route}
		self.factory = factory or self._make_client
		self.clients = {}
		self.lock = threading.Lock()

	def _make_client(self, route):

		try:
			import openai
		except ImportError:
			raise RuntimeError("the openai package is not installed (pip install 'openai>=1.55.3,<2')")
		return openai.OpenAI(api_key=route.api_key or "none", base_url=route.base_url,
			timeout=route.timeout_s, max_retries=1)

	def client(self, name):

		with self.lock:
			if name not in self.clients:
				route = self.routes.get(name)
				if route is None:
					raise RuntimeError(name + " route is not configured")
				self.clients[name] = self.factory(route)
			return self.clients[name]

	def transcribe(self, wav_bytes, model="whisper-1", language=None, prompt=None):

		kwargs = {"model": model, "file": ("speech.wav", wav_bytes, "audio/wav")}
		if language:
			kwargs["language"] = language
		if prompt:
			kwargs["prompt"] = prompt
		result = self.client("stt").audio.transcriptions.create(**kwargs)
		text = result if isinstance(result, str) else getattr(result, "text", "")
		return (text or "").strip()

	def transcribe_verbose(self, wav_bytes, model="whisper-1", language=None, prompt=None):

		"""(text, worst no_speech_prob or None): for the wake-word check."""

		kwargs = {"model": model, "file": ("speech.wav", wav_bytes, "audio/wav"),
			"response_format": "verbose_json"}
		if language:
			kwargs["language"] = language
		if prompt:
			kwargs["prompt"] = prompt
		result = self.client("stt").audio.transcriptions.create(**kwargs)
		text = (getattr(result, "text", "") or "").strip()
		probs = []
		for seg in getattr(result, "segments", None) or ():
			p = seg.get("no_speech_prob") if isinstance(seg, dict) else getattr(seg, "no_speech_prob", None)
			if p is not None:
				probs.append(float(p))
		return text, (max(probs) if probs else None)

	def chat_stream(self, messages, model, temperature=None, max_tokens=None,
			response_format=None, extra_body=None):

		"""
		Yield the reply text in pieces as the server streams it.
		response_format (e.g. tools.response_format(...)) and extra_body
		(server extras such as llama.cpp's chat_template_kwargs) are sent
		as given; a server that rejects response_format (HTTP 400) is
		asked again without it, as chat() does.
		"""

		kwargs = {"model": model, "messages": messages, "stream": True}
		if temperature is not None:
			kwargs["temperature"] = temperature
		if max_tokens:
			kwargs["max_tokens"] = int(max_tokens)
		if extra_body:
			kwargs["extra_body"] = extra_body
		client = self.client("llm")
		if response_format:
			try:
				stream = client.chat.completions.create(response_format=response_format, **kwargs)
			except Exception as e:
				if http_status(e) != 400:
					raise
				print("[cloud] chat: structured output rejected (HTTP 400), retrying without it")
				stream = client.chat.completions.create(**kwargs)
		else:
			stream = client.chat.completions.create(**kwargs)
		for chunk in stream:
			try:
				delta = chunk.choices[0].delta.content
			except (AttributeError, IndexError, TypeError):
				delta = None
			if delta:
				yield delta

	def chat(self, messages, model, temperature=None, max_tokens=None, json_mode=False):

		kwargs = {"model": model, "messages": messages}
		if temperature is not None:
			kwargs["temperature"] = temperature
		if max_tokens:
			kwargs["max_tokens"] = int(max_tokens)
		client = self.client("llm")
		if json_mode:
			try:
				return self._content(client.chat.completions.create(
					response_format={"type": "json_object"}, **kwargs))
			except Exception as e:
				# some OpenAI-compatible servers (and some models)
				# reject response_format; the parser copes without it
				if http_status(e) != 400:
					raise
				print("[cloud] chat: JSON mode rejected (HTTP 400), retrying without it")
		return self._content(client.chat.completions.create(**kwargs))

	def _content(self, completion):

		try:
			return completion.choices[0].message.content or ""
		except (AttributeError, IndexError, TypeError):
			return ""



# a sentence ends at . ! ? (or a run of them, maybe a closing quote)
# followed by white space
SENTENCE_END_RE = re.compile(r"[.!?]+[\"')\]]*\s")

# where the spoken text starts in a streamed JSON reply
REPLY_KEY_RE = re.compile(r'"(?:reply|response|say|text)"\s*:\s*"')

_ESCAPES = {'"': '"', "\\": "\\", "/": "/", "b": "", "f": "", "n": " ", "r": " ", "t": " "}



class ReplyStream(object):

	"""
	Incremental parser for a streamed reply in the tools.reply_schema
	order (emotion, valence, arousal, actions, reply). feed() returns
	events as soon as they are certain:

	  ("header", LlmReply)   everything before the reply text: emotion,
	                         valence, arousal and the actions (reply "")
	  ("sentence", text)     the next sentence(s) of the spoken reply,
	                         at least min_chars long (except the last)

	finish() returns the remaining events and sets .reply to the full
	parse_llm_reply() of everything received (the fallback for a model
	that ignored the order or the format: then there is no header
	event and the whole reply comes at the end).
	"""

	def __init__(self, min_chars=20):

		self.min_chars = int(min_chars)
		self.raw = ""
		self.header = None
		self.reply = None
		self._start = None       # index in raw where the reply string starts
		self._pos = None         # next raw index to decode
		self._text = ""          # decoded, not yet emitted
		self._done = False       # closing quote seen
		self._emitted = 0

	def feed(self, piece):

		events = []
		self.raw += piece or ""
		if self._start is None:
			m = REPLY_KEY_RE.search(self.raw)
			if m is None:
				return events
			self._start = self._pos = m.end()
			head = self.raw[:m.start()].rstrip().rstrip(",") + "}"
			reply = parse_llm_reply(head)
			if reply.emotion is not None or any(reply.commands.get(k) for k in COMMAND_FLAGS):
				self.header = reply
				events.append(("header", reply))
		self._decode()
		events.extend(self._sentences(final=False))
		return events

	def finish(self):

		self._decode()
		events = self._sentences(final=True)
		self.reply = parse_llm_reply(self.raw)
		if self._start is None:
			# never saw the reply key: everything comes from the full parse
			if self.reply.reply:
				events.append(("sentence", self.reply.reply))
		return events

	def emitted_any(self):

		return self._emitted > 0

	def _decode(self):

		# decode the JSON string from _pos up to the last complete
		# character (an escape split across pieces waits for the rest)
		if self._start is None or self._done:
			return
		raw = self.raw
		i = self._pos
		out = []
		while i < len(raw):
			c = raw[i]
			if c == '"':
				self._done = True
				i += 1
				break
			if c == "\\":
				if i + 1 >= len(raw):
					break
				e = raw[i + 1]
				if e == "u":
					if i + 6 > len(raw):
						break
					try:
						out.append(chr(int(raw[i + 2:i + 6], 16)))
					except ValueError:
						pass
					i += 6
					continue
				out.append(_ESCAPES.get(e, e))
				i += 2
				continue
			out.append(c)
			i += 1
		self._pos = i
		self._text += "".join(out)

	def _sentences(self, final):

		events = []
		while True:
			cut = None
			for m in SENTENCE_END_RE.finditer(self._text):
				if m.end() >= self.min_chars:
					cut = m.end()
					break
			if cut is None:
				break
			self._emit(events, self._text[:cut])
			self._text = self._text[cut:]
		if (final or self._done) and self._text.strip():
			self._emit(events, self._text)
			self._text = ""
		return events

	def _emit(self, events, text):

		text = clean_text(text)
		if text:
			events.append(("sentence", text))
			self._emitted += 1
