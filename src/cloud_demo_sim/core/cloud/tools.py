#
#	Hey MiRo - the behaviours the LLM may ask for ("tools"), in ONE place.
#
#	The registry below generates
#	  - the JSON Schema of the reply, sent to servers that support
#	    grammar-constrained decoding (llama.cpp, vLLM; json_schema), so a
#	    local model can only produce a valid reply: an emotion from the
#	    list, valence/arousal, at most two known actions with valid
#	    arguments (dance songs come from config/songs.yaml), and the text;
#	  - the "What you can do" section of the system prompt, so the prompt
#	    and the schema can never disagree;
#	  - the mapping of the "actions" list onto LlmReply.commands, the
#	    flags the dialogue and the actions already understand.
#
#	The reply fields come in a fixed order: emotion, valence, arousal,
#	actions, reply. A streamed reply therefore tells the body what to do
#	(pose, lights, which behaviour) after ~25 tokens, before the first
#	word of speech is generated.
#

import json


# emotions MiRo may report (HRI list, lower case); kept in sync with
# cloud.llm.EMOTIONS by the tests
EMOTIONS = ("happy", "fine", "angry", "worried", "sad", "scared")

# song value meaning "no particular song" (or one MiRo does not have)
ANY_SONG = "any"

MAX_ACTIONS = 2
MAX_REPLY_CHARS = 240


class Tool(object):

	def __init__(self, name, flag, does, use_when, not_when, args=None):

		self.name = name            # what the model writes
		self.flag = flag            # LlmReply.commands key it sets
		self.does = does
		self.use_when = use_when
		self.not_when = not_when
		self.args = args or {}      # arg name -> description


TOOLS = (
	Tool("spin", "spin", "spin around on the spot",
		"they ask you to spin, turn around or do a twirl",
		"they only talk about spinning"),
	Tool("move_ears", "move_ears", "wiggle your ears",
		"they ask you to move, wiggle or flap your ears",
		"they only mention or compliment your ears"),
	Tool("impression", "impression", "do your Mickey Mouse impression (then write the reply the way Mickey Mouse would say it)",
		"they ask for an impression or for Mickey Mouse",
		"they only talk about Mickey Mouse or cartoons"),
	Tool("sleep", "sleep", "go to sleep for a while",
		"they tell you to go to sleep, take a nap or rest",
		"they say THEY are tired or sleepy, or ask whether you sleep"),
	Tool("dance", "dance", "dance to a song",
		"they ask you to dance, sing or play music",
		"they only say they like dancing or music",
		args={"song": "the song they asked for, or \"any\" if they did not say or you do not have it"}),
	Tool("tricks", "tricks", "do tricks: they show you hand signs and you react",
		"they ask you to do tricks or to watch their hand signs",
		"they only ask what you can do"),
	Tool("identify_object", "identify_object", "guess an object they show you to your camera",
		"they ask you to guess, identify or look at something they are holding, or ask what an object is",
		"they only mention an object in conversation"),
)

TOOLS_BY_NAME = dict((t.name, t) for t in TOOLS)

# names a model may plausibly use instead (e.g. the old command flags)
TOOL_ALIASES = {
	"spin_around": "spin", "twirl": "spin",
	"ears": "move_ears", "wiggle_ears": "move_ears",
	"mickey_mouse": "impression", "do_impression": "impression",
	"go_to_sleep": "sleep", "go_sleep": "sleep", "nap": "sleep",
	"play_music": "dance", "sing": "dance",
	"gesture_mode": "tricks", "do_tricks": "tricks",
	"identify": "identify_object", "guess_object": "identify_object", "obj_detect_mode": "identify_object",
}


def song_choices(songs):

	"""
	The dance tool's song values: "Title - Artist" for each bundled song
	(SongLibrary.songs, or anything with .title / .artist), then "any".
	"""

	out = []
	for s in songs or ():
		title = str(getattr(s, "title", "") or "").strip()
		if not title:
			continue
		artist = str(getattr(s, "artist", "") or "").strip()
		label = title + (" - " + artist if artist else "")
		if label not in out:
			out.append(label)
	out.append(ANY_SONG)
	return out


def reply_schema(song_list=None):

	"""JSON Schema of one reply; property order is the generation order."""

	actions = []
	for t in TOOLS:
		props = {"tool": {"const": t.name}}
		required = ["tool"]
		if "song" in t.args:
			props["song"] = {"type": "string", "enum": list(song_list or [ANY_SONG])}
			required.append("song")
		actions.append({"type": "object", "properties": props, "required": required,
			"additionalProperties": False})
	return {
		"type": "object",
		"properties": {
			"emotion": {"type": "string", "enum": list(EMOTIONS)},
			"valence": {"type": "number", "minimum": 0, "maximum": 1},
			"arousal": {"type": "number", "minimum": 0, "maximum": 1},
			"actions": {"type": "array", "items": {"anyOf": actions}, "maxItems": MAX_ACTIONS},
			"reply": {"type": "string", "minLength": 1, "maxLength": MAX_REPLY_CHARS},
		},
		"required": ["emotion", "valence", "arousal", "actions", "reply"],
		"additionalProperties": False,
	}


def response_format(song_list=None):

	# OpenAI-style structured output (llama.cpp and vLLM accept it too)
	return {"type": "json_schema", "json_schema": {"name": "miro_reply", "schema": reply_schema(song_list)}}


def prompt_section(song_list=None):

	"""The "What you can do" part of the system prompt, from the registry."""

	lines = ["What you can do (\"actions\"):",
		"Add an action ONLY when the person's latest message asks or tells YOU to do it now (\"can you ...\", \"do ...\", \"show me ...\").",
		"Talking about something, compliments and stories are not requests. Most replies have no actions: []."]
	for t in TOOLS:
		line = "- \"" + t.name + "\""
		if t.args:
			line += "(" + ", ".join(t.args) + ")"
		line += ": " + t.does + ". Use when " + t.use_when + "; not when " + t.not_when + "."
		lines.append(line)
		for arg, desc in sorted(t.args.items()):
			lines.append("    " + arg + ": " + desc + ".")
	songs = [s for s in (song_list or []) if s != ANY_SONG]
	if songs:
		lines.append("Songs you have: " + "; ".join(songs) + ".")
	return "\n".join(lines)


def example_reply(emotion, valence, arousal, reply, actions=()):

	# one assistant turn in the exact output format (for few-shot examples)
	return json.dumps({"emotion": emotion, "valence": valence, "arousal": arousal,
		"actions": list(actions), "reply": reply}, ensure_ascii=False)


def tool_name(value):

	name = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
	name = TOOL_ALIASES.get(name, name)
	return name if name in TOOLS_BY_NAME else None


def actions_to_commands(actions, commands):

	"""
	Apply an "actions" list ([{"tool": "dance", "song": "..."}, ...], or
	bare names) to a commands dict in place; unknown tools are ignored,
	at most MAX_ACTIONS are used. Returns the tool names applied.
	"""

	applied = []
	if not isinstance(actions, (list, tuple)):
		actions = [actions]
	for item in actions:
		if len(applied) >= MAX_ACTIONS:
			break
		if isinstance(item, dict):
			name = tool_name(item.get("tool") or item.get("name"))
			args = item.get("arguments") if isinstance(item.get("arguments"), dict) else item
		else:
			name, args = tool_name(item), {}
		if name is None or name in applied:
			continue
		tool = TOOLS_BY_NAME[name]
		commands[tool.flag] = True
		if "song" in tool.args:
			song = str(args.get("song") or "").strip()
			if song and song.lower() != ANY_SONG:
				# "Title - Artist" -> the library's fuzzy match handles both
				commands["song"] = song
		applied.append(name)
	return applied
