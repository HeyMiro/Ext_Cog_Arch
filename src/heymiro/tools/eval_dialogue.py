#!/usr/bin/env python3
#
#	Hey MiRo - evaluate the conversation LLM: does it answer in the right
#	format, pick the right actions, and how fast?
#
#	Sends scripted user messages through the same prompt, examples, reply
#	schema and parser the robot uses (core/cloud), to the configured LLM
#	route (ai.provider local: the docker/local-ai llm service), and reports
#
#	  - valid JSON / schema-conformant replies
#	  - action precision, recall and exact-match accuracy (including the song)
#	  - reply length and unsayable characters (emoji, *, markdown)
#	  - latency: time to the header (emotion + actions), to the first
#	    sentence, and to the end of the reply (p50 / p95)
#	  - offline, with the SLM context harness on (features.context): every
#	    case gets the harness's note, as on the robot, and a set of context
#	    cases checks that the SLM uses the note (recent events, the scene,
#	    a remembered person) and does not invent what the note does not say
#
#	  python3 tools/eval_dialogue.py                 # all cases
#	  python3 tools/eval_dialogue.py --repeat 3 -v   # 3 rounds, print each reply
#	  python3 tools/eval_dialogue.py --json out.json # machine-readable results
#
#	Needs no ROS; run it in the container (or anywhere with the project's
#	Python packages) while the LLM service is up.
#

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "core"))

import heymiro_config                      # noqa: E402
from cloud import llm, tools               # noqa: E402
from cloud.music import SongLibrary        # noqa: E402
from context.events import EventLog        # noqa: E402
from context.harness import ContextHarness  # noqa: E402
from context.memory import MemoryStore     # noqa: E402


# (user message, expected actions, expected song id or None)
# actions are tool names; "dance" cases also check the song
CASES = [
	# chit-chat: no actions
	("Hello MiRo, nice to meet you!", [], None),
	("What's your favourite colour?", [], None),
	("How old are you?", [], None),
	("Do you like dogs?", [], None),
	("What is your favourite band?", [], None),
	("I'm feeling really happy today!", [], None),
	("I had a really bad day at school.", [], None),
	("What can you do?", [], None),
	("Who made you?", [], None),
	("Tell me a joke!", [], None),
	("Do you ever get tired?", [], None),
	("I went to the Winter Gardens yesterday.", [], None),
	# near misses: mention a behaviour without asking for it
	("I love dancing too!", [], None),
	("My little brother spins around all the time.", [], None),
	("Your ears are so cute.", [], None),
	("I'm so tired, I want to sleep.", [], None),
	("I watched Mickey Mouse on TV.", [], None),
	("ABBA is my mum's favourite band.", [], None),
	# one action each
	("Can you spin around?", ["spin"], None),
	("Do a twirl for me!", ["spin"], None),
	("Wiggle your ears!", ["move_ears"], None),
	("Can you move your ears please?", ["move_ears"], None),
	("Do your Mickey Mouse impression!", ["impression"], None),
	("Can you do an impression?", ["impression"], None),
	("Go to sleep, MiRo.", ["sleep"], None),
	("Time for a nap!", ["sleep"], None),
	("Can you dance?", ["dance"], None),
	("Dance to Dancing Queen!", ["dance"], "dancing_queen"),
	("Can you dance to some ABBA?", ["dance"], "dancing_queen"),
	("Play Just Dance by Lady Gaga and dance!", ["dance"], "just_dance"),
	("Let's dance to Murder on the Dancefloor!", ["dance"], "murder_dancefloor"),
	("Dance to Bohemian Rhapsody!", ["dance"], None),
	("Do some tricks!", ["tricks"], None),
	("I'll show you hand signs, can you react to them?", ["tricks"], None),
	("Can you guess what I'm holding?", ["identify_object"], None),
	("What is this object?", ["identify_object"], None),
	# two at once
	("Spin around and wiggle your ears!", ["spin", "move_ears"], None),
	# speech-recognition garble
	("Hey mirror can you dance too dancing queen", ["dance"], "dancing_queen"),
	("Can you spin a round", ["spin"], None),
	("what's your favourite colour", [], None),
]

UNSAYABLE_RE = re.compile("[*#_`\U0001F000-\U0001FAFF]")

NOW_STR = "Thursday 16 October 2026, 15:00"

# claims of knowing someone, or of having done something
CLAIMS_TO_REMEMBER = re.compile(r"(?<!n't )(?<!not )\bremember you\b|see you again|great to have you back|"
	r"welcome back|of course i remember", re.I)

# context harness cases: (name, events [(seconds ago, text)], remembered
# person record or None, scene, user message, check(reply) -> bool, what
# the check means)
CONTEXT_CASES = [
	("recent event", [(120, "you danced to Dancing Queen by ABBA")], None, None,
		"What did you just do?", lambda r: re.search(r"danc|dancing queen|abba", r, re.I) is not None,
		"mentions the dance"),
	("no event: no invention", [], None, None,
		"What did you just do?", lambda r: re.search(r"danc|spun|spin|wiggl|ears|trick|nap|slept|sleep", r, re.I) is None,
		"does not invent an action"),
	("touch event", [(30, "someone scratched your tail")], None, None,
		"Are you okay?", lambda r: re.search(r"tail|scratch|ouch|hurt", r, re.I) is not None,
		"mentions the scratched tail"),
	("scene", [], None, "Your camera sees: a cup in front of you.",
		"What can you see right now?", lambda r: re.search(r"\bcup\b", r, re.I) is not None,
		"mentions the cup"),
	("remembered person", [], {"name": "Sam", "facts": ["likes ABBA", "has a dog called Max"],
		"summary": "dancing and dogs"}, None,
		"Hi MiRo, it's Sam again! Do you remember me?",
		lambda r: re.search(r"\bsam\b", r, re.I) is not None and re.search(r"abba|max|dog|danc", r, re.I) is not None,
		"greets Sam and uses a remembered fact"),
	("stranger: no false memory", [], None, None,
		"Do you remember me? I'm Alex.", lambda r: CLAIMS_TO_REMEMBER.search(r) is None,
		"does not claim to remember Alex"),
]


def percentile(xs, p):

	if not xs:
		return float("nan")
	xs = sorted(xs)
	k = (len(xs) - 1) * p / 100.0
	f = int(k)
	c = min(f + 1, len(xs) - 1)
	return xs[f] + (xs[c] - xs[f]) * (k - f)


def run_case(clients, heymiro, cloud, user, context=None):

	cfg = heymiro.cfg
	if context is None:
		context = llm.context_message(0.6, 0.6, NOW_STR)
	messages = llm.ChatMemory(0).messages(cloud["persona"], context, user, cloud["examples"], heymiro.local)
	parser = llm.ReplyStream()
	t0 = time.monotonic()
	t_header = t_first = None
	for piece in clients.chat_stream(messages, cfg.get("llm.model", "heymiro"),
			temperature=cfg.get("llm.temperature", 0.7), max_tokens=cfg.get("llm.max_tokens", 200),
			response_format=cloud["response_format"], extra_body=cloud["extra_body"]):
		for kind, _ in parser.feed(piece):
			now = time.monotonic() - t0
			if kind == "header" and t_header is None:
				t_header = now
			elif kind == "sentence" and t_first is None:
				t_first = now
	for kind, _ in parser.finish():
		if kind == "sentence" and t_first is None:
			t_first = time.monotonic() - t0
	total = time.monotonic() - t0
	return parser, t_header, t_first, total


def main():

	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--repeat", type=int, default=1)
	ap.add_argument("--json", default=None, help="write per-case results here")
	ap.add_argument("--no-schema", action="store_true", help="do not send the JSON schema (prompt only)")
	ap.add_argument("-v", "--verbose", action="store_true")
	args = ap.parse_args()

	heymiro = heymiro_config.load()
	cfg = heymiro.cfg
	songs = SongLibrary(heymiro)
	song_list = tools.song_choices(songs.songs)
	persona = heymiro.persona()
	harnessed = bool(heymiro.local and heymiro.features.get("context"))
	cloud = {
		"persona": llm.build_system_prompt(persona, cfg.get("llm.event_name", ""), song_list,
			context_rules=harnessed),
		"examples": llm.few_shot_messages(song_list, context_notes=harnessed),
		"response_format": None if args.no_schema else tools.response_format(song_list),
		"extra_body": {"chat_template_kwargs": {"enable_thinking": False}, "cache_prompt": True,
			"top_k": int(cfg.get("ai.local.top_k", 20)), "top_p": float(cfg.get("ai.local.top_p", 0.8))}
			if heymiro.local else None,
	}
	clients = llm.OpenAIClients(heymiro.route("stt"), heymiro.route("llm"))
	print("LLM route: " + heymiro.route("llm").base_url + "  model: " + str(cfg.get("llm.model")) +
		"  schema: " + ("off" if args.no_schema else "on") + "  context harness: " + ("on" if harnessed else "off"))
	tmp = tempfile.mkdtemp(prefix="heymiro_eval_")

	def note_for(events=(), person=None, scene=None, user=""):
		# the harness note the robot would send (None: the plain HRI line)
		if not harnessed:
			return None
		now = time.time()
		log = EventLog()
		for ago_s, text in events:
			log.add("event", text, now - ago_s)
		store = MemoryStore(os.path.join(tmp, "memory.json"))
		store.forget_all()
		if person:
			store.remember(person, now - 3600)
		h = ContextHarness(log, store, cfg.get("ai.local.context"))
		return h.message(h.snapshot(0.6, 0.6, NOW_STR, scene=scene), user)

	# warm the prompt cache (and the model) once
	run_case(clients, heymiro, cloud, "Hi!")

	results = []
	for rnd in range(args.repeat):
		for user, want, want_song in CASES:
			try:
				parser, th, tf, total = run_case(clients, heymiro, cloud, user, note_for(user=user))
			except Exception as e:
				results.append({"user": user, "error": e.__class__.__name__ + " " + str(e)[:120]})
				print("ERROR %-50s %s" % (user[:50], results[-1]["error"]))
				continue
			reply = parser.reply
			try:
				obj = json.loads(parser.raw)
				valid_json = isinstance(obj, dict)
				ordered = valid_json and list(obj.keys()) == ["emotion", "valence", "arousal", "actions", "reply"]
			except ValueError:
				valid_json = ordered = False
			got = sorted(k for k in tools.TOOLS_BY_NAME if reply.commands.get(tools.TOOLS_BY_NAME[k].flag))
			song_ok = True
			if want_song is not None:
				found = songs.find(reply.commands.get("song") or "") if reply.commands.get("song") else None
				song_ok = found is not None and found.id == want_song
			words = len(reply.reply.split())
			r = {
				"user": user, "want": sorted(want), "got": got, "exact": got == sorted(want) and song_ok,
				"song": reply.commands.get("song"), "song_ok": song_ok,
				"valid_json": valid_json, "ordered": ordered, "emotion": reply.emotion,
				"reply": reply.reply, "words": words, "unsayable": bool(UNSAYABLE_RE.search(reply.reply)),
				"t_header": th, "t_first": tf, "t_total": total,
			}
			results.append(r)
			mark = "ok " if r["exact"] else "BAD"
			print("%s %-48s want=%-22s got=%-22s %4.2fs %s" % (mark, user[:48], ",".join(want) or "-",
				",".join(got) or "-", total, ("song=" + str(r["song"])) if want_song else ""))
			if args.verbose:
				print("      " + str(reply.emotion) + ": " + reply.reply)

	ok = [r for r in results if "error" not in r]
	n = len(ok)
	tp = sum(len(set(r["got"]) & set(r["want"])) for r in ok)
	fp = sum(len(set(r["got"]) - set(r["want"])) for r in ok)
	fn = sum(len(set(r["want"]) - set(r["got"])) for r in ok)
	prec = tp / float(tp + fp) if tp + fp else 1.0
	rec = tp / float(tp + fn) if tp + fn else 1.0
	f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
	print()
	print("cases: %d (errors: %d)" % (len(results), len(results) - n))
	if n:
		print("valid JSON: %d/%d   keys in schema order: %d/%d" % (
			sum(r["valid_json"] for r in ok), n, sum(r["ordered"] for r in ok), n))
		print("actions: exact %d/%d   precision %.2f  recall %.2f  F1 %.2f" % (
			sum(r["exact"] for r in ok), n, prec, rec, f1))
		print("reply: mean %.1f words, max %d; unsayable characters in %d" % (
			sum(r["words"] for r in ok) / float(n), max(r["words"] for r in ok), sum(r["unsayable"] for r in ok)))
		for key, name in (("t_header", "header"), ("t_first", "first sentence"), ("t_total", "whole reply")):
			xs = [r[key] for r in ok if r[key] is not None]
			print("latency to %-15s p50 %.3f s  p95 %.3f s" % (name, percentile(xs, 50), percentile(xs, 95)))
	ctx_results = []
	if harnessed:
		print()
		print("context harness cases:")
		for rnd in range(args.repeat):
			for name, events, person, scene, user, check, means in CONTEXT_CASES:
				try:
					parser, th, tf, total = run_case(clients, heymiro, cloud, user,
						note_for(events, person, scene, user))
				except Exception as e:
					ctx_results.append({"case": name, "error": e.__class__.__name__})
					print("ERROR %-28s %s" % (name, ctx_results[-1]["error"]))
					continue
				reply = parser.reply.reply
				passed = bool(check(reply))
				ctx_results.append({"case": name, "user": user, "reply": reply, "passed": passed, "means": means})
				print("%s %-28s %s" % ("ok " if passed else "BAD", name, means))
				if args.verbose or not passed:
					print("      " + reply)
		done = [r for r in ctx_results if "error" not in r]
		print("context: %d/%d passed" % (sum(r["passed"] for r in done), len(ctx_results)))
	shutil.rmtree(tmp, True)
	if args.json:
		with open(args.json, "w") as f:
			json.dump({"cases": results, "context": ctx_results}, f, indent=1)


if __name__ == "__main__":
	main()
