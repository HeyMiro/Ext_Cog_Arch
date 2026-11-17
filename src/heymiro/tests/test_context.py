#
#	Tests for the SLM context harness (core/context): recent events, the
#	memory of people across conversations, the per-turn note, and that
#	the online pipeline is left exactly as it was.
#

import _testpath

import json
import os
import shutil
import tempfile
import unittest

import numpy as np

from context.events import EventLog, ago
from context import memory as mem
from context.harness import ContextHarness, Snapshot
from context import budget as bud
from cloud.llm import ChatMemory, build_system_prompt, context_message, CONTEXT_INSTRUCTIONS, ACTION_REMINDER
from forebrain import dialogue as dlg
from test_config import ConfigTestCase



class Clock(object):

	def __init__(self, t=1000000.0):
		self.t = t

	def __call__(self):
		return self.t



class TestEvents(unittest.TestCase):

	def test_ago(self):

		self.assertEqual(ago(5), "just now")
		self.assertEqual(ago(42), "40 seconds ago")
		self.assertEqual(ago(60), "a minute ago")
		self.assertEqual(ago(130), "2 minutes ago")
		self.assertEqual(ago(3700), "an hour ago")
		self.assertEqual(ago(3 * 86400), "3 days ago")

	def test_newest_first_and_age_limit(self):

		log = EventLog()
		log.add("task", "you danced to Dancing Queen by ABBA", now=100)
		log.add("touch", "someone scratched your tail", now=500)
		self.assertEqual(log.recent(now=520, max_age_s=600),
			["someone scratched your tail (just now)", "you danced to Dancing Queen by ABBA (7 minutes ago)"])
		self.assertEqual(log.recent(now=900, max_age_s=600), ["someone scratched your tail (7 minutes ago)"])
		self.assertIsNone(log.describe(now=5000, max_age_s=600))

	def test_repeats_merge_and_max_items(self):

		log = EventLog()
		for t in (10, 20, 30):
			log.add("touch", "someone stroked you", now=t)
		log.add("object", "you saw a cup in the object game", now=35)
		self.assertEqual(log.recent(now=40), ["you saw a cup in the object game (just now)",
			"someone stroked you (3 times, just now)"])
		self.assertEqual(len(log.recent(now=40, max_items=1)), 1)
		self.assertEqual(log.describe(now=40, max_items=1), "Recently: you saw a cup in the object game (just now).")

	def test_blank_lines_are_ignored_and_clear(self):

		log = EventLog()
		log.add("x", "   ", now=1)
		self.assertEqual(len(log), 0)
		log.add("x", "something", now=1)
		log.clear()
		self.assertEqual(log.recent(now=2), [])



class MemoryTestCase(unittest.TestCase):

	def setUp(self):

		self.tmp = tempfile.mkdtemp(prefix="heymiro_mem_")
		self.addCleanup(shutil.rmtree, self.tmp, True)
		self.path = os.path.join(self.tmp, "memory", "memory.json")
		self.clock = Clock()

	def store(self, **kw):

		return mem.MemoryStore(self.path, clock=self.clock, **kw)



class TestMemoryStore(MemoryTestCase):

	def test_remember_persist_and_describe(self):

		s = self.store()
		self.assertEqual(s.remember({"name": "sam", "facts": ["likes ABBA", "has a dog called Max"],
			"summary": "talked about dancing"}), "Sam")
		self.clock.t += 3600
		again = self.store()
		self.assertEqual(again.names(), ["Sam"])
		self.assertEqual(again.describe("Sam"), "You remember Sam (you last talked an hour ago): "
			"likes ABBA; has a dog called Max. Last time you talked about dancing.")
		# the file is plain JSON, written atomically (no temp files left)
		self.assertEqual(json.load(open(self.path))["version"], mem.VERSION)
		self.assertEqual(os.listdir(os.path.dirname(self.path)), ["memory.json"])

	def test_merge_facts_newest_first_and_cap(self):

		s = self.store(max_facts=3)
		s.remember({"name": "Sam", "facts": ["likes ABBA", "is 8"], "summary": "a"})
		s.remember({"name": "SAM", "facts": ["has a cat", "Likes ABBA"], "summary": "b"})
		p = s.person("sam")
		self.assertEqual(p["facts"], ["has a cat", "Likes ABBA", "is 8"])
		self.assertEqual(p["times"], 2)
		self.assertEqual(p["summary"], "b")

	def test_ttl_forgets_people(self):

		s = self.store(ttl_hours=1)
		s.remember({"name": "Sam", "facts": [], "summary": "hi"})
		self.clock.t += 2 * 3600
		self.assertEqual(self.store(ttl_hours=1).names(), [])

	def test_anonymous_conversations_are_episodes_only(self):

		s = self.store()
		self.assertIsNone(s.remember({"name": None, "facts": ["likes cake"], "summary": "talked about cake"}))
		self.assertEqual(s.stats(), {"people": 0, "episodes": 1})
		self.assertIsNone(s.find_name("I like cake too"))

	def test_find_name_whole_words_only(self):

		s = self.store()
		s.remember({"name": "Sam", "facts": [], "summary": ""})
		self.assertEqual(s.find_name("Hi MiRo, it's Sam again!"), "Sam")
		self.assertEqual(s.find_name("do you remember me? i'm sam"), "Sam")
		self.assertIsNone(s.find_name("Samantha says hello"))
		self.assertIsNone(s.find_name("same again"))

	def test_max_people_drops_the_oldest(self):

		s = self.store(max_people=2)
		for i, n in enumerate(("Ann", "Bob", "Cat")):
			self.clock.t += 10
			s.remember({"name": n, "facts": [], "summary": ""})
		self.assertEqual(sorted(s.names()), ["Bob", "Cat"])

	def test_forget_all(self):

		s = self.store()
		s.remember({"name": "Sam", "facts": [], "summary": ""})
		s.forget_all()
		self.assertEqual(self.store().names(), [])

	def test_corrupt_file_starts_empty(self):

		os.makedirs(os.path.dirname(self.path))
		with open(self.path, "w") as f:
			f.write("{not json")
		self.assertEqual(self.store().names(), [])



class TestConsolidation(unittest.TestCase):

	def test_parse_valid_and_fenced(self):

		r = mem.parse_consolidation('```json\n{"name": "sam", "facts": ["likes ABBA.", "likes ABBA"], '
			'"summary": "dancing"}\n```')
		self.assertEqual(r, {"name": "Sam", "facts": ["likes ABBA"], "summary": "dancing"})

	def test_parse_rejects_non_names_and_garbage(self):

		self.assertIsNone(mem.parse_consolidation('{"name": "MiRo", "facts": [], "summary": ""}')["name"])
		self.assertIsNone(mem.parse_consolidation('{"name": "two words", "facts": [], "summary": ""}')["name"])
		self.assertIsNone(mem.parse_consolidation("no json here"))
		self.assertIsNone(mem.parse_consolidation("[1, 2]"))
		self.assertEqual(mem.parse_consolidation('{"facts": ["a", "b", "c"]}', max_facts=2)["facts"], ["a", "b"])

	def test_messages_show_the_person_and_only_the_question_before_a_short_answer(self):

		r = lambda t: json.dumps({"emotion": "happy", "valence": 0.8, "arousal": 0.7, "actions": [], "reply": t})
		msgs = mem.consolidation_messages([
			("Hi MiRo, I'm Leo.", r("Hi Leo! I love red. Do you like dinosaurs?")),
			("Yes! I have twenty of them.", r("WOW! I love the T-rex.")),
			("What is your favourite colour?", r("Red! Which is your favourite?")),
		])
		self.assertEqual(msgs[0]["role"], "system")
		self.assertEqual(msgs[1]["content"],
			"Person: Hi MiRo, I'm Leo.\n"
			"(MiRo asked: Do you like dinosaurs?)\n"
			"Person: Yes! I have twenty of them.\n"
			"Person: What is your favourite colour?")
		# MiRo's own likes are never shown
		self.assertNotIn("I love red", msgs[1]["content"])

	def test_question_facts_are_dropped(self):

		r = mem.parse_consolidation('{"name": "", "facts": ["asks about music", "has a cat"], "summary": ""}')
		self.assertEqual(r["facts"], ["has a cat"])

	def test_schema(self):

		fmt = mem.consolidation_format(3)
		schema = fmt["json_schema"]["schema"]
		self.assertEqual(fmt["type"], "json_schema")
		self.assertEqual(schema["properties"]["facts"]["maxItems"], 3)
		self.assertEqual(schema["required"], ["name", "facts", "summary"])



class TestHarness(MemoryTestCase):

	def harness(self, chat=None, cfg=None):

		self.events = EventLog(clock=self.clock)
		self.memory = self.store()
		return ContextHarness(self.events, self.memory, cfg, chat, clock=self.clock)

	def test_full_note_in_order(self):

		h = self.harness()
		self.memory.remember({"name": "Sam", "facts": ["likes ABBA"], "summary": "dancing"})
		self.events.add("task", "you danced to Dancing Queen by ABBA")
		snap = h.snapshot(0.8, 0.7, "Monday 17 November 2026, 14:32", True, "dance",
			"Your camera sees: a person in front of you.")
		note = h.note(snap, "Hi MiRo, it's Sam again!")
		self.assertEqual(note,
			"Now: Monday 17 November 2026, 14:32. "
			"You are awake and feeling happy (valence 0.80, arousal 0.70); you are dancing. "
			"Recently: you danced to Dancing Queen by ABBA (just now). "
			"Your camera sees: a person in front of you. "
			"You remember Sam (you last talked just now): likes ABBA. Last time you talked about dancing. "
			+ ACTION_REMINDER)

	def test_empty_sections_are_left_out(self):

		h = self.harness()
		note = h.note(h.snapshot(None, None, None), "hello")
		self.assertEqual(note, "You are awake. Recently: nothing; you have just been listening. " + ACTION_REMINDER)
		bare = ContextHarness(clock=self.clock)
		self.assertEqual(bare.note(bare.snapshot(None, None, None), "hello"), "You are awake. " + ACTION_REMINDER)

	def test_self_state_off_keeps_the_hri_affect_line(self):

		h = self.harness(cfg={"self_state": False})
		note = h.note(h.snapshot(0.5, 0.5, "noon"), "hello")
		self.assertEqual(note, "Now: noon. Your current valence is 0.50 and your current arousal is 0.50. "
			"Recently: nothing; you have just been listening. " + ACTION_REMINDER)

	def test_recall_needs_a_name_and_sticks_for_the_conversation(self):

		h = self.harness()
		self.memory.remember({"name": "Sam", "facts": ["likes ABBA"], "summary": ""})
		snap = h.snapshot(None, None, None)
		self.assertNotIn("remember", h.note(snap, "Do you remember me?"))
		self.assertIn("You remember Sam", h.note(snap, "I'm Sam!"))
		self.assertIn("You remember Sam", h.note(snap, "What's my dog called?"))
		self.assertEqual(h.end_conversation(), "Sam")
		self.assertNotIn("remember", h.note(snap, "Do you remember me?"))

	def test_long_sections_are_capped(self):

		h = self.harness()
		snap = Snapshot(scene="Your camera sees: " + "a cup, " * 100)
		note = h.note(snap, "hi")
		self.assertLess(len(note), 400)
		self.assertIn("...", note)

	def test_consolidate_stores_and_uses_the_recalled_name(self):

		answers = ['{"name": "", "facts": ["has a dog called Max"], "summary": "dogs"}']
		seen = []

		def chat(messages, fmt):
			seen.append((messages, fmt))
			return answers.pop(0)

		h = self.harness(chat=chat)
		raw = json.dumps({"reply": "Woof!"})
		self.assertEqual(h.consolidate([("I have a dog called Max", raw)], known="Sam"), "Sam")
		self.assertEqual(self.memory.person("Sam")["facts"], ["has a dog called Max"])
		self.assertEqual(seen[0][1]["json_schema"]["name"], "miro_memory")

	def test_consolidate_never_raises(self):

		def broken(messages, fmt):
			raise RuntimeError("server down")

		h = self.harness(chat=broken)
		self.assertIsNone(h.consolidate([("hi", "{}")]))
		h = self.harness(chat=lambda m, f: "not json")
		self.assertIsNone(h.consolidate([("hi", "{}")]))
		self.assertIsNone(self.harness(chat=None).consolidate([("hi", "{}")]))
		self.assertEqual(self.memory.names(), [])



class TestBudget(unittest.TestCase):

	def test_history_budget_arithmetic(self):

		b = bud.TokenBudget(n_ctx=8192, prefix_tokens=1900, reserve_reply=200, reserve_user=150, reserve_note=250,
			margin=0.05)
		self.assertEqual(b.history_tokens, int(8192 - 1900 - 200 - 150 - 250 - 0.05 * 8192))
		self.assertEqual(b.low_tokens, b.history_tokens // 2)
		self.assertTrue(b.measured)

	def test_estimates_and_unmeasured_prefix(self):

		b = bud.TokenBudget(chars_per_token=4.0, prefix_chars=8000)
		self.assertEqual(b.prefix_tokens, 2000)
		self.assertFalse(b.measured)
		self.assertEqual(b.estimate("abcd" * 10), 10)
		self.assertEqual(b.estimate(""), 0)

	def test_cap_user_keeps_the_beginning(self):

		b = bud.TokenBudget(chars_per_token=4.0, reserve_user=10)
		self.assertEqual(b.cap_user("short words"), "short words")
		long = "one two three four five six seven eight nine ten eleven twelve"
		capped = b.cap_user(long)
		self.assertTrue(capped.startswith("one two"))
		self.assertTrue(capped.endswith(bud.ELLIPSIS))
		self.assertLessEqual(len(capped), 40)

	def test_pick_turns_head_and_tail(self):

		turns = list(range(30))
		self.assertEqual(bud.pick_turns(turns, 4, 8), [0, 1, 2, 3] + list(range(22, 30)))
		self.assertEqual(bud.pick_turns(turns[:10], 4, 8), list(range(10)))

	def test_calibrate_with_a_fake_server(self):

		class Resp(object):
			def __init__(self, data):
				self.data = data

			def json(self):
				return self.data

		class Session(object):
			def get(self, url, timeout=None):
				assert url == "http://x:8080/props"
				return Resp({"default_generation_settings": {"n_ctx": 8192}})

			def post(self, url, json=None, timeout=None):
				if url.endswith("/apply-template"):
					return Resp({"prompt": "p" * 400})
				return Resp({"tokens": list(range(100))})

		self.assertEqual(bud.calibrate("http://x:8080/v1", [{"role": "system", "content": "hi"}], session=Session()),
			(8192, 100, 4.0))

	def test_calibrate_never_raises(self):

		class Broken(object):
			def get(self, *a, **k):
				raise IOError("down")
			post = get

		self.assertEqual(bud.calibrate("http://x:8080/v1", [], session=Broken()), (None, None, None))

	def test_from_config(self):

		b = bud.from_config({"budget": {"reserve_user": 120, "high_turns": 12, "low_turns": 6}}, 180,
			n_ctx=4096, prefix_tokens=1000, chars_per_token=3.5)
		self.assertEqual((b.n_ctx, b.reserve_user, b.reserve_reply, b.high_turns, b.low_turns), (4096, 120, 180, 12, 6))



class TestHistoryHysteresis(unittest.TestCase):

	def budget(self, **kw):

		args = dict(n_ctx=8192, prefix_tokens=1900, chars_per_token=4.0, high_turns=16, low_turns=8)
		args.update(kw)
		return bud.TokenBudget(**args)

	def test_no_trim_below_high_then_one_trim_to_low(self):

		m = ChatMemory(10, budget=self.budget())
		for i in range(16):
			m.add("question %d" % i, "answer %d" % i)
		self.assertEqual(len(m), 16)
		self.assertEqual(m.trims, 0)
		m.add("question 16", "answer 16")
		self.assertEqual(len(m), 8)
		self.assertEqual(m.trims, 1)
		self.assertEqual(m.snapshot()[0][0], "question 9")

	def test_prefix_is_stable_between_trims(self):

		m = ChatMemory(10, budget=self.budget())
		for i in range(17):
			m.add("q%d" % i, "a%d" % i)
		first = m.messages("SYS", None, "now")[:3]
		for i in range(5):
			m.add("more%d" % i, "x")
			self.assertEqual(m.messages("SYS", None, "now")[:3], first)
		self.assertEqual(m.trims, 1)

	def test_token_limit_wins_over_turn_limit(self):

		b = self.budget(n_ctx=3000, prefix_tokens=1900)
		m = ChatMemory(10, budget=b)
		big = "word " * 120     # ~150 tokens per side
		for i in range(4):
			m.add(big, big)
		self.assertLessEqual(m.tokens(), b.history_tokens)
		self.assertGreaterEqual(m.trims, 1)

	def test_online_memory_is_unchanged(self):

		m = ChatMemory(3)
		for i in range(5):
			m.add("q%d" % i, "a%d" % i)
		self.assertEqual([u for u, _ in m.snapshot()], ["q2", "q3", "q4"])
		self.assertEqual(m.tokens(), 0)



class TestNoteBudget(MemoryTestCase):

	def test_drops_scene_then_events_then_memory(self):

		events = EventLog(clock=self.clock)
		events.add("x", "you danced to a very long song title " * 3)
		memory = self.store()
		memory.remember({"name": "Sam", "facts": ["likes ABBA"], "summary": "music"})
		h = ContextHarness(events, memory, clock=self.clock)
		snap = h.snapshot(0.5, 0.5, "noon", scene="Your camera sees: a cup, a person and a bottle.")
		full = h.note(snap, "It's Sam")
		self.assertIn("camera", full)
		h.budget = bud.TokenBudget(chars_per_token=4.0, reserve_note=len(full) // 4 - 5)
		less = h.note(snap, "It's Sam")
		self.assertNotIn("camera", less)
		self.assertIn("Recently", less)
		h.budget = bud.TokenBudget(chars_per_token=4.0, reserve_note=10)
		least = h.note(snap, "It's Sam")
		for gone in ("camera", "Recently", "remember"):
			self.assertNotIn(gone, least)
		self.assertTrue(least.startswith("Now: noon. You are awake"))
		self.assertTrue(least.endswith(ACTION_REMINDER))



class TestLongConversationMemory(MemoryTestCase):

	def test_a_name_from_turn_one_of_thirty_reaches_the_memory(self):

		seen = []

		def chat(messages, fmt):
			seen.append(messages[1]["content"])
			ok = "Person: Hi, I'm Sam" in messages[1]["content"]
			return json.dumps({"name": "Sam" if ok else "", "facts": [], "summary": "lots"})

		h = ContextHarness(None, self.store(), None, chat, clock=self.clock)
		turns = [("Hi, I'm Sam", json.dumps({"reply": "Hi Sam!"}))]
		turns += [("message %d" % i, json.dumps({"reply": "ok"})) for i in range(29)]
		self.assertEqual(h.consolidate(turns), "Sam")
		lines = seen[0].split("\n")
		self.assertEqual(sum(1 for l in lines if l.startswith("Person:")), mem.MAX_TURNS_IN_TRANSCRIPT)
		self.assertIn("Person: message 28", seen[0])


class FakeClients(object):

	def __init__(self, text):
		self.text = text
		self.messages = None

	def transcribe(self, wav, model="whisper-1", language=None):
		return self.text

	def chat(self, messages, model, temperature=None, max_tokens=None, json_mode=False):
		self.messages = messages
		return json.dumps({"emotion": "happy", "valence": 0.8, "arousal": 0.7, "actions": [], "reply": "Hi!"})



class TestTurnRunner(MemoryTestCase):

	def run_turn(self, context=None):

		clients = FakeClients("Hi MiRo, I'm Sam")
		runner = dlg.TurnRunner(clients, None, ChatMemory(), "SYSTEM", {}, context_in_user=True)
		runner(np.zeros(1600, dtype=np.float32), 0.5, 0.5, "noon", scene="Your camera sees: a cup.", context=context)
		return clients.messages

	def test_without_harness_the_hri_context_line_is_sent(self):

		msgs = self.run_turn()
		expected = context_message(0.5, 0.5, "noon", extra="Your camera sees: a cup.")["content"]
		self.assertEqual(msgs[-1]["content"], "[" + expected + "]\nHi MiRo, I'm Sam")

	def test_with_harness_the_note_is_sent(self):

		events = EventLog(clock=self.clock)
		events.add("touch", "someone stroked you")
		memory = self.store()
		memory.remember({"name": "Sam", "facts": ["likes ABBA"], "summary": ""})
		h = ContextHarness(events, memory, clock=self.clock)
		snap = h.snapshot(0.5, 0.5, "noon", scene="Your camera sees: a cup.")
		msgs = self.run_turn(context=(h, snap))
		last = msgs[-1]["content"]
		self.assertTrue(last.startswith("[Now: noon. You are awake and feeling fine"))
		self.assertIn("Recently: someone stroked you (just now).", last)
		self.assertIn("You remember Sam", last)
		self.assertTrue(last.endswith("]\nHi MiRo, I'm Sam"))



class TestSystemPrompt(unittest.TestCase):

	def test_context_rules_only_when_asked(self):

		plain = build_system_prompt("You are MiRo at $event_name.", "the fair")
		self.assertNotIn(CONTEXT_INSTRUCTIONS.strip(), plain)
		with_rules = build_system_prompt("You are MiRo at $event_name.", "the fair", context_rules=True)
		self.assertTrue(with_rules.startswith(plain))
		self.assertIn("Never claim to remember a person", with_rules)



class TestExamples(unittest.TestCase):

	def test_examples_get_notes_only_with_the_harness(self):

		from cloud.llm import few_shot_messages, FEW_SHOT, EXAMPLE_NOTES
		self.assertEqual(len(EXAMPLE_NOTES), len(FEW_SHOT))
		plain = few_shot_messages(["any"])
		noted = few_shot_messages(["any"], context_notes=True)
		self.assertEqual(len(plain), len(noted))
		self.assertEqual(plain[0]["content"], FEW_SHOT[0][0])
		self.assertTrue(noted[0]["content"].startswith("[You are awake"))
		self.assertTrue(noted[0]["content"].endswith("]\n" + FEW_SHOT[0][0]))
		# the answers are the same
		self.assertEqual([m for m in plain if m["role"] == "assistant"], [m for m in noted if m["role"] == "assistant"])



class TestFeatures(ConfigTestCase):

	def test_offline_has_context_and_memory(self):

		f = self.load({"HEYMIRO_SET": "ai.provider=local"}).features
		self.assertTrue(f.context)
		self.assertTrue(f.memory)

	def test_online_never_has_them(self):

		f = self.load({"HEYMIRO_SET": "ai.provider=cloud;features.context=true;features.memory=true"}).features
		self.assertFalse(f.context)
		self.assertFalse(f.memory)

	def test_memory_needs_the_harness(self):

		f = self.load({"HEYMIRO_SET": "ai.provider=local;features.context=false"}).features
		self.assertFalse(f.memory)


if __name__ == "__main__":
	unittest.main()
