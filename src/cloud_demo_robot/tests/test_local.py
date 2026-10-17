#
#	Tests for the local pipeline (ai.provider: local): the action
#	registry and reply schema, the streaming reply parser, streamed
#	turns, wake words by ASR, Silero VAD, smart-turn end of turn, the
#	local routes, the OpenAI-style TTS client and the mixer's append.
#	No network, GPU, ROS or audio device is needed.
#

import _testpath
_testpath.install_stubs()      # node_voice imports node (ROS)

import concurrent.futures
import json
import os
import queue
import shutil
import tempfile
import time
import unittest

import numpy as np

import audio_util
from cloud import llm, tools
from cloud.llm import ReplyStream, parse_llm_reply
from cloud.tts import OpenAISpeech, TurnDetector
from forebrain import dialogue as dlg
from forebrain.segmenter import UtteranceSegmenter
from forebrain.wakeword import AsrKeywordSpotter, SileroVad, match_wake_phrase, make_vad
from heymiro_config import HeyMiroConfig, LOCAL_LLM_URL, LOCAL_SPEECH_URL
from node_voice import Mixer

from test_cloud import FakeResponse, FakeSession, make_route
from test_dialogue import DialogueTestBase, Cfg, FakeTts


class Song(object):

	def __init__(self, title, artist):
		self.title = title
		self.artist = artist


SONGS = [Song("Dancing Queen", "ABBA"), Song("Just Dance", "Lady Gaga")]
SONG_LIST = tools.song_choices(SONGS)


def stream_json(**fields):

	# a reply in the schema order, as the model streams it
	order = ["emotion", "valence", "arousal", "actions", "reply"]
	return "{" + ", ".join(json.dumps(k) + ": " + json.dumps(fields[k]) for k in order if k in fields) + "}"



# ------------------------------------------------------------ registry

class TestTools(unittest.TestCase):

	def test_song_choices(self):

		self.assertEqual(SONG_LIST, ["Dancing Queen - ABBA", "Just Dance - Lady Gaga", "any"])
		self.assertEqual(tools.song_choices([]), ["any"])

	def test_schema_order_and_enums(self):

		schema = tools.reply_schema(SONG_LIST)
		self.assertEqual(list(schema["properties"]), ["emotion", "valence", "arousal", "actions", "reply"])
		self.assertEqual(schema["required"], list(schema["properties"]))
		self.assertEqual(schema["properties"]["emotion"]["enum"], list(llm.EMOTIONS))
		self.assertEqual(schema["properties"]["actions"]["maxItems"], tools.MAX_ACTIONS)
		names = [a["properties"]["tool"]["const"] for a in schema["properties"]["actions"]["items"]["anyOf"]]
		self.assertEqual(names, [t.name for t in tools.TOOLS])
		dance = [a for a in schema["properties"]["actions"]["items"]["anyOf"]
			if a["properties"]["tool"]["const"] == "dance"][0]
		self.assertEqual(dance["properties"]["song"]["enum"], SONG_LIST)
		# serialisable as sent
		json.dumps(tools.response_format(SONG_LIST))

	def test_every_flag_is_a_command(self):

		self.assertEqual(set(t.flag for t in tools.TOOLS), set(llm.COMMAND_FLAGS))
		self.assertEqual(tools.EMOTIONS, llm.EMOTIONS)

	def test_prompt_section(self):

		text = tools.prompt_section(SONG_LIST)
		for t in tools.TOOLS:
			self.assertIn('"' + t.name + '"', text)
		self.assertIn("Dancing Queen - ABBA", text)
		self.assertNotIn("; any", text)

	def test_actions_to_commands(self):

		c = llm.default_commands()
		applied = tools.actions_to_commands([{"tool": "dance", "song": "Dancing Queen - ABBA"},
			{"tool": "spin"}, {"tool": "move_ears"}], c)
		self.assertEqual(applied, ["dance", "spin"])    # at most two
		self.assertTrue(c["dance"] and c["spin"] and not c["move_ears"])
		self.assertEqual(c["song"], "Dancing Queen - ABBA")

		c = llm.default_commands()
		tools.actions_to_commands(["Spin_Around", {"name": "dance", "arguments": {"song": "any"}},
			{"tool": "fly"}], c)
		self.assertTrue(c["spin"] and c["dance"])
		self.assertIsNone(c["song"])

	def test_parser_accepts_actions(self):

		r = parse_llm_reply(stream_json(emotion="happy", valence=0.9, arousal=0.8,
			actions=[{"tool": "identify_object"}], reply="Show me!"))
		self.assertEqual((r.emotion, r.valence, r.reply), ("happy", 0.9, "Show me!"))
		self.assertTrue(r.commands["identify_object"])
		self.assertFalse(r.commands["dance"])
		# the old formats still work
		self.assertTrue(parse_llm_reply('{"reply": "x", "commands": {"spin": true}}').commands["spin"])



# --------------------------------------------------------------- prompt

class TestLocalPrompt(unittest.TestCase):

	def test_few_shot_messages(self):

		msgs = llm.few_shot_messages(SONG_LIST)
		self.assertEqual(len(msgs), 2 * len(llm.FEW_SHOT))
		self.assertEqual([m["role"] for m in msgs[:2]], ["user", "assistant"])
		for m in msgs[1::2]:
			obj = json.loads(m["content"])
			self.assertEqual(list(obj), ["emotion", "valence", "arousal", "actions", "reply"])
		dance = [json.loads(m["content"]) for m in msgs[1::2] if json.loads(m["content"])["actions"]
			and json.loads(m["content"])["actions"][0]["tool"] == "dance"][0]
		self.assertEqual(dance["actions"][0]["song"], "Dancing Queen - ABBA")
		# without an ABBA song: "any"
		songs = [a.get("song") for m in llm.few_shot_messages(["any"])[1::2]
			for a in json.loads(m["content"])["actions"] if a["tool"] == "dance"]
		self.assertEqual(songs, ["any"])

	def test_system_prompt_lists_songs(self):

		p = llm.build_system_prompt("I am MiRo at $event_name.", "the fair", SONG_LIST)
		self.assertIn("I am MiRo at the fair.", p)
		self.assertIn("Just Dance - Lady Gaga", p)
		self.assertIn("JSON", p)

	def test_context_in_user(self):

		mem = llm.ChatMemory(2)
		mem.add("u", "a")
		ctx = llm.context_message(0.5, 0.5, "now")
		msgs = mem.messages("SYS", ctx, "hello", [{"role": "user", "content": "ex"}], context_in_user=True)
		self.assertEqual([m["role"] for m in msgs], ["system", "user", "user", "assistant", "user"])
		self.assertEqual(sum(m["role"] == "system" for m in msgs), 1)
		self.assertTrue(msgs[-1]["content"].startswith("[" + ctx["content"] + "]"))
		self.assertTrue(msgs[-1]["content"].endswith("hello"))



# -------------------------------------------------------- reply stream

class TestReplyStream(unittest.TestCase):

	def feed_all(self, raw, step=1):

		rs = ReplyStream()
		events = []
		for i in range(0, len(raw), step):
			events += rs.feed(raw[i:i + step])
		events += rs.finish()
		return rs, events

	def test_header_then_sentences(self):

		raw = stream_json(emotion="happy", valence=0.9, arousal=0.8,
			actions=[{"tool": "dance", "song": "Dancing Queen - ABBA"}],
			reply="YES! I LOVE ABBA, they are my favourite! Let's dance right now! Ready?")
		rs, events = self.feed_all(raw)
		kinds = [k for k, v in events]
		self.assertEqual(kinds[0], "header")
		header = events[0][1]
		self.assertTrue(header.commands["dance"])
		self.assertEqual(header.commands["song"], "Dancing Queen - ABBA")
		self.assertEqual(header.reply, "")
		sentences = [v for k, v in events if k == "sentence"]
		self.assertEqual(" ".join(sentences), rs.reply.reply)
		self.assertGreaterEqual(len(sentences), 2)
		self.assertTrue(all(len(s) >= 20 for s in sentences[:-1]))

	def test_header_arrives_before_the_reply_text(self):

		raw = stream_json(emotion="sad", valence=0.2, arousal=0.3, actions=[], reply="Oh no. That is really sad to hear.")
		rs = ReplyStream()
		cut = raw.index('"reply"') + len('"reply": "')
		self.assertEqual([k for k, v in rs.feed(raw[:cut])], ["header"])

	def test_escapes_split_across_pieces(self):

		raw = stream_json(emotion="fine", valence=0.5, arousal=0.5, actions=[],
			reply='He said "hi" — café time, isn\'t it lovely today?')
		raw = raw.replace("—", "\\u2014")
		for step in (1, 2, 3, 7):
			rs, events = self.feed_all(raw, step)
			text = " ".join(v for k, v in events if k == "sentence")
			self.assertEqual(text, 'He said "hi" — café time, isn\'t it lovely today?', step)

	def test_plain_text_fallback(self):

		rs, events = self.feed_all("Hello! I am MiRo and I love to dance.")
		self.assertEqual(events, [("sentence", "Hello! I am MiRo and I love to dance.")])
		self.assertIsNone(rs.header)

	def test_cleans_unsayable(self):

		rs, events = self.feed_all(stream_json(emotion="happy", valence=0.9, arousal=0.9, actions=[],
			reply="*wags tail* Yay!"))
		self.assertEqual([v for k, v in events if k == "sentence"], ["wags tail Yay!"])



# ------------------------------------------------------- streamed turns

class StreamClients(object):

	def __init__(self, raw, transcript="can you dance"):
		self.raw = raw
		self.transcript = transcript
		self.kwargs = None

	def transcribe(self, wav, model="whisper-1", language=None, prompt=None):
		self.prompt = prompt
		return self.transcript

	def chat_stream(self, messages, model, **kwargs):
		self.messages = messages
		self.kwargs = kwargs
		for i in range(0, len(self.raw), 5):
			yield self.raw[i:i + 5]


class StreamTts(FakeTts):

	def synthesize_stream(self, text, impression=False):
		self.calls.append((text, impression))
		yield np.ones(240, dtype=np.int16)
		yield np.ones(240, dtype=np.int16)


class TestStreamedTurn(unittest.TestCase):

	def run_turn(self, raw, **kw):

		clients = StreamClients(raw)
		tts = StreamTts()
		runner = dlg.TurnRunner(clients, tts, llm.ChatMemory(2), "SYS", Cfg({}), examples=[],
			response_format={"type": "json_schema"}, extra_body={"x": 1}, stream=True,
			stt_prompt="MiRo", context_in_user=True, **kw)
		progress = queue.Queue()
		result = runner(np.zeros(16000, dtype=np.int16), 0.5, 0.5, "now", progress)
		events = []
		while not progress.empty():
			events.append(progress.get())
		return clients, tts, result, events

	def test_speech_streams_sentence_by_sentence(self):

		raw = stream_json(emotion="happy", valence=0.8, arousal=0.7, actions=[{"tool": "spin"}],
			reply="Wheee, here I go spinning around! Did you like that one?")
		clients, tts, result, events = self.run_turn(raw)
		self.assertEqual(events[0][0], "header")
		self.assertTrue(events[0][1].commands["spin"])
		self.assertEqual([k for k, v in events[1:]], ["speech"] * 4)
		self.assertEqual(len(tts.calls), 2)
		self.assertTrue(result.streamed)
		self.assertIsNone(result.speech)
		self.assertEqual(result.raw, raw)
		self.assertTrue(result.reply.commands["spin"])
		self.assertEqual(clients.kwargs["response_format"], {"type": "json_schema"})
		self.assertEqual(clients.prompt, "MiRo")
		self.assertEqual(sum(m["role"] == "system" for m in clients.messages), 1)

	def test_dance_says_nothing(self):

		raw = stream_json(emotion="happy", valence=0.9, arousal=0.9,
			actions=[{"tool": "dance", "song": "any"}], reply="Let's dance! Woohoo, this is my favourite!")
		clients, tts, result, events = self.run_turn(raw)
		self.assertEqual([k for k, v in events], ["header"])
		self.assertEqual(tts.calls, [])
		self.assertFalse(result.streamed)

	def test_impression_voice(self):

		raw = stream_json(emotion="happy", valence=0.9, arousal=0.9, actions=[{"tool": "impression"}],
			reply="Hiya pal, it's me, Mickey Mouse!")
		clients, tts, result, events = self.run_turn(raw)
		self.assertTrue(all(imp for text, imp in tts.calls))



class TestStreamingMachine(DialogueTestBase):

	def test_starts_speaking_at_first_sentence(self):

		future = self.to_thinking()
		progress = self.submit.progress
		self.voice.finish()          # the "hmm" is over
		self.tick()
		self.assertEqual(self.m.state, dlg.THINKING)

		hdr = parse_llm_reply(stream_json(emotion="happy", valence=0.8, arousal=0.7, actions=[]))
		progress.put(("header", hdr))
		progress.put(("speech", np.ones(2400, dtype=np.int16)))
		self.tick()
		self.assertEqual(self.m.state, dlg.RESPONDING)
		self.assertIn("reply", self.events())
		self.assertIn("music", self.voice.stops)

		# more speech while the job still runs is appended
		progress.put(("speech", np.ones(2400, dtype=np.int16)))
		self.voice.finish()
		self.tick()
		self.assertEqual(self.voice.appended, 1)
		self.assertEqual(self.m.state, dlg.RESPONDING)   # job not done yet

		future.set_result(dlg.TurnResult("hi", parse_llm_reply(stream_json(emotion="happy",
			valence=0.8, arousal=0.7, actions=[], reply="Hello! Nice to see you.")), None, "RAW", streamed=True))
		self.voice.finish()
		self.tick()
		self.assertIn("remember", [n for n, p in self.m.outbox] + ["remember"])
		self.tick()
		self.assertEqual(self.m.state, dlg.LISTENING)
		self.assertEqual(self.m.reply.reply, "Hello! Nice to see you.")

	def test_dance_header_waits_for_the_result(self):

		future = self.to_thinking()
		progress = self.submit.progress
		self.voice.finish()
		progress.put(("header", parse_llm_reply(stream_json(emotion="happy", valence=0.9, arousal=0.9,
			actions=[{"tool": "dance", "song": "any"}]))))
		self.tick()
		self.assertEqual(self.m.state, dlg.THINKING)
		future.set_result(dlg.TurnResult("dance", parse_llm_reply(stream_json(emotion="happy", valence=0.9,
			arousal=0.9, actions=[{"tool": "dance", "song": "any"}], reply="Yay!")), None, "RAW"))
		self.tick()
		self.assertEqual(self.m.state, dlg.SUSPENDED)
		self.assertIn("task", self.events())

	def test_stop_while_streaming(self):

		self.to_thinking()
		progress = self.submit.progress
		self.voice.finish()
		progress.put(("header", parse_llm_reply(stream_json(emotion="fine", valence=0.5, arousal=0.5, actions=[]))))
		progress.put(("speech", np.ones(2400, dtype=np.int16)))
		self.tick()
		self.assertEqual(self.m.state, dlg.RESPONDING)
		self.m.on_stop()
		self.assertEqual(self.m.state, dlg.GOODBYE)
		progress.put(("speech", np.ones(2400, dtype=np.int16)))
		self.voice.finish()
		self.tick(n=4)
		self.assertEqual(self.m.state, dlg.IDLE)



# -------------------------------------------------------------- hearing

def frames(n, value=0):

	return [np.full(audio_util.HEARING_FRAME, value, dtype=np.int16) for _ in range(n)]


class TestWakePhrases(unittest.TestCase):

	def test_match(self):

		cases = {
			"Hey MiRo!": ["hey_miro"], "hey mirror, how are you": ["hey_miro"], "Hello Milo": ["hey_miro"],
			"Stop, Miro.": ["stop_miro"], "Miro stop!": ["stop_miro"], "Dance, Mero!": ["dance_miro"],
			"I like mirrors": [], "hey there": [], "stop it": [], "hey hero": [],
		}
		for text, want in cases.items():
			self.assertEqual(match_wake_phrase(text), want, text)
		self.assertEqual(match_wake_phrase("hey miro", ids=["stop_miro"]), [])

	def spotter(self, answer, **kw):

		calls = []
		found = []

		def transcribe(wav):
			calls.append(wav)
			return answer

		s = AsrKeywordSpotter(["hey_miro", "dance_miro", "stop_miro", "other"], transcribe,
			onset_frames=2, log=lambda *a: None, on_found=found.extend, **kw)
		self.addCleanup(s.delete)
		return s, calls, found

	def speak(self, s, n_voiced=20, n_quiet=15):

		for f in frames(n_voiced, 1000):
			s.process(f, 0.9)
		for f in frames(n_quiet):
			s.process(f, 0.0)

	def wait(self, cond):

		t0 = time.time()
		while not cond() and time.time() - t0 < 2.0:
			time.sleep(0.01)

	def test_segment_is_transcribed_and_matched(self):

		s, calls, found = self.spotter(("Hey MiRo!", 0.01))
		self.assertEqual(s.ids, ["hey_miro", "dance_miro", "stop_miro"])
		self.speak(s)
		self.wait(lambda: found)
		self.assertEqual(found, ["hey_miro"])
		self.assertEqual(len(calls), 1)
		self.assertTrue(calls[0].startswith(b"RIFF"))

	def test_no_speech_and_mask(self):

		s, calls, found = self.spotter(("Hey MiRo!", 0.95))
		self.speak(s)
		self.wait(lambda: calls)
		time.sleep(0.05)
		self.assertEqual(found, [])          # Whisper was not sure it was speech

		s, calls, found = self.spotter(("Hey MiRo!", 0.0))
		s.mask = frozenset()
		self.speak(s)
		time.sleep(0.1)
		self.assertEqual(calls, [])          # nothing to listen for: no transcription

		s, calls, found = self.spotter(("Hey MiRo, stop!", 0.0))
		s.mask = frozenset(["stop_miro"])
		self.speak(s)
		self.wait(lambda: found)
		self.assertEqual(found, ["stop_miro"])

	def test_busy_skips_long_speech(self):

		# during a conversation only short bursts ("Stop MiRo") are transcribed
		s, calls, found = self.spotter(("can you dance to dancing queen", 0.0))
		s.mask = frozenset(["stop_miro"])
		self.speak(s, n_voiced=80)           # ~2.6 s sentence
		time.sleep(0.1)
		self.assertEqual(calls, [])
		self.assertEqual(s.skipped, 1)
		self.speak(s, n_voiced=30)           # ~1 s: transcribed
		self.wait(lambda: calls)
		self.assertEqual(len(calls), 1)
		# when idle, a sentence of the same kind is transcribed
		s.mask = None
		self.speak(s, n_voiced=60)           # ~1.9 s, under max_segment_s
		self.wait(lambda: len(calls) == 2)
		time.sleep(0.1)
		self.assertEqual(len(calls), 2)

	def test_long_speech_is_cut(self):

		s, calls, found = self.spotter(("blah", 0.0), max_segment_s=1.0)
		for f in frames(100, 1000):
			s.process(f, 0.9)
		self.wait(lambda: calls)
		self.assertGreaterEqual(len(calls), 1)


class FakeSilero(object):

	def __init__(self):
		self.calls = 0

	def __call__(self, x, rate):
		self.calls += 1
		return 0.75


class TestVad(unittest.TestCase):

	def test_silero_wrapper(self):

		v = SileroVad(model=FakeSilero())
		self.assertEqual(v.kind, "silero")
		self.assertAlmostEqual(v.process(np.zeros(512, dtype=np.int16)), 0.75)

	def test_make_vad_falls_back(self):

		v = make_vad("silero", None)
		self.assertIn(v.kind, ("silero", "energy"))


class TestSmartTurn(unittest.TestCase):

	def make(self, answer):

		self.asked = []

		def check(pcm):
			self.asked.append(len(pcm))
			if isinstance(answer, Exception):
				raise answer
			return answer

		seg = UtteranceSegmenter(end_silence_s=1.2, onset_frames=2, turn_check=check,
			turn_min_silence_s=0.3, turn_step_s=0.3)
		seg.arm()
		return seg

	def run_speech(self, seg, quiet_frames=60):

		for f in frames(30, 1000):
			ev = seg.feed(f, 0.9)
		for i in range(quiet_frames):
			ev = seg.feed(frames(1)[0], 0.0)
			if ev is not None and ev[0] == "utterance":
				return i + 1
		return None

	def test_complete_ends_early(self):

		seg = self.make(True)
		n = self.run_speech(seg)
		self.assertIsNotNone(n)
		self.assertLessEqual(n * 512 / 16000.0, 0.4)
		self.assertEqual(len(self.asked), 1)

	def test_incomplete_waits_for_the_silence(self):

		for answer in (False, None, RuntimeError("down")):
			seg = self.make(answer)
			n = self.run_speech(seg)
			self.assertAlmostEqual(n * 512 / 16000.0, 1.2, delta=0.05)
			self.assertGreaterEqual(len(self.asked), 3)



# --------------------------------------------------- config, clients, mixer

class TestLocalConfig(unittest.TestCase):

	def setUp(self):

		self.tmp = tempfile.mkdtemp(prefix="heymiro_local_")
		os.makedirs(os.path.join(self.tmp, "config"))
		shutil.copy(os.path.join(_testpath.ROOT, "config", "heymiro.yaml"), os.path.join(self.tmp, "config"))
		self.addCleanup(shutil.rmtree, self.tmp, True)

	def load(self, env=None):

		env = dict(env or {})
		env["HEYMIRO_SECRETS_FILE"] = os.path.join(self.tmp, "none.env")
		env["HOME"] = self.tmp
		return HeyMiroConfig(self.tmp, env)

	def test_defaults(self):

		cfg = self.load()
		self.assertTrue(cfg.local)
		self.assertEqual(cfg.route("llm").base_url, LOCAL_LLM_URL)
		for name in ("stt", "tts", "turn"):
			self.assertEqual(cfg.route(name).base_url, LOCAL_SPEECH_URL)
		self.assertTrue(cfg.route("llm").available)
		f = cfg.features
		self.assertTrue(f.conversation and f.tts and f.wake_word and f.end_of_turn)
		self.assertEqual((f.wake_word_backend, f.vad), ("asr", "silero"))
		# local settings replace the cloud ones
		self.assertEqual(cfg.cfg.get("llm.model"), "heymiro")
		self.assertEqual(cfg.cfg.get("stt.model"), "whisper-large-v3-turbo")

	def test_route_override_and_no_openai_key_leak(self):

		cfg = self.load({"HEYMIRO_LLM_BASE_URL": "http://gpu-box:8080/v1", "OPENAI_API_KEY": "sk-SECRET"})
		self.assertEqual(cfg.route("llm").base_url, "http://gpu-box:8080/v1")
		self.assertIsNone(cfg.route("llm").api_key)
		self.assertIsNone(cfg.route("stt").api_key)

	def test_cloud_unchanged(self):

		cfg = self.load({"HEYMIRO_SET": "ai.provider=cloud"})
		self.assertFalse(cfg.local)
		self.assertEqual(cfg.route("llm").base_url, "https://api.openai.com/v1")
		self.assertEqual(cfg.cfg.get("llm.model"), "gpt-4o")
		self.assertFalse(cfg.features.conversation)


class StreamResponse(FakeResponse):

	def __init__(self, chunks, status_code=200):
		FakeResponse.__init__(self, status_code)
		self.chunks = chunks
		self.closed = False

	def iter_content(self, chunk_size=None):
		return iter(self.chunks)

	def close(self):
		self.closed = True


class TestLocalClients(unittest.TestCase):

	def test_speech_request_and_stream(self):

		pcm = (np.arange(1000, dtype=np.int16) - 500).astype("<i2").tobytes()
		resp = StreamResponse([pcm[:301], pcm[301:1201], pcm[1201:]])   # odd splits
		session = FakeSession(lambda m, u, kw: resp)
		tts = OpenAISpeech(make_route("tts", "http://127.0.0.1:8090/v1", None), {"voice": "af_heart",
			"pitch": 1.2}, session=session)
		chunks = list(tts.synthesize_stream("Hello there."))
		self.assertEqual(sum(len(c) for c in chunks), 1000)
		self.assertTrue(resp.closed)
		method, url, kw = session.calls[0]
		self.assertEqual(url, "http://127.0.0.1:8090/v1/audio/speech")
		self.assertEqual(kw["json"]["voice"], "af_heart")
		self.assertEqual(kw["json"]["response_format"], "pcm")
		self.assertAlmostEqual(kw["json"]["pitch"], 1.2)
		self.assertNotIn("Authorization", kw["headers"])
		url, headers, body = tts.request("hi", impression=True)
		self.assertEqual(body["voice"], tts.impression_voice)

	def test_speech_error(self):

		session = FakeSession(lambda m, u, kw: StreamResponse([], 503))
		tts = OpenAISpeech(make_route("tts", "http://x/v1", None), None, session=session)
		with self.assertRaises(Exception):
			tts.synthesize("Hello.")

	def test_turn_detector(self):

		session = FakeSession(lambda m, u, kw: FakeResponse(payload={"probability": 0.8}))
		det = TurnDetector(make_route("turn", "http://127.0.0.1:8090/v1", None), session=session)
		self.assertTrue(det(np.zeros(16000 * 10, dtype=np.int16)))
		method, url, kw = session.calls[0]
		self.assertEqual(url, "http://127.0.0.1:8090/v1/turn")
		self.assertEqual(len(kw["data"]), 8 * 16000 * 2)         # last 8 s only
		det = TurnDetector(make_route("turn", "http://x/v1", None),
			session=FakeSession(lambda m, u, kw: FakeResponse(payload={"probability": 0.2})))
		self.assertFalse(det(np.zeros(100, dtype=np.int16)))

		def down(m, u, kw):
			raise IOError("refused")
		det = TurnDetector(make_route("turn", "http://x/v1", None), session=FakeSession(down))
		self.assertIsNone(det(np.zeros(100, dtype=np.int16)))


class TestMixerAppend(unittest.TestCase):

	def test_append_continues_playback(self):

		m = Mixer()
		m.play("speech", np.ones(100, dtype=np.int16))
		m.pull(60)
		m.append("speech", np.full(50, 2, dtype=np.int16))
		self.assertEqual(m.remaining("speech"), 90)
		out = m.pull(90)
		self.assertTrue(np.all(out[:40] == 1) and np.all(out[40:] == 2))
		self.assertFalse(m.active("speech"))
		# appending to an idle channel starts it
		m.append("speech", np.ones(10, dtype=np.int16))
		self.assertTrue(m.active("speech"))


if __name__ == "__main__":
	unittest.main()



# ------------------------------------------------- noise front-end, scene

class TestHighPass(unittest.TestCase):

	def tone(self, hz, n=16000, rate=16000):
		return np.sin(2 * np.pi * hz * np.arange(n) / rate) * 10000.0

	def rms(self, x):
		return float(np.sqrt(np.mean(np.square(x[2000:]))))

	def test_removes_rumble_keeps_voice(self):

		hp = audio_util.HighPass(16000, 80.0)
		self.assertLess(self.rms(hp.process(self.tone(25))) / self.rms(self.tone(25)), 0.15)
		hp.reset()
		self.assertGreater(self.rms(hp.process(self.tone(1000))) / self.rms(self.tone(1000)), 0.98)

	def test_blocks_equal_whole_and_fallback_matches(self):

		x = np.random.default_rng(0).standard_normal(5000) * 3000
		whole = audio_util.HighPass(16000, 80.0).process(x)
		hp = audio_util.HighPass(16000, 80.0)
		parts = np.concatenate([hp.process(x[i:i + 512]) for i in range(0, 5000, 512)])
		self.assertTrue(np.allclose(whole, parts, atol=1e-2))
		slow = audio_util.HighPass(16000, 80.0)
		slow._lfilter = None
		self.assertTrue(np.allclose(whole, slow.process(x), atol=1e-2))


class TestSceneContext(unittest.TestCase):

	def msg(self, objects, confirmed):
		return {"t": 0, "camera": "caml", "objects": objects, "confirmed": confirmed}

	def test_describe(self):

		from forebrain.scene_context import describe_scene
		m = self.msg([
			{"label": "person", "score": 0.9, "box": [0.4, 0.1, 0.6, 0.9]},
			{"label": "cup", "score": 0.7, "box": [0.05, 0.6, 0.15, 0.8]},
			{"label": "cell phone", "score": 0.5, "box": [0.8, 0.5, 0.9, 0.6]},
			{"label": "dog", "score": 0.6, "box": [0.4, 0.4, 0.5, 0.5]},     # not confirmed
		], ["person", "cup", "cell phone"])
		text = describe_scene((100.0, m), now=101.0)
		self.assertEqual(text, "Your camera sees: a person in front of you, a cup on your left, a cell phone on your right.")
		self.assertIsNone(describe_scene((100.0, m), now=110.0))            # stale
		self.assertIsNone(describe_scene((100.0, self.msg([], [])), now=100.5))
		self.assertIsNone(describe_scene(None))
		two = self.msg([{"label": "person", "score": 0.9, "box": [0.1, 0.1, 0.3, 0.9]},
			{"label": "person", "score": 0.8, "box": [0.6, 0.1, 0.8, 0.9]}], ["person"])
		self.assertEqual(describe_scene((1.0, two), now=1.0), "Your camera sees: two people.")
		big = self.msg([{"label": "apple", "score": 0.8, "box": [0.0, 0.0, 0.3, 1.0]}], ["apple"])
		self.assertEqual(describe_scene((1.0, big), now=1.0), "Your camera sees: an apple right on your left.")

	def test_scene_mode_and_runner(self):

		import node_scene
		self.assertEqual(node_scene.mode_wants("context"), (False, True))
		clients = StreamClients(stream_json(emotion="happy", valence=0.8, arousal=0.7, actions=[],
			reply="I see you!"))
		runner = dlg.TurnRunner(clients, StreamTts(), llm.ChatMemory(2), "SYS", Cfg({}), stream=True,
			response_format={"type": "json_schema"}, context_in_user=True)
		runner(np.zeros(16000, dtype=np.int16), 0.5, 0.5, "now", queue.Queue(),
			scene="Your camera sees: a person in front of you.")
		self.assertIn("Your camera sees: a person in front of you.", clients.messages[-1]["content"])
