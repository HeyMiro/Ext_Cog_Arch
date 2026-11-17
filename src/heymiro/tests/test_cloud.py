#
#	Tests for core/cloud: worker, reply parser, prompt and memory,
#	OpenAI wrapper (fake client), ElevenLabs TTS (fake HTTP session),
#	song library, mp3 decoding and the Spotify lookup (fake session).
#	No network, audio device or API key is needed.
#

import _testpath

import contextlib
import io
import json
import os
import shutil
import subprocess
import time
import types
import unittest

import numpy as np

import heymiro_config
from cloud import llm, music, tts, worker
from cloud.llm import parse_llm_reply


HAVE_FFMPEG = shutil.which("ffmpeg") is not None
SECRET = "sk-test-SECRET-value-123"



# ------------------------------------------------------------------ fakes

class FakeResponse(object):

	def __init__(self, status_code=200, content=b"", headers=None, payload=None):

		self.status_code = status_code
		self.content = content
		self.headers = headers or {}
		self.payload = payload

	def json(self):

		return self.payload if self.payload is not None else json.loads(self.content.decode("utf-8"))


class FakeSession(object):

	# handler(method, url, kwargs) -> FakeResponse; every call recorded
	def __init__(self, handler):

		self.handler = handler
		self.calls = []

	def post(self, url, **kwargs):

		self.calls.append(("POST", url, kwargs))
		return self.handler("POST", url, kwargs)

	def get(self, url, **kwargs):

		self.calls.append(("GET", url, kwargs))
		return self.handler("GET", url, kwargs)


class StatusError(Exception):

	# looks like openai.APIStatusError: has .status_code, and a message
	# that (like OpenAI's 401) mentions the key
	def __init__(self, status_code):

		Exception.__init__(self, "Incorrect API key provided: " + SECRET)
		self.status_code = status_code


class FakeCompletions(object):

	def __init__(self, fail_json_with=None):

		self.calls = []
		self.fail_json_with = fail_json_with

	def create(self, **kwargs):

		self.calls.append(kwargs)
		if "response_format" in kwargs and self.fail_json_with:
			raise StatusError(self.fail_json_with)
		msg = types.SimpleNamespace(content='{"reply": "hi"}')
		return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


class FakeTranscriptions(object):

	def __init__(self):

		self.calls = []

	def create(self, **kwargs):

		self.calls.append(kwargs)
		return types.SimpleNamespace(text="  hello miro  ")


class FakeOpenAI(object):

	def __init__(self, route, fail_json_with=None):

		self.route = route
		self.chat = types.SimpleNamespace(completions=FakeCompletions(fail_json_with))
		self.audio = types.SimpleNamespace(transcriptions=FakeTranscriptions())


def make_route(name="llm", base="http://localhost:9/v1", key=SECRET, timeout=5.0):

	return heymiro_config.Route(name, base, key, timeout, False)


def sine(n=2400, rate=24000, amp=1000.0):

	t = np.arange(n) / float(rate)
	return (amp * np.sin(2 * np.pi * 440.0 * t)).astype(np.int16)



# ----------------------------------------------------------------- worker

class TestWorker(unittest.TestCase):

	def test_submit_result_and_exception(self):

		w = worker.CloudWorker(2)
		try:
			ok = w.submit(lambda a, b=0: a + b, 2, b=3)
			self.assertEqual(ok.result(timeout=5), 5)

			def boom():
				raise worker.HttpError("tts", 401)
			bad = w.submit(boom)
			exc = bad.exception(timeout=5)
			self.assertIsInstance(exc, worker.HttpError)
			self.assertEqual(worker.describe_error(exc), "HttpError 401")
		finally:
			w.shutdown()

	def test_submit_after_shutdown_gives_failed_future(self):

		w = worker.CloudWorker(1)
		w.shutdown()
		f = w.submit(lambda: 1)
		self.assertTrue(f.done())
		self.assertIsInstance(f.exception(), RuntimeError)
		w.shutdown()	# twice is fine

	def test_describe_error_is_sanitized(self):

		text = worker.describe_error(StatusError(401))
		self.assertEqual(text, "StatusError 401")
		self.assertNotIn(SECRET, text)

		class RequestsLike(Exception):
			pass
		e = RequestsLike("https://x/?key=" + SECRET)
		e.response = types.SimpleNamespace(status_code=503)
		self.assertEqual(worker.describe_error(e), "RequestsLike 503")
		self.assertEqual(worker.describe_error(ValueError(SECRET)), "ValueError")
		self.assertEqual(worker.describe_error(None), "no error")



# ----------------------------------------------------------------- parser

HRI_LEGACY = """Valence: 0.9
Arousal: 0.8
Emotion: Happy
Spin_around: False
Go_sleep: False
Move_ears: True
Do_impression: False
Dance: True
Song: Dancing Queen by ABBA
Gesture_mode: False
Obj_Detect_mode: False
WOW, I LOVE ABBA!!
Let's DANCE together, shall we?"""


class TestParser(unittest.TestCase):

	def test_valid_json(self):

		text = json.dumps({
			"valence": 0.8, "arousal": 0.7, "emotion": "Happy", "reply": "Hello there!",
			"commands": {"spin": True, "sleep": False, "move_ears": False, "impression": False,
				"dance": False, "song": None, "tricks": False, "identify_object": True}})
		r = parse_llm_reply(text)
		self.assertAlmostEqual(r.valence, 0.8)
		self.assertAlmostEqual(r.arousal, 0.7)
		self.assertEqual(r.emotion, "happy")
		self.assertEqual(r.reply, "Hello there!")
		self.assertTrue(r.commands["spin"])
		self.assertTrue(r.commands["identify_object"])
		self.assertFalse(r.commands["dance"])
		self.assertIsNone(r.commands["song"])
		self.assertEqual(r.raw, text)
		self.assertEqual(set(r.commands), set(llm.COMMAND_FLAGS) | set(["song"]))

	def test_fenced_json(self):

		text = 'Sure!\n```json\n{"valence": 0.2, "arousal": 0.3, "emotion": "sad", "reply": "Oh no."}\n```\n'
		r = parse_llm_reply(text)
		self.assertEqual(r.emotion, "sad")
		self.assertEqual(r.reply, "Oh no.")
		self.assertAlmostEqual(r.valence, 0.2)

	def test_json_with_trailing_text_and_braces_in_strings(self):

		text = ('Here you go: {"valence": 0.5, "arousal": 0.5, "reply": "I like {curly} braces!", '
			'"commands": {"dance": "yes", "song": "Abba"}} hope that helps {not json}')
		r = parse_llm_reply(text)
		self.assertEqual(r.reply, "I like {curly} braces!")
		self.assertTrue(r.commands["dance"])
		self.assertEqual(r.commands["song"], "Abba")
		self.assertEqual(r.emotion, "fine")	# inferred

	def test_python_style_dict(self):

		r = parse_llm_reply("{'valence': 0.9, 'arousal': 0.9, 'reply': 'Yay!', 'commands': {'spin': True}}")
		self.assertEqual(r.reply, "Yay!")
		self.assertTrue(r.commands["spin"])
		self.assertEqual(r.emotion, "happy")

	def test_hri_legacy_format(self):

		r = parse_llm_reply(HRI_LEGACY)
		self.assertAlmostEqual(r.valence, 0.9)
		self.assertAlmostEqual(r.arousal, 0.8)
		self.assertEqual(r.emotion, "happy")
		self.assertTrue(r.commands["move_ears"])
		self.assertTrue(r.commands["dance"])
		self.assertFalse(r.commands["spin"])
		self.assertFalse(r.commands["tricks"])
		self.assertFalse(r.commands["identify_object"])
		self.assertEqual(r.commands["song"], "Dancing Queen by ABBA")
		self.assertEqual(r.reply, "WOW, I LOVE ABBA!! Let's DANCE together, shall we?")

	def test_hri_legacy_song_not_specified_and_missing_lines(self):

		text = "Valence: 0.1\nArousal: 0.9\nDance: True\nSong: False\nGRRR, no!"
		r = parse_llm_reply(text)
		self.assertIsNone(r.commands["song"])
		self.assertTrue(r.commands["dance"])
		self.assertEqual(r.emotion, "angry")	# inferred: low valence, high arousal
		self.assertEqual(r.reply, "GRRR, no!")
		for song in ("Not specified", "none", "Not specified.", "null", ""):
			r = parse_llm_reply(json.dumps({"reply": "x", "commands": {"song": song}}))
			self.assertIsNone(r.commands["song"], song)

	def test_garbage_and_plain_text(self):

		for text in (None, "", "   ", "}{", "{{{", "[1, 2, 3]", "null"):
			r = parse_llm_reply(text)
			self.assertIsInstance(r, llm.LlmReply)
			self.assertIsNone(r.valence)
			self.assertIsNone(r.emotion)
			self.assertFalse(any(r.commands[k] for k in llm.COMMAND_FLAGS))
		r = parse_llm_reply("Hello! *wags tail* I am MiRo \U0001F600❤️")
		self.assertEqual(r.reply, "Hello! wags tail I am MiRo")
		self.assertIsNone(r.emotion)
		# a single "key: value"-ish line is just speech
		r = parse_llm_reply("Fun fact: I was made in 2017!")
		self.assertEqual(r.reply, "Fun fact: I was made in 2017!")

	def test_json_without_known_keys_is_plain_text(self):

		r = parse_llm_reply('{"foo": 1}')
		self.assertEqual(r.reply, '{"foo": 1}')

	def test_clipping_and_coercion(self):

		r = parse_llm_reply(json.dumps({
			"valence": 1.7, "arousal": "-0.4", "reply": "*Hi*",
			"commands": {"spin": "True", "sleep": "yes", "move_ears": 1, "impression": "False",
				"dance": "no", "tricks": "Yes.", "identify_object": None}}))
		self.assertEqual(r.valence, 1.0)
		self.assertEqual(r.arousal, 0.0)
		self.assertEqual(r.reply, "Hi")
		self.assertTrue(r.commands["spin"])
		self.assertTrue(r.commands["sleep"])
		self.assertTrue(r.commands["move_ears"])
		self.assertFalse(r.commands["impression"])
		self.assertFalse(r.commands["dance"])
		self.assertTrue(r.commands["tricks"])
		self.assertFalse(r.commands["identify_object"])
		r = parse_llm_reply(json.dumps({"valence": "0.65 (quite good)", "arousal": "NaN", "reply": "ok"}))
		self.assertAlmostEqual(r.valence, 0.65)
		self.assertIsNone(r.arousal)
		r = parse_llm_reply(json.dumps({"valence": "high", "reply": "ok"}))
		self.assertIsNone(r.valence)
		self.assertIsNone(r.emotion)

	def test_flat_and_alias_keys(self):

		r = parse_llm_reply(json.dumps({"Valence": 0.5, "Arousal": 0.5, "Response": "Hi",
			"Spin_around": "True", "Gesture_mode": "True", "Obj_Detect_mode": "False", "Do_impression": "True"}))
		self.assertEqual(r.reply, "Hi")
		self.assertTrue(r.commands["spin"])
		self.assertTrue(r.commands["tricks"])
		self.assertTrue(r.commands["impression"])
		self.assertFalse(r.commands["identify_object"])

	def test_emotion_names_and_inference(self):

		self.assertEqual(parse_llm_reply('{"emotion": "Scared", "reply": "eek"}').emotion, "scared")
		self.assertEqual(parse_llm_reply('{"emotion": "neutral", "reply": "ok"}').emotion, "fine")
		# an unknown word falls back to inference from the values
		r = parse_llm_reply('{"emotion": "bamboozled", "valence": 0.1, "arousal": 0.3, "reply": "oh"}')
		self.assertEqual(r.emotion, "sad")
		cases = [
			((0.9, 0.9), "happy"), ((0.5, 0.5), "fine"), ((0.1, 0.9), "angry"),
			((0.3, 0.65), "worried"), ((0.1, 0.25), "sad"), ((0.9, 0.2), "fine"),
			((None, None), None), ((0.9, None), "happy"), ((None, 0.3), "fine"),
		]
		for (v, a), want in cases:
			self.assertEqual(llm.infer_emotion(v, a), want, (v, a))



# -------------------------------------------------------- prompt / memory

class TestPrompt(unittest.TestCase):

	def test_build_system_prompt_with_real_persona(self):

		with open(os.path.join(_testpath.ROOT, "config", "persona.md")) as f:
			persona = f.read()
		p = llm.build_system_prompt(persona, "the Festival of Mind")
		self.assertIn("the Festival of Mind", p)
		self.assertNotIn("$event_name", p)
		self.assertIn("JSON", p)
		self.assertIn('"identify_object"', p)

	def test_template_is_safe(self):

		p = llm.build_system_prompt("I cost $5 at $event_name ${unknown}", None)
		self.assertIn("I cost $5 at an event ${unknown}", p)

	def test_context_message(self):

		m = llm.context_message(0.25, 0.75, "Tuesday 14:05")
		self.assertEqual(m["role"], "system")
		self.assertIn("0.25", m["content"])
		self.assertIn("0.75", m["content"])
		self.assertIn("Tuesday 14:05", m["content"])
		m = llm.context_message(None, None, None, extra="The person is waving.")
		self.assertNotIn("valence", m["content"])
		self.assertIn("waving", m["content"])

	def test_object_comment_prompt(self):

		msgs = llm.object_comment_prompt("cell_phone")
		self.assertEqual([m["role"] for m in msgs], ["system", "user"])
		self.assertIn("one sentence", msgs[0]["content"])
		self.assertIn("cell phone", msgs[1]["content"])

	def test_chat_memory_trims_and_orders(self):

		mem = llm.ChatMemory(2)
		for i in range(4):
			mem.add("u%d" % i, "a%d" % i)
		self.assertEqual(len(mem), 2)
		ctx = llm.context_message(0.5, 0.5, "now")
		msgs = mem.messages("SYS", ctx, "hello")
		self.assertEqual([m["content"] for m in msgs], ["SYS", "u2", "a2", "u3", "a3", ctx["content"], "hello"])
		self.assertEqual([m["role"] for m in msgs],
			["system", "user", "assistant", "user", "assistant", "system", "user"])
		# context is not stored
		self.assertEqual(len(mem.messages("SYS", None, "x")), 6)
		mem.clear()
		self.assertEqual(mem.messages(None, None, "x"), [{"role": "user", "content": "x"}])
		self.assertEqual(len(llm.ChatMemory(0).messages("S", None, "x")), 2)
		mem0 = llm.ChatMemory(0)
		mem0.add("u", "a")
		self.assertEqual(len(mem0), 0)



# ----------------------------------------------------------------- OpenAI

class TestOpenAIClients(unittest.TestCase):

	def test_chat_retries_without_json_mode_on_400(self):

		made = []

		def factory(route):
			c = FakeOpenAI(route, fail_json_with=400)
			made.append(c)
			return c

		clients = llm.OpenAIClients(make_route("stt"), make_route("llm"), factory=factory)
		with contextlib.redirect_stdout(io.StringIO()) as out:
			text = clients.chat([{"role": "user", "content": "hi"}], "gpt-4o", 0.8, 400, json_mode=True)
		self.assertEqual(text, '{"reply": "hi"}')
		calls = made[0].chat.completions.calls
		self.assertEqual(len(calls), 2)
		self.assertEqual(calls[0]["response_format"], {"type": "json_object"})
		self.assertNotIn("response_format", calls[1])
		self.assertEqual(calls[1]["max_tokens"], 400)
		self.assertEqual(calls[1]["temperature"], 0.8)
		self.assertNotIn(SECRET, out.getvalue())
		# the client is created once and reused
		clients.chat([], "m")
		self.assertEqual(len(made), 1)

	def test_chat_other_errors_propagate(self):

		clients = llm.OpenAIClients(None, make_route(), factory=lambda r: FakeOpenAI(r, fail_json_with=401))
		with self.assertRaises(StatusError):
			clients.chat([], "gpt-4o", json_mode=True)

	def test_chat_without_json_mode(self):

		fake = FakeOpenAI(None)
		clients = llm.OpenAIClients(None, make_route(), factory=lambda r: fake)
		clients.chat([], "m", json_mode=False)
		self.assertNotIn("response_format", fake.chat.completions.calls[0])
		self.assertNotIn("temperature", fake.chat.completions.calls[0])

	def test_transcribe(self):

		fake = FakeOpenAI(None)
		clients = llm.OpenAIClients(make_route("stt"), None, factory=lambda r: fake)
		self.assertEqual(clients.transcribe(b"RIFF....", "whisper-1", "en"), "hello miro")
		call = fake.audio.transcriptions.calls[0]
		self.assertEqual(call["model"], "whisper-1")
		self.assertEqual(call["language"], "en")
		self.assertEqual(call["file"][0], "speech.wav")
		self.assertEqual(call["file"][1], b"RIFF....")
		with self.assertRaises(RuntimeError):
			clients.chat([], "m")	# no llm route

	def test_real_openai_client_gets_explicit_base_url(self):

		try:
			import openai	# noqa: F401
		except ImportError:
			self.skipTest("openai not installed")
		clients = llm.OpenAIClients(None, make_route(base="http://localhost:9/v1", key=None))
		c = clients.client("llm")
		self.assertIn("localhost:9", str(c.base_url))
		self.assertEqual(c.max_retries, 1)



# -------------------------------------------------------------------- TTS

class TestTextToSpeech(unittest.TestCase):

	def cfg(self):

		return {"model_id": "eleven_multilingual_v2", "voice_id": "VOICE1", "stability": 0.5,
			"similarity_boost": 0.75, "timeout_s": 7}

	def test_pcm_request_and_normalisation(self):

		pcm = sine(amp=1000.0)
		session = FakeSession(lambda m, u, k: FakeResponse(200, pcm.tobytes() + b"\x01",
			{"Content-Type": "audio/pcm"}))
		route = make_route("tts", base="https://tts.example", key=SECRET)
		t = tts.TextToSpeech(route, self.cfg(), session=session)
		out = t.synthesize("Hello there!", voice_id="IMPRESSION")
		method, url, kw = session.calls[0]
		self.assertEqual(method, "POST")
		self.assertEqual(url, "https://tts.example/v1/text-to-speech/IMPRESSION")
		self.assertEqual(kw["params"], {"output_format": "pcm_24000"})
		self.assertEqual(kw["headers"]["xi-api-key"], SECRET)
		self.assertEqual(kw["json"]["text"], "Hello there!")
		self.assertEqual(kw["json"]["model_id"], "eleven_multilingual_v2")
		self.assertEqual(kw["json"]["voice_settings"], {"stability": 0.5, "similarity_boost": 0.75})
		self.assertEqual(kw["timeout"], (5.0, 7.0))
		self.assertEqual(out.dtype, np.int16)
		self.assertEqual(len(out), len(pcm))
		self.assertAlmostEqual(np.max(np.abs(out)) / 32767.0, 0.9, places=2)
		# default voice
		t.synthesize("again")
		self.assertTrue(session.calls[1][1].endswith("/VOICE1"))

	def test_silence_empty_text_and_errors(self):

		session = FakeSession(lambda m, u, k: FakeResponse(200, b"\x00\x00" * 100, {"Content-Type": "audio/pcm"}))
		t = tts.TextToSpeech(make_route("tts"), self.cfg(), session=session)
		out = t.synthesize("shh")
		self.assertTrue(np.all(out == 0))
		self.assertEqual(len(t.synthesize("   ")), 0)
		self.assertEqual(len(session.calls), 1)

		t.session = FakeSession(lambda m, u, k: FakeResponse(401, b'{"detail": "bad key ' + SECRET.encode() + b'"}'))
		with self.assertRaises(worker.HttpError) as ctx:
			t.synthesize("hi")
		self.assertEqual(ctx.exception.status_code, 401)
		self.assertNotIn(SECRET, str(ctx.exception))
		self.assertNotIn(SECRET, repr(t.route))

	def test_section_config(self):

		section = heymiro_config.Section({"voice_id": "V", "model_id": "M"})
		t = tts.TextToSpeech(make_route("tts"), section, session=FakeSession(None))
		self.assertEqual(t.model_id, "M")
		self.assertEqual(t.stability, 0.5)

	@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg not installed")
	def test_mp3_response_is_decoded(self):

		mp3 = subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=0.5",
			"-ar", "44100", "-f", "mp3", "pipe:1"], stdout=subprocess.PIPE, check=True).stdout
		session = FakeSession(lambda m, u, k: FakeResponse(200, mp3, {"Content-Type": "audio/mpeg"}))
		t = tts.TextToSpeech(make_route("tts"), self.cfg(), session=session)
		out = t.synthesize("hi")
		self.assertEqual(out.dtype, np.int16)
		self.assertGreater(len(out), int(0.4 * 24000))
		self.assertLess(len(out), int(0.7 * 24000))
		self.assertAlmostEqual(np.max(np.abs(out)) / 32767.0, 0.9, places=2)
		# also recognised without a content type
		self.assertTrue(tts.is_mp3(None, mp3))
		self.assertFalse(tts.is_mp3("audio/pcm", b"\x00\x01\x02"))



# ------------------------------------------------------------------ music

class TestSongs(unittest.TestCase):

	@classmethod
	def setUpClass(cls):

		cls.cfg = heymiro_config.load(_testpath.ROOT, environ={})
		cls.lib = music.SongLibrary(cls.cfg)

	def test_manifest_loaded(self):

		self.assertGreaterEqual(len(self.lib.songs), 8)
		for s in self.lib.songs:
			self.assertTrue(os.path.isfile(s.path), s.path)
			self.assertGreater(s.bpm, 40.0)
		self.assertIsNotNone(self.lib.get("dancing_queen"))
		self.assertIn(self.lib.random(), self.lib.songs)

	def test_find(self):

		cases = {
			"dancing queen": "dancing_queen",
			"Dancing Queen by ABBA": "dancing_queen",
			"abba": "dancing_queen",
			"ABBA!": "dancing_queen",
			"whitney": "whitney_dance",
			"Whitney Houston": "whitney_dance",
			"I wanna dance with somebody": "whitney_dance",
			"murder on the dance floor": "murder_dancefloor",
			"Murder on the Dancefloor": "murder_dancefloor",
			"dancing in the moonlight": "in_the_moonlight",
			"lady gaga": "just_dance",
			"just dance": "just_dance",
			"everybody dance now": "make_you_sweat",
			"higher and higher": "higher_and_higher",
			"dua lipa": "dance_the_night",
			"dancing queeen": "dancing_queen",	# typo
		}
		for query, want in cases.items():
			song = self.lib.find(query)
			self.assertIsNotNone(song, query)
			self.assertEqual(song.id, want, query)

	def test_find_rejects_unknown(self):

		for query in ("bohemian rhapsody", "", None, "the", "xyzzy"):
			self.assertIsNone(self.lib.find(query), query)

	@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg not installed")
	def test_decode_mp3(self):

		pcm = music.decode_mp3(self.lib.get("dancing_queen").path, rate=24000, max_s=1)
		self.assertEqual(pcm.dtype, np.int16)
		self.assertEqual(pcm.ndim, 1)
		self.assertLessEqual(abs(len(pcm) - 24000), 1200)
		self.assertGreater(np.max(np.abs(pcm)), 100)
		with self.assertRaises(RuntimeError):
			music.decode_mp3("/nonexistent/song.mp3", max_s=1)

	def test_genre_from_spotify(self):

		self.assertEqual(music.genre_from_spotify(["classic soul", "soul", "vocal jazz"]), "soul")
		self.assertEqual(music.genre_from_spotify(["vocal jazz", "dance pop"]), "pop")
		self.assertEqual(music.genre_from_spotify(["swedish pop", "europop"]), "pop")
		self.assertEqual(music.genre_from_spotify(["nu disco"]), "disco")
		self.assertEqual(music.genre_from_spotify(["electro house"]), "dance")
		self.assertEqual(music.genre_from_spotify(["pop rock"]), "rock")
		self.assertEqual(music.genre_from_spotify([]), "default")
		self.assertEqual(music.genre_from_spotify(None), "default")



class TestSpotify(unittest.TestCase):

	def route(self):

		r = heymiro_config.Route("spotify", "https://api.example/v1", None, 10.0, True)
		r.token_url = "https://accounts.example/api/token"
		r.client_id = "client-id"
		r.client_secret = SECRET
		return r

	def handler(self, features_status=403, token_status=200):

		def handle(method, url, kw):
			if url.endswith("/api/token"):
				return FakeResponse(token_status, payload={"access_token": "TOKEN", "expires_in": 3600})
			if url.endswith("/search"):
				return FakeResponse(200, payload={"tracks": {"items": [{"id": "T1", "name": "Dancing Queen",
					"artists": [{"id": "A1", "name": "ABBA"}]}]}})
			if url.endswith("/artists/A1"):
				return FakeResponse(200, payload={"genres": ["europop", "swedish pop"]})
			if "/audio-features/" in url:
				return FakeResponse(features_status, payload={"tempo": 100.4})
			return FakeResponse(404, payload={})
		return handle

	def test_lookup_with_blocked_audio_features(self):

		session = FakeSession(self.handler(403))
		sp = music.SpotifyClient(self.route(), session=session)
		self.assertTrue(sp.available)
		with contextlib.redirect_stdout(io.StringIO()) as out:
			info = sp.lookup("dancing queen")
			self.assertEqual(info, {"title": "Dancing Queen", "artist": "ABBA", "genre": "pop", "bpm": None})
			self.assertTrue(sp.features_blocked)
			n = len(session.calls)
			sp.lookup("Dancing Queen!")	# cached (same normalised query)
			self.assertEqual(len(session.calls), n)
			sp.lookup("abba")
		urls = [c[1] for c in session.calls]
		self.assertEqual(sum(1 for u in urls if "/audio-features/" in u), 1)
		self.assertEqual(sum(1 for u in urls if u.endswith("/api/token")), 1)
		self.assertNotIn(SECRET, out.getvalue())
		# client credentials go as basic auth to the token URL only
		token_call = session.calls[0]
		self.assertEqual(token_call[2]["auth"], ("client-id", SECRET))
		self.assertEqual(token_call[2]["data"], {"grant_type": "client_credentials"})
		for method, url, kw in session.calls[1:]:
			self.assertEqual(kw["headers"], {"Authorization": "Bearer TOKEN"})
			self.assertIn("timeout", kw)

	def test_lookup_with_tempo(self):

		sp = music.SpotifyClient(self.route(), session=FakeSession(self.handler(200)))
		self.assertEqual(sp.lookup("dancing queen")["bpm"], 100.4)

	def test_lookup_failure_returns_none(self):

		sp = music.SpotifyClient(self.route(), session=FakeSession(self.handler(token_status=400)))
		with contextlib.redirect_stdout(io.StringIO()) as out:
			self.assertIsNone(sp.lookup("dancing queen"))
		self.assertIn("HttpError 400", out.getvalue())
		self.assertNotIn(SECRET, out.getvalue())

	def test_unavailable_without_credentials(self):

		r = self.route()
		r.client_secret = None
		sp = music.SpotifyClient(r, session=FakeSession(None))
		self.assertFalse(sp.available)
		self.assertIsNone(sp.lookup("abba"))

	def test_expired_token_is_renewed_once(self):

		state = {"n": 0}
		base = self.handler(200)

		def handle(method, url, kw):
			if url.endswith("/search"):
				state["n"] += 1
				if state["n"] == 1:
					return FakeResponse(401, payload={})
			return base(method, url, kw)

		session = FakeSession(handle)
		sp = music.SpotifyClient(self.route(), session=session)
		self.assertEqual(sp.lookup("abba")["title"], "Dancing Queen")
		self.assertEqual(sum(1 for c in session.calls if c[1].endswith("/api/token")), 2)



if __name__ == "__main__":
	unittest.main()
