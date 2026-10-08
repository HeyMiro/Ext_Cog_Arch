#
#	Tests for the conversation state machine (core/forebrain/dialogue.py)
#	and the cloud turn runner, with fake clock, hearing, voice and worker.
#

import _testpath

import concurrent.futures
import json
import unittest

import numpy as np

from forebrain import dialogue as dlg
from cloud.llm import LlmReply, ChatMemory



class FakeClock(object):

	def __init__(self):
		self.t = 100.0

	def __call__(self):
		return self.t

	def advance(self, dt):
		self.t += dt



class FakeHearing(object):

	def __init__(self):
		self.arms = 0
		self.cancels = 0

	def arm_utterance(self):
		self.arms += 1

	def cancel_utterance(self):
		self.cancels += 1



class FakeVoice(object):

	# speech starts "playing" when asked; tests end it with finish()
	def __init__(self):
		self.clips = []
		self.said = []
		self.stops = []
		self.speaking = False
		self.music = False

	def play_clip(self, category, channel="speech"):
		self.clips.append((category, channel))
		if channel == "speech":
			self.speaking = True
		else:
			self.music = True
		return 1.0

	def say_pcm(self, pcm):
		self.said.append(pcm)
		self.speaking = True
		return len(pcm) / 24000.0

	def stop(self, channel=None):
		self.stops.append(channel)
		if channel in (None, "speech"):
			self.speaking = False
		if channel in (None, "music"):
			self.music = False

	def is_speaking(self):
		return self.speaking

	def finish(self):
		self.speaking = False

	def speech_clips(self):
		return [c for c, ch in self.clips if ch == "speech"]



class FakeSubmit(object):

	# hands out Futures that the test completes
	def __init__(self):
		self.calls = []

	def __call__(self, pcm):
		future = concurrent.futures.Future()
		self.calls.append((pcm, future))
		return future

	def last(self):
		return self.calls[-1][1]



def reply(text="Hello there!", emotion="happy", **commands):

	return LlmReply(valence=0.8, arousal=0.7, emotion=emotion, reply=text, commands=commands,
		raw=json.dumps({"reply": text}))


def speech(n=2400):

	return np.ones(n, dtype=np.int16)



class DialogueTestBase(unittest.TestCase):

	face_user = "none"

	def setUp(self):

		self.clock = FakeClock()
		self.hearing = FakeHearing()
		self.voice = FakeVoice()
		self.submit = FakeSubmit()
		self.logs = []
		self.m = dlg.DialogueMachine(self.clock, self.hearing, self.voice, self.submit,
			face_user=self.face_user, face_timeout_s=8.0, thinking_music=True,
			turn_timeout_s=60.0, listen_timeout_s=40.0, greet_min_s=1.0,
			log=self.logs.append)

	def tick(self, dt=0.25, n=1):

		for _ in range(n):
			self.clock.advance(dt)
			self.m.tick()

	def events(self):

		return [name for name, payload in self.m.take_outbox()]

	def to_listening(self):

		# wake -> greeting clip -> listening
		self.assertTrue(self.m.on_wake())
		self.assertEqual(self.m.state, dlg.GREETING)
		self.tick()
		self.assertEqual(self.voice.speech_clips()[-1], "greeting")
		self.voice.finish()
		self.tick(dt=1.1)
		self.assertEqual(self.m.state, dlg.LISTENING)

	def to_thinking(self):

		self.to_listening()
		self.m.on_hearing("utterance", np.zeros(16000, dtype=np.int16))
		self.assertEqual(self.m.state, dlg.THINKING)
		return self.submit.last()



class TestNormalTurn(DialogueTestBase):

	def test_full_turn(self):

		self.to_listening()
		self.assertEqual(self.hearing.arms, 1)
		self.assertEqual(self.events(), ["started"])

		# the user starts speaking: nod
		self.m.on_hearing("voiced")
		self.assertEqual(self.events(), ["nod"])

		# utterance: "hmm", one job submitted
		self.m.on_hearing("utterance", np.zeros(16000, dtype=np.int16))
		self.assertEqual(self.m.state, dlg.THINKING)
		self.assertEqual(self.voice.speech_clips()[-1], "thinking")
		self.assertEqual(len(self.submit.calls), 1)
		future = self.submit.last()

		# loading tune only after the "hmm"
		self.tick()
		self.assertNotIn(("loading", "music"), self.voice.clips)
		self.voice.finish()
		self.tick()
		self.assertIn(("loading", "music"), self.voice.clips)

		# the cloud answers
		pcm = speech()
		future.set_result(dlg.TurnResult("hello miro", reply(), pcm, raw='{"reply": "Hello there!"}'))
		self.tick()
		self.assertEqual(self.m.state, dlg.RESPONDING)
		self.assertIs(self.voice.said[-1], pcm)
		self.assertIn("music", self.voice.stops)
		self.assertFalse(self.voice.music)
		out = self.m.take_outbox()
		names = [n for n, p in out]
		self.assertEqual(names, ["remember", "reply"])
		self.assertEqual(out[0][1][0], "hello miro")
		self.assertEqual(self.m.reply.reply, "Hello there!")

		# still talking: stay
		self.tick(n=5)
		self.assertEqual(self.m.state, dlg.RESPONDING)

		# finished: listen again
		self.voice.finish()
		self.tick()
		self.assertEqual(self.m.state, dlg.LISTENING)
		self.assertEqual(self.hearing.arms, 2)

	def test_wake_ignored_when_busy(self):

		self.to_listening()
		self.assertFalse(self.m.on_wake())
		self.assertEqual(self.m.state, dlg.LISTENING)

	def test_greeting_waits_for_quiet(self):

		# MiRo is still saying something (e.g. "ouch"): do not talk over it
		self.voice.speaking = True
		self.m.on_wake()
		self.tick()
		self.assertEqual(self.voice.clips, [])
		self.voice.finish()
		self.tick()
		self.assertEqual(self.voice.speech_clips(), ["greeting"])

	def test_no_tts_plays_acknowledge(self):

		future = self.to_thinking()
		self.voice.finish()
		future.set_result(dlg.TurnResult("hi", reply("I am printed"), None))
		self.tick()
		self.assertEqual(self.m.state, dlg.RESPONDING)
		self.assertEqual(self.voice.speech_clips()[-1], "acknowledge")
		self.assertTrue(any("I am printed" in line for line in self.logs))



class TestFacing(DialogueTestBase):

	face_user = "apriltag"

	def test_face_done(self):

		self.m.on_wake()
		self.assertEqual(self.m.state, dlg.FACING)
		self.tick(n=10)
		self.assertEqual(self.m.state, dlg.FACING)
		self.m.face_done()
		self.tick()
		self.assertEqual(self.m.state, dlg.GREETING)

	def test_face_backstop_timeout(self):

		self.m.on_wake()
		self.tick(dt=11.0)
		self.assertEqual(self.m.state, dlg.GREETING)



class TestFailures(DialogueTestBase):

	def test_empty_transcript(self):

		future = self.to_thinking()
		self.voice.finish()
		self.tick()
		future.set_result(dlg.TurnResult(""))
		self.tick()
		self.assertEqual(self.m.state, dlg.LISTENING)
		self.assertEqual(self.hearing.arms, 2)
		self.assertFalse(self.voice.music)
		self.assertNotIn("reply", self.events())

	def test_error(self):

		future = self.to_thinking()
		self.voice.finish()
		future.set_exception(RuntimeError("secret-ish details"))
		self.tick()
		self.assertEqual(self.m.state, dlg.LISTENING)
		self.assertEqual(self.voice.speech_clips()[-1], "acknowledge")
		# only the class name is logged
		self.assertTrue(any("RuntimeError" in line for line in self.logs))
		self.assertFalse(any("secret-ish" in line for line in self.logs))

	def test_timeout_then_late_result_ignored(self):

		future = self.to_thinking()
		self.voice.finish()
		self.tick(dt=61.0)
		self.assertEqual(self.m.state, dlg.LISTENING)
		self.assertEqual(self.voice.speech_clips()[-1], "acknowledge")
		self.voice.finish()
		self.events()

		# the cloud answers after all: ignored
		future.set_result(dlg.TurnResult("late", reply("too late"), speech()))
		self.tick(n=3)
		self.assertEqual(self.m.state, dlg.LISTENING)
		self.assertEqual(self.voice.said, [])
		self.assertEqual(self.events(), [])

	def test_listen_backstop(self):

		self.to_listening()
		self.tick(dt=41.0)
		self.assertEqual(self.m.state, dlg.GOODBYE)



class TestEnding(DialogueTestBase):

	def test_no_speech_goodbye(self):

		self.to_listening()
		self.events()
		self.m.on_hearing("no_speech")
		self.assertEqual(self.m.state, dlg.GOODBYE)
		self.assertEqual(self.voice.speech_clips()[-1], "goodbye")
		self.tick(n=5)
		self.assertEqual(self.m.state, dlg.GOODBYE)
		self.voice.finish()
		self.tick()
		self.assertEqual(self.m.state, dlg.IDLE)
		self.assertEqual(self.events(), ["ended"])
		self.assertFalse(self.m.active())

	def test_stop_word(self):

		future = self.to_thinking()
		cancels = self.hearing.cancels
		self.m.on_stop()
		self.assertEqual(self.m.state, dlg.GOODBYE)
		self.assertGreater(self.hearing.cancels, cancels)
		self.assertIsNone(self.m.job)
		self.voice.finish()
		self.tick()
		self.assertEqual(self.m.state, dlg.IDLE)
		# a stop when idle does nothing
		self.m.on_stop()
		self.assertEqual(self.m.state, dlg.IDLE)

	def test_sleep_command(self):

		future = self.to_thinking()
		self.voice.finish()
		future.set_result(dlg.TurnResult("go to sleep", reply("Night night!", sleep=True), speech()))
		self.tick()
		self.assertEqual(self.m.state, dlg.RESPONDING)
		self.voice.finish()
		self.tick(dt=0.3)
		self.assertEqual(self.m.state, dlg.GOODBYE)
		self.voice.finish()
		self.tick(dt=0.3)
		self.assertEqual(self.m.state, dlg.IDLE)
		names = self.events()
		self.assertIn("sleep", names)
		self.assertEqual(names[-1], "ended")



class TestTasks(DialogueTestBase):

	def test_tricks_hand_off_and_resume(self):

		future = self.to_thinking()
		self.voice.finish()
		future.set_result(dlg.TurnResult("do some tricks", reply("Show me!", tricks=True), speech()))
		self.tick()
		# the reply is said first, then the task
		self.assertEqual(self.m.state, dlg.RESPONDING)
		self.events()
		self.voice.finish()
		self.tick(dt=0.3)
		self.assertEqual(self.m.state, dlg.SUSPENDED)
		self.assertFalse(self.m.engaged())
		self.assertTrue(self.m.active())
		out = self.m.take_outbox()
		self.assertEqual(out, [("task", ("tricks", {"intro": False}))])

		# hearing events while suspended are ignored
		self.m.on_hearing("utterance", np.zeros(10, dtype=np.int16))
		self.assertEqual(self.m.state, dlg.SUSPENDED)

		# back: say the prompt, then listen
		prompt = speech(4800)
		self.m.resume(prompt)
		self.assertEqual(self.m.state, dlg.GREETING)
		self.tick()
		self.assertIs(self.voice.said[-1], prompt)
		self.voice.finish()
		self.tick(dt=1.1)
		self.assertEqual(self.m.state, dlg.LISTENING)

	def test_dance_hand_off_is_immediate(self):

		future = self.to_thinking()
		self.voice.finish()
		self.tick()
		future.set_result(dlg.TurnResult("dance to abba", reply("Yay!", dance=True, song="abba"), None))
		self.tick()
		self.assertEqual(self.m.state, dlg.SUSPENDED)
		self.assertEqual(self.voice.said, [])
		self.assertFalse(self.voice.music)
		out = dict(self.m.take_outbox())
		self.assertEqual(out["task"], ("dance", {"intro": False, "query": "abba"}))

	def test_resume_without_prompt_plays_greeting(self):

		self.to_listening()
		self.m.suspend("identify")
		self.assertEqual(self.m.state, dlg.SUSPENDED)
		self.m.resume(None)
		self.tick()
		self.assertEqual(self.voice.speech_clips()[-1], "greeting")

	def test_stop_while_suspended(self):

		self.to_listening()
		self.m.suspend("dance")
		self.m.on_stop()
		self.assertEqual(self.m.state, dlg.GOODBYE)



class TestStale(DialogueTestBase):

	def test_stale_future_after_new_episode(self):

		old = self.to_thinking()
		self.m.on_stop()
		self.voice.finish()
		self.tick()
		self.assertEqual(self.m.state, dlg.IDLE)
		self.events()

		# a new conversation reaches THINKING with a new job
		new = self.to_thinking()
		self.assertIsNot(old, new)
		self.voice.finish()
		self.tick()

		# the old job finishes first: ignored
		old.set_result(dlg.TurnResult("old", reply("stale!"), speech()))
		self.tick(n=3)
		self.assertEqual(self.m.state, dlg.THINKING)
		self.assertNotIn("reply", self.events())
		self.assertEqual(self.voice.said, [])

		# the current one is used
		new.set_result(dlg.TurnResult("new", reply("fresh!"), speech()))
		self.tick()
		self.assertEqual(self.m.state, dlg.RESPONDING)
		self.assertEqual(self.m.reply.reply, "fresh!")



class FakeClients(object):

	def __init__(self, transcript="hello miro", answer=None, fail_chat=False):
		self.transcript = transcript
		self.answer = answer or json.dumps({"valence": 0.9, "arousal": 0.8, "emotion": "happy",
			"reply": "Hi!", "commands": {"spin": True}})
		self.fail_chat = fail_chat
		self.wav = None
		self.messages = None

	def transcribe(self, wav, model="whisper-1", language=None):
		self.wav = wav
		return self.transcript

	def chat(self, messages, model, temperature=None, max_tokens=None, json_mode=False):
		if self.fail_chat:
			raise RuntimeError("boom")
		self.messages = messages
		return self.answer



class FakeTts(object):

	def __init__(self, fail=False):
		self.calls = []
		self.fail = fail

	def synthesize(self, text, voice_id=None):
		self.calls.append((text, voice_id))
		if self.fail:
			raise RuntimeError("tts down")
		return np.ones(480, dtype=np.int16)



class Cfg(object):

	def __init__(self, data):
		self.data = data

	def get(self, key, default=None):
		return self.data.get(key, default)



class TestTurnRunner(unittest.TestCase):

	def make(self, clients, tts):

		self.memory = ChatMemory(4)
		self.memory.add("earlier question", "earlier answer")
		cfg = Cfg({"tts.impression_voice_id": "impression-voice"})
		return dlg.TurnRunner(clients, tts, self.memory, "SYSTEM PROMPT", cfg)

	def test_turn(self):

		clients = FakeClients()
		tts = FakeTts()
		runner = self.make(clients, tts)
		result = runner(np.zeros(1600, dtype=np.int16), 0.5, 0.6, "Monday")
		self.assertEqual(result.transcript, "hello miro")
		self.assertEqual(result.reply.reply, "Hi!")
		self.assertTrue(result.reply.commands["spin"])
		self.assertEqual(len(result.speech), 480)
		self.assertEqual(tts.calls, [("Hi!", None)])
		self.assertTrue(clients.wav.startswith(b"RIFF"))
		roles = [m["role"] for m in clients.messages]
		self.assertEqual(roles, ["system", "user", "assistant", "system", "user"])
		self.assertEqual(clients.messages[-1]["content"], "hello miro")
		# the runner never writes the memory (the tick does, if it accepts)
		self.assertEqual(len(self.memory), 1)

	def test_silence(self):

		tts = FakeTts()
		runner = self.make(FakeClients(transcript="  you "), tts)
		result = runner(np.zeros(1600, dtype=np.int16))
		self.assertEqual(result.transcript, "")
		self.assertEqual(tts.calls, [])

	def test_tts_failure_keeps_reply(self):

		runner = self.make(FakeClients(), FakeTts(fail=True))
		result = runner(np.zeros(1600, dtype=np.int16))
		self.assertIsNone(result.speech)
		self.assertEqual(result.tts_error, "RuntimeError")
		self.assertEqual(result.reply.reply, "Hi!")

	def test_impression_voice_and_dance(self):

		answer = json.dumps({"reply": "Hot dog!", "commands": {"impression": True}})
		tts = FakeTts()
		self.make(FakeClients(answer=answer), tts)(np.zeros(160, dtype=np.int16))
		self.assertEqual(tts.calls, [("Hot dog!", "impression-voice")])

		answer = json.dumps({"reply": "Let's dance!", "commands": {"dance": True, "song": "abba"}})
		tts = FakeTts()
		result = self.make(FakeClients(answer=answer), tts)(np.zeros(160, dtype=np.int16))
		self.assertEqual(tts.calls, [])
		self.assertEqual(result.reply.commands["song"], "abba")

	def test_chat_error_propagates(self):

		runner = self.make(FakeClients(fail_chat=True), FakeTts())
		with self.assertRaises(RuntimeError):
			runner(np.zeros(160, dtype=np.int16))

	def test_no_clients(self):

		runner = self.make(None, None)
		with self.assertRaises(RuntimeError):
			runner(np.zeros(160, dtype=np.int16))



if __name__ == "__main__":
	unittest.main()
