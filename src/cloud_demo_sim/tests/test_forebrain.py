#
#	Tests for NodeForebrain (commands, touch, keyword mask, task slot,
#	scene mode), NodeDialogue's outbox handling, NodeCues, and the
#	ForebrainAction priority logic, with stub nodes (no ROS, no cloud).
#

import _testpath
_testpath.install_stubs()

import concurrent.futures
import json
import types
import unittest

import numpy as np

import heymiro_config
from heymiro_config import Section
import node_forebrain
from node_forebrain import NodeForebrain, parse_command, RESUME_PROMPT
from node_dialogue import NodeDialogue
from node_cues import NodeCues
from action import forebrain_action
from action.action_types import ActionInput
from action.action_converse import ActionConverse
from action.action_tricks import ActionTricks
from action.action_identify import ActionIdentify
from cloud.llm import LlmReply
from forebrain import dialogue as dlg


FEATURES = {
	"conversation": True, "wake_word": True, "dance": True, "tricks": True, "identify": True,
	"touch_reactions": True, "purr": True, "sleep_command": True, "spontaneous_gestures": False,
	"spotify": False, "log_transcripts": False, "tts": False, "vad_cobra": False,
}


def make_config(**features):

	# the real settings, with a fixed feature set (no keys needed)
	cfg = heymiro_config.load(environ={"HEYMIRO_SECRETS_FILE": "/nonexistent/secrets.env"})
	f = dict(FEATURES)
	f.update(features)
	cfg.features = Section(f)
	return cfg



class Clock(object):

	def __init__(self):
		self.t = 1000.0

	def __call__(self):
		return self.t



class FakeWorker(object):

	def __init__(self):
		self.jobs = []
		self.closed = False

	def submit(self, fn, *args, **kwargs):
		future = concurrent.futures.Future()
		self.jobs.append((fn, args, future))
		return future

	def shutdown(self, wait=False):
		self.closed = True



class FakeHearing(object):

	def __init__(self):
		self.events = []
		self.mask = "never set"
		self.arms = 0
		self.cancels = 0

	def poll_events(self):
		out = self.events
		self.events = []
		return out

	def set_keyword_mask(self, ids):
		self.mask = ids

	def arm_utterance(self):
		self.arms += 1

	def cancel_utterance(self):
		self.cancels += 1



class FakeVoice(object):

	def __init__(self):
		self.clips = []
		self.said = []
		self.stops = []
		self.speaking = False
		self.music = False

	def play_clip(self, category, channel="speech"):
		self.clips.append(category)
		return 1.0

	def say_pcm(self, pcm):
		self.said.append(pcm)
		return 1.0

	def stop(self, channel=None):
		self.stops.append(channel)

	def is_speaking(self):
		return self.speaking

	def is_music_playing(self):
		return self.music



class FakeAffect(object):

	def __init__(self):
		self.targets = []
		self.woken = 0
		self.sleeps = 0
		self.emotion = types.SimpleNamespace(valence=0.5, arousal=0.5)

	def set_forebrain_target(self, valence, arousal, gain=0.05, seconds=2.0):
		self.targets.append((valence, arousal))

	def wake(self):
		self.woken += 1

	def request_sleep(self):
		self.sleeps += 1



class FakeDialogue(object):

	def __init__(self):
		self.state = "idle"
		self.calls = []

	def engaged(self):
		return self.state not in ("idle", "suspended")

	def active(self):
		return self.state != "idle"

	def suspended(self):
		return self.state == "suspended"

	def state_name(self):
		return self.state

	def on_wake(self):
		self.calls.append("wake")
		self.state = "greeting"

	def on_stop(self):
		self.calls.append("stop")
		if self.state != "idle":
			self.state = "goodbye"

	def on_hearing(self, name, payload=None):
		self.calls.append(name)

	def suspend_for(self, task):
		self.calls.append("suspend:" + task)
		self.state = "suspended"

	def resume(self):
		self.calls.append("resume")
		self.state = "greeting"



class FakeCues(object):

	def __init__(self):
		self.mode = None
		self.perks = 0

	def set_mode(self, mode):
		self.mode = mode

	def perk(self):
		self.perks += 1



class FakeExpress(object):

	def __init__(self):
		self.overrides = {}
		self.released = []

	def override(self, channel, value, ttl_ticks=5, owner=""):
		self.overrides[channel] = (list(value), ttl_ticks, owner)

	def release(self, owner):
		self.released.append(owner)
		for k in list(self.overrides):
			if self.overrides[k][2] == owner:
				del self.overrides[k]



class FakeSongs(object):

	def __init__(self):
		self.songs = [types.SimpleNamespace(id="dancing_queen", title="Dancing Queen", path="x.mp3", bpm=100.0, genre="disco")]

	def find(self, query):
		return self.songs[0] if query and "abba" in query.lower() else None

	def random(self):
		return self.songs[0]



def make_system(heymiro=None):

	pars = types.SimpleNamespace(
		forebrain=heymiro or make_config(),
		action=types.SimpleNamespace(priority_forebrain=0.84, priority_medium=0.5, priority_attend=0.8),
		express=types.SimpleNamespace(double_blink_prob=0.2),
		timing=types.SimpleNamespace(tick_hz=50.0),
		dev=types.SimpleNamespace(DEBUG_ACTION_PARAMS=False))
	state = types.SimpleNamespace(pet=0, stroke=0.0, scene_gesture=None, scene_objects=None,
		detect_objects_for_50Hz=[None, None], audio_events_for_50Hz=[], in_speaking=0.0)
	output = types.SimpleNamespace(scene_mode="off", forebrain_state=None)
	nodes = types.SimpleNamespace(hearing=FakeHearing(), voice=FakeVoice(), affect=FakeAffect(),
		cues=FakeCues(), express=FakeExpress())
	return types.SimpleNamespace(pars=pars, kc_s=None, kc_m=None, input=types.SimpleNamespace(stream=None),
		state=state, output=output, nodes=nodes)



class ForebrainTestBase(unittest.TestCase):

	def setUp(self):

		self.sys = make_system()
		self.clock = Clock()
		self.wall = Clock()
		self.worker = FakeWorker()
		self.cloud = node_forebrain.CloudServices()
		self.cloud.songs = FakeSongs()
		self.fb = NodeForebrain(self.sys, clock=self.clock, wall=self.wall, worker=self.worker,
			cloud=self.cloud, rng=types.SimpleNamespace(random=lambda: 0.0))
		self.sys.nodes.forebrain = self.fb
		self.dialogue = FakeDialogue()
		self.sys.nodes.dialogue = self.dialogue
		self.nodes = self.sys.nodes

	def tick(self, dt=0.02, n=1):

		for _ in range(n):
			self.clock.t += dt
			self.wall.t += dt
			self.fb.tick()



class TestCommands(ForebrainTestBase):

	def test_parse(self):

		self.assertEqual(parse_command("dance abba"), ("dance", "abba"))
		self.assertEqual(parse_command("  Dance  Dancing Queen "), ("dance", "Dancing Queen"))
		self.assertEqual(parse_command("dance"), ("dance", ""))
		self.assertEqual(parse_command("STOP"), ("stop", ""))
		self.assertEqual(parse_command("say Hello there"), ("say", "Hello there"))
		self.assertEqual(parse_command("converse"), ("converse", ""))
		self.assertEqual(parse_command("tricks"), ("tricks", ""))
		self.assertEqual(parse_command("identify"), ("identify", ""))
		self.assertEqual(parse_command("fly away"), (None, "fly away"))
		self.assertEqual(parse_command(""), (None, ""))

	def test_dance_command(self):

		self.fb.on_command("dance abba")
		self.tick()
		self.assertEqual(len(self.worker.jobs), 1)
		fn, args, future = self.worker.jobs[0]
		self.assertEqual(args, ("abba",))
		self.assertTrue(self.fb.engaged())
		# busy: only the stop word is listened for
		self.assertEqual(self.nodes.hearing.mask, frozenset(["stop_miro"]))

		# the song is ready: the dance task is posted
		pcm = np.zeros(2400, dtype=np.int16)
		future.set_result({"song": self.cloud.songs.songs[0], "pcm": pcm, "intro": True, "query": "abba", "found": True})
		self.tick()
		self.assertEqual(self.fb.pending_task(), "dance")
		params = self.fb.claim_task("dance")
		self.assertIs(params["pcm"], pcm)
		self.assertEqual(self.fb.active_task, "dance")
		self.assertEqual(self.fb.compute_cue_mode(), "dancing")

	def test_resolve_song_on_worker(self):

		import cloud.music
		saved = cloud.music.decode_mp3
		cloud.music.decode_mp3 = lambda path, rate=24000, max_s=None: np.ones(10, dtype=np.int16)
		try:
			params = self.fb._resolve_song("abba please")
			self.assertEqual(params["song"].id, "dancing_queen")
			self.assertTrue(params["found"])
			params = self.fb._resolve_song("unknown song")
			self.assertFalse(params["found"])
			self.assertEqual(len(params["pcm"]), 10)
		finally:
			cloud.music.decode_mp3 = saved

	def test_say_without_tts(self):

		self.fb.on_command("say hello")
		self.tick()
		self.assertEqual(self.worker.jobs, [])
		self.assertEqual(self.nodes.voice.said, [])

	def test_say_with_tts(self):

		self.cloud.tts = types.SimpleNamespace(synthesize=lambda text, voice_id=None: None)
		self.fb.on_command("say Hello there")
		self.tick()
		self.assertEqual(len(self.worker.jobs), 1)
		fn, args, future = self.worker.jobs[0]
		self.assertEqual(args[0], "Hello there")
		self.assertTrue(self.fb.is_saying())
		pcm = np.ones(100, dtype=np.int16)
		future.set_result(pcm)
		self.tick()
		self.assertIs(self.nodes.voice.said[-1], pcm)
		self.assertFalse(self.fb.is_saying())
		# cached: the second time is immediate
		self.fb.say("Hello there")
		self.assertEqual(len(self.worker.jobs), 1)
		self.assertEqual(len(self.nodes.voice.said), 2)

	def test_stop_command(self):

		self.fb.request_task("tricks", {})
		self.fb.claim_task("tricks")
		self.dialogue.state = "suspended"
		self.fb.on_command("stop")
		self.tick()
		self.assertIsNone(self.fb.active_task)
		self.assertIsNone(self.fb.pending_task())
		self.assertIn(None, self.nodes.voice.stops)
		self.assertIn("stop", self.dialogue.calls)

	def test_stop_cancels_dance_preparation(self):

		self.fb.on_command("dance")
		self.tick()
		fn, args, future = self.worker.jobs[0]
		self.fb.on_command("stop")
		self.tick()
		future.set_result({"song": self.cloud.songs.songs[0], "pcm": np.zeros(10), "intro": True})
		self.tick()
		self.assertIsNone(self.fb.pending_task())

	def test_converse_command_and_wake_word(self):

		self.fb.on_command("converse")
		self.tick()
		self.assertEqual(self.dialogue.calls, ["wake"])
		self.assertEqual(self.nodes.affect.woken, 1)
		self.assertEqual(self.nodes.cues.perks, 1)

	def test_wake_words(self):

		self.nodes.hearing.events = [("wake", "hey_miro")]
		self.tick()
		self.assertEqual(self.dialogue.calls, ["wake"])
		# hearing events in a conversation go to the dialogue
		self.nodes.hearing.events = [("voiced", None), ("utterance", np.zeros(4))]
		self.tick()
		self.assertEqual(self.dialogue.calls, ["wake", "voiced", "utterance"])
		self.nodes.hearing.events = [("wake", "stop_miro")]
		self.tick()
		self.assertIn("stop", self.dialogue.calls)

	def test_dance_wake_word(self):

		self.nodes.hearing.events = [("wake", "dance_miro")]
		self.tick()
		self.assertEqual(len(self.worker.jobs), 1)
		self.assertEqual(self.worker.jobs[0][1], (None,))

	def test_wake_without_conversation(self):

		self.sys.pars.forebrain = make_config(conversation=False)
		fb = NodeForebrain(self.sys, clock=self.clock, worker=self.worker, cloud=self.cloud)
		self.sys.nodes.forebrain = fb
		fb.on_command("converse")
		fb.tick()
		self.assertEqual(self.dialogue.calls, [])
		self.assertEqual(self.nodes.voice.clips, ["greeting"])



class TestTouch(ForebrainTestBase):

	def hits(self, n, where="ear", dt=0.1):

		for _ in range(n):
			self.nodes.hearing.events = [("touch", where)]
			self.tick(dt=dt)

	def test_hits_cooldown(self):

		self.hits(3)
		self.assertEqual(self.nodes.voice.clips, [])
		self.hits(1)
		self.assertEqual(self.nodes.voice.clips, ["ouch_ear"])
		self.assertEqual(self.nodes.affect.targets, [(0.2, 0.8)])

		# within the cooldown (3 s): nothing
		self.hits(4)
		self.assertEqual(self.nodes.voice.clips, ["ouch_ear"])

		# after it: again (tail this time)
		self.tick(dt=3.0)
		self.hits(4, "tail")
		self.assertEqual(self.nodes.voice.clips, ["ouch_ear", "ouch_tail"])

	def test_hits_outside_window(self):

		# 4 hits spread over more than window_s (2 s) do not count
		self.hits(4, dt=0.8)
		self.assertEqual(self.nodes.voice.clips, [])

	def test_no_reaction_when_engaged(self):

		self.dialogue.state = "listening"
		self.hits(6)
		self.assertEqual(self.nodes.voice.clips, [])
		self.assertEqual(self.nodes.affect.targets, [])

	def test_petting(self):

		self.sys.state.pet = 2
		self.tick()
		self.assertEqual(self.nodes.voice.clips, ["pleased"])
		self.tick(n=10)
		self.assertEqual(self.nodes.voice.clips, ["pleased"])
		self.tick(dt=7.0)
		self.assertEqual(self.nodes.voice.clips, ["pleased", "pleased"])



class TestModulation(ForebrainTestBase):

	def test_keyword_mask(self):

		self.tick()
		self.assertIsNone(self.nodes.hearing.mask)
		self.dialogue.state = "listening"
		self.tick()
		self.assertEqual(self.nodes.hearing.mask, frozenset(["stop_miro"]))
		self.dialogue.state = "idle"
		self.tick()
		self.assertIsNone(self.nodes.hearing.mask)

	def test_scene_mode(self):

		self.tick()
		self.assertEqual(self.sys.output.scene_mode, "off")
		self.fb.request_task("tricks", {})
		self.tick()
		self.assertEqual(self.sys.output.scene_mode, "gestures")
		self.fb.claim_task("tricks")
		self.tick()
		self.assertEqual(self.sys.output.scene_mode, "gestures")
		self.assertEqual(self.nodes.cues.mode, "tricks")
		self.fb.end_task("tricks")
		self.fb.request_task("identify", {})
		self.fb.claim_task("identify")
		self.tick()
		self.assertEqual(self.sys.output.scene_mode, "objects")
		self.fb.end_task("identify")
		self.tick()
		self.assertEqual(self.sys.output.scene_mode, "off")

	def test_spontaneous_gestures(self):

		self.sys.pars.forebrain = make_config(spontaneous_gestures=True)
		fb = NodeForebrain(self.sys, clock=self.clock, wall=self.wall, worker=self.worker, cloud=self.cloud)
		self.sys.nodes.forebrain = fb
		fb.tick()
		self.assertEqual(self.sys.output.scene_mode, "gestures")
		self.sys.state.scene_gesture = (self.wall.t, {"t": 1.0, "gesture": "Victory", "score": 0.9})
		fb.tick()
		self.assertEqual(fb.pending_task(), "tricks")

	def test_cue_mode_follows_dialogue(self):

		for state, mode in (("idle", ""), ("listening", "listening"), ("thinking", "thinking"),
				("responding", "responding"), ("suspended", "")):
			self.dialogue.state = state
			self.tick()
			self.assertEqual(self.nodes.cues.mode, mode)

	def test_state_json(self):

		self.tick()
		state = json.loads(self.sys.output.forebrain_state)
		self.assertEqual(state["dialogue"], "idle")
		self.sys.output.forebrain_state = None
		self.tick()
		self.assertIsNone(self.sys.output.forebrain_state)
		self.dialogue.state = "listening"
		self.tick()
		self.assertEqual(json.loads(self.sys.output.forebrain_state)["dialogue"], "listening")



class TestTaskSlot(ForebrainTestBase):

	def test_request_claim_end(self):

		self.assertTrue(self.fb.request_task("tricks", {"intro": True}))
		self.assertEqual(self.fb.pending_task(), "tricks")
		self.assertIsNone(self.fb.claim_task("dance"))
		self.assertEqual(self.fb.claim_task("tricks"), {"intro": True})
		self.assertIsNone(self.fb.claim_task("tricks"))
		self.assertEqual(self.fb.active_task, "tricks")
		self.assertTrue(self.fb.engaged())
		self.fb.end_task("tricks", "done")
		self.assertIsNone(self.fb.active_task)
		self.assertFalse(self.fb.engaged())

	def test_suspend_and_resume_dialogue(self):

		self.dialogue.state = "listening"
		self.fb.request_task("identify", {})
		self.assertIn("suspend:identify", self.dialogue.calls)
		self.fb.claim_task("identify")
		self.tick()
		self.fb.end_task("identify", "done")
		self.tick()
		self.assertEqual(self.dialogue.calls[-1], "resume")

	def test_resume_waits_for_closing_line(self):

		self.cloud.tts = types.SimpleNamespace(synthesize=lambda text, voice_id=None: None)
		self.dialogue.state = "suspended"
		self.fb.request_task("identify", {})
		self.fb.claim_task("identify")
		self.fb.say("I couldn't see anything")
		self.fb.end_task("identify", "done")
		self.tick()
		self.assertNotIn("resume", self.dialogue.calls)
		# the closing line arrives and is said; then the dialogue resumes
		for fn, args, future in self.worker.jobs:
			future.set_result(np.ones(10, dtype=np.int16))
		self.tick()
		self.tick()
		self.assertIn("resume", self.dialogue.calls)

	def test_disabled_task(self):

		self.sys.pars.forebrain = make_config(identify=False)
		fb = NodeForebrain(self.sys, clock=self.clock, worker=self.worker, cloud=self.cloud)
		self.assertFalse(fb.request_task("identify", {}))
		self.assertIsNone(fb.pending_task())

	def test_unclaimed_task_dropped(self):

		self.dialogue.state = "suspended"
		self.fb.request_task("tricks", {})
		self.tick(dt=1.0)
		self.assertEqual(self.fb.pending_task(), "tricks")
		self.tick(dt=5.0)
		self.assertIsNone(self.fb.pending_task())
		self.assertEqual(self.dialogue.calls[-1], "resume")

	def test_shutdown(self):

		self.fb.shutdown()
		self.assertTrue(self.worker.closed)



class TestNodeDialogue(ForebrainTestBase):

	def setUp(self):

		ForebrainTestBase.setUp(self)
		self.nd = NodeDialogue(self.sys, clock=self.clock)
		self.sys.nodes.dialogue = self.nd

	def test_reply_drives_affect(self):

		self.nd.machine.outbox.append(("reply", LlmReply(valence=0.9, arousal=0.2, emotion="happy", reply="Yay")))
		self.nd.machine.outbox.append(("reply", LlmReply(emotion="sad", reply="Oh no")))
		self.nd.tick()
		self.assertEqual(self.nodes.affect.targets, [(0.9, 0.2), (0.15, 0.3)])

	def test_sleep_and_memory(self):

		self.nd.machine.outbox.append(("remember", ("hi", "hello")))
		self.nd.tick()
		self.assertEqual(len(self.nd.memory), 1)
		self.nd.machine.outbox.append(("sleep", None))
		self.nd.machine.outbox.append(("ended", 1))
		self.nd.tick()
		self.assertEqual(self.nodes.affect.sleeps, 1)
		self.assertEqual(len(self.nd.memory), 0)

	def test_task_from_reply(self):

		self.nd.machine.state = dlg.SUSPENDED
		self.nd.machine.outbox.append(("task", ("tricks", {"intro": False})))
		self.nd.tick()
		self.assertEqual(self.fb.pending_task(), "tricks")
		self.nd.machine.outbox.append(("task", ("dance", {"intro": False, "query": "abba"})))
		self.nd.tick()
		self.assertEqual(self.worker.jobs[-1][1], ("abba",))

	def test_wake_reaches_greeting(self):

		self.fb.on_command("converse")
		self.tick()
		self.assertTrue(self.nd.engaged())
		self.assertIn(self.nd.state_name(), (dlg.FACING, dlg.GREETING))



class TestCues(unittest.TestCase):

	def test_modes(self):

		sys = make_system()
		express = sys.nodes.express
		cues = NodeCues(sys, rng=types.SimpleNamespace(random=lambda: 0.99))
		cues.tick()
		self.assertNotIn("eyelids", express.overrides)
		cues.set_mode("thinking")
		cues.tick()
		self.assertEqual(express.overrides["eyelids"][0], [0.3, 0.3])
		cues.set_mode("listening")
		cues.rng = types.SimpleNamespace(random=lambda: 0.0)
		cues.tick()
		self.assertEqual(express.overrides["eyelids"][0], [1.0, 1.0])
		cues.set_mode("bogus")
		self.assertEqual(cues.mode, "")
		cues.perk()
		cues.tick()
		self.assertEqual(express.overrides["ears"][0], [0.0, 0.0])



class FakeKc(object):

	def __init__(self):
		self.cfg = [-0.1, 0.6, 0.0, 0.0]
		self.set = []

	def getConfig(self):
		return list(self.cfg)

	def setConfig(self, cfg):
		self.cfg = list(cfg)
		self.set.append(list(cfg))



class FakeForebrain(object):

	def __init__(self):
		self.pending = None
		self.active_task = None
		self.ended = []
		self.said = []
		self.wall = lambda: 0.0
		self.worker = FakeWorker()
		self.cloud = node_forebrain.CloudServices()

	def pending_task(self):
		return self.pending

	def claim_task(self, name):
		if self.pending != name:
			return None
		self.pending = None
		self.active_task = name
		return {}

	def end_task(self, name, reason="done"):
		self.ended.append((name, reason))
		if self.active_task == name:
			self.active_task = None

	def say(self, text):
		self.said.append(text)
		return False

	def is_saying(self):
		return False

	def fresh_gesture(self):
		return None



class StubParent(object):

	def __init__(self):
		sys = make_system()
		self.pars = sys.pars
		self.kc_m = FakeKc()
		self.action_input = ActionInput()
		self.state = sys.state
		self.output = sys.output
		self.nodes = sys.nodes
		self.nodes.forebrain = FakeForebrain()
		self.nodes.dialogue = FakeDialogue()
		self.pushes = []

	def apply_push(self, push, retreatable):
		self.pushes.append(push)



class ProbeAction(forebrain_action.ForebrainAction):

	NAME = "probe"
	TASK = "probe"

	def init(self):
		self.want = False
		self.started = 0

	def wanted(self):
		return self.want

	def on_start(self):
		self.started += 1



class TestForebrainAction(unittest.TestCase):

	def setUp(self):

		self.parent = StubParent()
		self.a = ProbeAction(self.parent)

	def test_finalize(self):

		self.assertEqual(self.a.name, "probe")
		self.assertFalse(self.a.modulate_by_wakefulness)

	def test_priority(self):

		self.a.ascending()
		self.assertEqual(self.a.interface.priority, 0.0)
		self.a.want = True
		self.a.ascending()
		self.assertAlmostEqual(self.a.interface.priority, 0.84)
		self.parent.action_input.conf_surf = 0.5
		self.a.ascending()
		self.assertAlmostEqual(self.a.interface.priority, 0.42)

	def test_selected_and_running(self):

		self.a.want = True
		self.a.interface.inhibition = 0.0
		self.a.descending()
		self.assertEqual(self.a.started, 1)
		self.assertTrue(self.a.clock.isActive())
		self.a.ascending()
		self.assertAlmostEqual(self.a.interface.priority, 0.84)

		# held open: the clock never runs out while wanted
		for _ in range(200):
			self.a.descending()
		self.assertTrue(self.a.clock.isActive())

		# no longer wanted: it ends itself, releases overrides, ends its task
		self.parent.nodes.forebrain.active_task = "probe"
		self.parent.nodes.express.override("illum", [0] * 6, owner="probe")
		self.a.want = False
		self.a.descending()
		self.assertFalse(self.a.clock.isActive())
		self.assertIn("probe", self.parent.nodes.express.released)
		self.assertEqual(self.parent.nodes.forebrain.ended, [("probe", "done")])
		self.a.ascending()
		self.assertEqual(self.a.interface.priority, 0.0)

	def test_preempted(self):

		self.a.want = True
		self.a.interface.inhibition = 0.0
		self.a.descending()
		self.parent.nodes.forebrain.active_task = "probe"
		self.a.interface.inhibition = 1.0
		self.a.descending()
		self.assertFalse(self.a.clock.isActive())
		self.assertEqual(self.parent.nodes.forebrain.ended, [("probe", "preempted")])

	def test_not_started_if_unwanted(self):

		self.a.interface.inhibition = 0.0
		self.a.descending()
		self.assertEqual(self.a.started, 0)
		self.assertFalse(self.a.clock.isActive())

	def test_motor_helpers(self):

		self.a.set_head([0.0, 5.0, -5.0, float("nan")])
		cfg = self.parent.kc_m.set[-1]
		self.assertAlmostEqual(cfg[1], 60.0 * np.pi / 180.0, places=4)
		self.assertAlmostEqual(cfg[2], -55.0 * np.pi / 180.0, places=4)
		self.assertTrue(np.isfinite(cfg[3]))

		self.parent.action_input.conf_surf = 0.5
		self.a.body_velocity(0.2, 0.4)
		push = self.parent.pushes[-1]
		self.assertEqual(list(push.vec), [0.1, 0.4, 0.0])
		import miro2
		self.assertEqual(push.flags, miro2.constants.PUSH_FLAG_VELOCITY)



class TestActions(unittest.TestCase):

	def setUp(self):

		self.parent = StubParent()
		self.fb = self.parent.nodes.forebrain

	def test_task_actions_wanted(self):

		tricks = ActionTricks(self.parent)
		identify = ActionIdentify(self.parent)
		self.assertFalse(tricks.wanted())
		self.fb.pending = "tricks"
		self.assertTrue(tricks.wanted())
		self.assertFalse(identify.wanted())
		tricks.interface.inhibition = 0.0
		tricks.descending()
		self.assertEqual(self.fb.active_task, "tricks")
		self.assertTrue(tricks.wanted())
		self.assertTrue(tricks.clock.isActive())

	def test_tricks_end_after_duration(self):

		tricks = ActionTricks(self.parent)
		t = [0.0]
		tricks.clock_fn = lambda: t[0]
		self.fb.pending = "tricks"
		tricks.interface.inhibition = 0.0
		tricks.descending()
		t[0] = 25.0
		tricks.descending()
		self.assertFalse(tricks.clock.isActive())
		self.assertEqual(self.fb.ended, [("tricks", "done")])

	def test_converse_faces_apriltag(self):

		converse = ActionConverse(self.parent)
		converse.face_user = "apriltag"
		d = self.parent.nodes.dialogue
		faced = []
		d.state = "facing"
		d.state_age = lambda: 0.0
		d.face_done = lambda: faced.append(True)
		self.assertTrue(converse.wanted())
		converse.interface.inhibition = 0.0
		converse.descending()
		tagged = types.SimpleNamespace(tags=[object()])
		for i in range(5):
			self.parent.state.detect_objects_for_50Hz = [tagged if i % 2 == 0 else None, None]
			converse.descending()
		self.assertTrue(faced)
		# it rotated on the spot while looking
		self.assertTrue(len(self.parent.pushes) >= 1)
		self.assertAlmostEqual(self.parent.pushes[0].vec[1], 0.08)


	def start_identify(self):

		identify = ActionIdentify(self.parent)
		t = [0.0]
		identify.clock_fn = lambda: t[0]
		self.fb.pending = "identify"
		identify.interface.inhibition = 0.0
		identify.descending()
		self.assertTrue(identify.clock.isActive())
		return identify, t

	def test_identify_confirms_and_comments(self):

		identify, t = self.start_identify()
		state = self.parent.state
		for i in range(4):
			t[0] += 0.2
			state.scene_objects = (0.0, {"t": float(i), "objects": [{"label": "banana", "score": 0.9}]})
			identify.descending()
			if i < 3:
				self.assertEqual(identify.phase, "look")
		self.assertEqual(identify.phase, "comment")
		self.assertEqual(identify.seen, ["banana"])

		# no LLM / TTS here: the worker job falls back to a template line
		fn, args, future = self.fb.worker.jobs[-1]
		text, pcm = fn(*args)
		self.assertIn("banana", text)
		self.assertIsNone(pcm)
		future.set_result((text, pcm))
		identify.descending()
		identify.descending()
		self.assertEqual(identify.phase, "look")

	def test_identify_nothing_seen(self):

		identify, t = self.start_identify()
		t[0] = 1.0
		identify.descending()
		t[0] = 17.0
		identify.descending()
		self.assertFalse(identify.clock.isActive())
		self.assertEqual(len(self.fb.said), 1)
		self.assertEqual(self.fb.ended, [("identify", "done")])

	def test_identify_stopped_says_nothing(self):

		identify, t = self.start_identify()
		self.fb.active_task = None      # what stop_all does
		identify.descending()
		self.assertFalse(identify.clock.isActive())
		self.assertEqual(self.fb.said, [])

	def test_converse_responding_flourishes(self):

		converse = ActionConverse(self.parent)
		t = [0.0]
		converse.clock_fn = lambda: t[0]
		d = self.parent.nodes.dialogue
		d.state = "responding"
		d.current_reply = lambda: LlmReply(emotion="sad", reply="x", commands={"spin": True, "move_ears": True})
		converse.interface.inhibition = 0.0
		converse.descending()
		t[0] = 0.1
		converse.descending()
		express = self.parent.nodes.express
		self.assertIn("illum", express.overrides)
		self.assertIn("ears", express.overrides)
		self.assertAlmostEqual(self.parent.pushes[-1].vec[1], 0.4)
		n = len(self.parent.pushes)
		t[0] = 6.0
		converse.descending()
		# spin is time-bounded (converse.spin_s = 3 s)
		self.assertEqual(len(self.parent.pushes), n)



if __name__ == "__main__":
	unittest.main()
