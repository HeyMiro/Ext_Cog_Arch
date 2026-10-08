#
#	Tests for the scene client (core/node_scene.py) and tools/make_clip.py.
#	The pure helpers are tested directly; NodeScene runs on a fake
#	DemoSystem with fake cv2 / mediapipe / YOLO objects, so neither ROS,
#	OpenCV, torch nor mediapipe is needed. make_clip runs with a fake TTS.
#

import _testpath
_testpath.install_stubs()

import contextlib
import io
import json
import os
import shutil
import tempfile
import time
import types
import unittest
import wave
from unittest import mock

import numpy as np

import heymiro_config
import node_scene
from node_scene import (KofN, GestureSmoother, ModeTracker, NodeScene, camera_intrinsics,
	class_ids_for, gesture_message, mode_wants, objects_from_result, objects_message,
	resolve_weights, top_gesture, yolo_device)
import make_clip

try:
	import cv2
except Exception:
	cv2 = None


# COCO names as an ultralytics model reports them (id -> name)
COCO = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
	"traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
	"horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
	"handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
	"baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
	"wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
	"broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
	"bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
	"microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
	"teddy bear", "hair drier", "toothbrush"]
COCO_NAMES = dict(enumerate(COCO))

# the ids HRI'25 hardcoded for its 13 classes
HRI_IDS = [2, 73, 67, 64, 47, 46, 49, 41, 42, 43, 44, 39, 32]

SECRET = "xi-test-SECRET-value-123"


def load_config(**overrides):

	env = {"HEYMIRO_CACHE_DIR": overrides.pop("cache", "/tmp/heymiro-test-cache")}
	if overrides:
		env["HEYMIRO_SET"] = ";".join(k + "=" + v for k, v in overrides.items())
	return heymiro_config.load(environ=env)


def category(name, score):

	return types.SimpleNamespace(category_name=name, score=score)



# --------------------------------------------------------------- helpers

class TestModes(unittest.TestCase):

	def test_mode_wants(self):

		self.assertEqual(mode_wants("off"), (False, False))
		self.assertEqual(mode_wants("gestures"), (True, False))
		self.assertEqual(mode_wants("objects"), (False, True))
		self.assertEqual(mode_wants("both"), (True, True))

	def test_tracker_staleness(self):

		m = ModeTracker(stale_s=3.0)
		self.assertEqual(m.current(100.0), "off")
		self.assertTrue(m.update("Gestures ", 100.0))
		self.assertEqual(m.current(102.9), "gestures")
		self.assertEqual(m.current(103.1), "off")
		m.update("objects", 103.5)
		self.assertEqual(m.current(104.0), "objects")

	def test_tracker_rejects_unknown(self):

		m = ModeTracker()
		m.update("both", 0.0)
		self.assertFalse(m.update("faces", 1.0))
		self.assertFalse(m.update(None, 1.0))
		self.assertEqual(m.current(1.0), "both")


class TestKofN(unittest.TestCase):

	def test_hri_four_of_ten(self):

		k = KofN(4, 10)
		for i in range(3):
			self.assertEqual(k.update({"cup": 0.8}), [])
		got = k.update({"cup": 0.6, "book": 0.9})
		self.assertEqual([c[0] for c in got], ["cup"])
		self.assertAlmostEqual(got[0][1], (0.8 * 3 + 0.6) / 4)
		# six empty frames: cup still 4 of the last 10
		for i in range(6):
			self.assertEqual([c[0] for c in k.update({})], ["cup"])
		# the oldest cup frame drops out
		self.assertEqual(k.update({}), [])

	def test_order_and_reset(self):

		k = KofN(2, 3)
		k.update({"a": 0.5, "b": 0.9})
		got = k.update({"a": 0.5, "b": 0.9})
		# equal counts: higher mean score first
		self.assertEqual([c[0] for c in got], ["b", "a"])
		k.reset()
		self.assertEqual(k.update({"a": 1.0}), [])

	def test_k_clamped(self):

		k = KofN(9, 2)
		self.assertEqual((k.k, k.n), (2, 2))
		k = KofN(0, 0)
		self.assertEqual((k.k, k.n), (1, 1))
		self.assertEqual(k.update({"x": 0.5}), [("x", 0.5)])


class TestGestures(unittest.TestCase):

	def test_three_of_five(self):

		g = GestureSmoother()
		self.assertEqual(g.update("Open_Palm", 0.9), (None, 0.0))
		self.assertEqual(g.update("Open_Palm", 0.7), (None, 0.0))
		name, score = g.update("Open_Palm", 0.8)
		self.assertEqual(name, "Open_Palm")
		self.assertAlmostEqual(score, 0.8)
		# held over two missing frames, then dropped
		self.assertEqual(g.update(None, 0.0)[0], "Open_Palm")
		self.assertEqual(g.update(None, 0.0)[0], "Open_Palm")
		self.assertEqual(g.update(None, 0.0)[0], None)

	def test_ignored_gestures_never_confirm(self):

		g = GestureSmoother()
		for name in ["Closed_Fist", "Thumb_Up", "Thumb_Down", "None", "Closed_Fist"]:
			self.assertEqual(g.update(name, 0.99), (None, 0.0))

	def test_top_gesture(self):

		result = types.SimpleNamespace(gestures=[
			[category("Closed_Fist", 0.99)],
			[category("Victory", 0.7), category("Open_Palm", 0.2)],
			[]])
		self.assertEqual(top_gesture(result), ("Victory", 0.7))
		result.gestures.append([category("Pointing_Up", 0.8)])
		self.assertEqual(top_gesture(result), ("Pointing_Up", 0.8))
		self.assertEqual(top_gesture(types.SimpleNamespace(gestures=[])), (None, 0.0))
		self.assertEqual(top_gesture(types.SimpleNamespace(gestures=[[category("None", 0.9)]])), (None, 0.0))


class FakeTensor(object):

	# torch-like: .cpu().numpy()
	def __init__(self, values):
		self.values = np.asarray(values, dtype=np.float32)

	def cpu(self):
		return self

	def numpy(self):
		return self.values


def fake_result(rows, tensor=True):

	# rows: (class id, score, [x0, y0, x1, y1] normalised)
	wrap = FakeTensor if tensor else np.asarray
	if not rows:
		boxes = types.SimpleNamespace(cls=wrap([]), conf=wrap([]), xyxyn=wrap(np.zeros((0, 4))))
	else:
		boxes = types.SimpleNamespace(
			cls=wrap([r[0] for r in rows]),
			conf=wrap([r[1] for r in rows]),
			xyxyn=wrap([r[2] for r in rows]))
	return types.SimpleNamespace(boxes=boxes)


class TestObjects(unittest.TestCase):

	def test_class_ids_match_hri(self):

		cfg = load_config()
		classes = cfg.cfg.identify.classes
		ids, missing = class_ids_for(COCO_NAMES, classes)
		self.assertEqual(missing, [])
		self.assertEqual(ids, sorted(HRI_IDS))
		# a list-style names table and odd spelling work too
		ids, missing = class_ids_for(COCO, ["Cell Phone ", "unicorn"])
		self.assertEqual(ids, [67])
		self.assertEqual(missing, ["unicorn"])

	def test_objects_from_result(self):

		rows = [(41, 0.55, [0.1, 0.2, 0.3, 0.4]), (73, 0.91, [-0.01, 0.5, 1.2, 0.9])]
		for tensor in (True, False):
			objs = objects_from_result(fake_result(rows, tensor), COCO_NAMES)
			self.assertEqual([o["label"] for o in objs], ["book", "cup"])
			self.assertAlmostEqual(objs[0]["score"], 0.91, places=4)
			self.assertEqual(objs[0]["box"], [0.0, 0.5, 1.0, 0.9])
			self.assertTrue(all(isinstance(v, float) for v in objs[1]["box"]))
		self.assertEqual(objects_from_result(fake_result([]), COCO_NAMES), [])
		self.assertEqual(objects_from_result(types.SimpleNamespace(boxes=None), COCO_NAMES), [])

	def test_messages(self):

		msg = json.loads(objects_message(12.34567, "caml", [{"label": "cup", "score": 0.5, "box": [0, 0, 1, 1]}], ["cup"]))
		self.assertEqual(msg["t"], 12.346)
		self.assertEqual(msg["camera"], "caml")
		self.assertEqual(msg["objects"][0]["label"], "cup")
		self.assertEqual(msg["confirmed"], ["cup"])
		msg = json.loads(gesture_message(1.0, "caml", "Victory", np.float32(0.123456)))
		self.assertEqual((msg["gesture"], msg["score"]), ("Victory", 0.1235))
		msg = json.loads(gesture_message(1.0, "caml", None, 0.7))
		self.assertEqual((msg["gesture"], msg["score"]), (None, 0.0))
		self.assertEqual(set(msg.keys()), {"t", "camera", "gesture", "score"})


class TestSettings(unittest.TestCase):

	def test_yolo_device(self):

		self.assertIsNone(yolo_device("auto"))
		self.assertIsNone(yolo_device(None))
		self.assertEqual(yolo_device("cpu"), "cpu")
		self.assertEqual(yolo_device("cuda"), "0")
		self.assertEqual(yolo_device("cuda:1"), "1")

	def test_resolve_weights(self):

		cfg = load_config(cache="/tmp/hm-cache")
		self.assertEqual(resolve_weights(cfg, "yolov10m.pt"), "/tmp/hm-cache/weights/yolov10m.pt")
		self.assertEqual(resolve_weights(cfg, "assets/models/x.pt"), os.path.join(cfg.root, "assets", "models", "x.pt"))
		self.assertEqual(resolve_weights(cfg, "/opt/w.pt"), "/opt/w.pt")

	def test_camera_intrinsics(self):

		K, D = camera_intrinsics(None)
		self.assertEqual(K.shape, (3, 3))
		self.assertEqual(D.shape, (1, 5))
		K, D = camera_intrinsics({"camera_matrix": [[500, 0, 320], [0, 500, 180], [0, 0, 1]], "dist_coeffs": [0, 0, 0, 0]})
		self.assertEqual(D.shape, (1, 4))
		with self.assertRaises(ValueError):
			camera_intrinsics({"camera_matrix": [[1, 0], [0, 1]], "dist_coeffs": [0, 0, 0, 0]})



# ---------------------------------------------------------------- the node

class FakePub(object):

	def __init__(self, topic):
		self.topic = topic
		self.sent = []

	def publish_this(self, msg):
		self.sent.append(msg)

	def json(self):
		return [json.loads(m.data) for m in self.sent]


class FakeSub(object):

	def __init__(self, topic, callback):
		self.topic = topic
		self.callback = callback
		self.unregistered = False

	def unregister(self):
		self.unregistered = True


class FakeSystem(object):

	# just what node.Node and NodeScene use of client_demo.DemoSystem
	def __init__(self, heymiro):

		self.pars = types.SimpleNamespace(forebrain=heymiro)
		self.kc_s = self.kc_m = None
		self.input = self.state = self.output = types.SimpleNamespace()
		self.nodes = types.SimpleNamespace()
		self.sub = []
		self.pubs = {}

	def subscribe(self, topic, data_type, callback, queue_size=1):

		sub = FakeSub(topic, callback)
		self.sub.append(sub)
		return sub

	def publish(self, topic, data_type):

		self.pubs[topic] = FakePub(topic)
		return self.pubs[topic]

	def live_topics(self):

		return [s.topic for s in self.sub if not s.unregistered]


class FakeCv2(object):

	# decodes our fake "JPEG" (raw bytes of a 4x6x3 image) and records calls
	IMREAD_COLOR = 1
	COLOR_BGR2RGB = 4

	def __init__(self):
		self.converted = []
		self.undistorted = 0

	def imdecode(self, buf, flags):
		if buf.size != 4 * 6 * 3:
			return None
		return buf.reshape(4, 6, 3).copy()

	def cvtColor(self, img, code):
		self.converted.append(code)
		return img[:, :, ::-1]

	def undistort(self, img, K, D):
		self.undistorted += 1
		return img


class FakeRecognizer(object):

	def __init__(self, names):
		self.names = list(names)
		self.images = []
		self.closed = False

	def recognize(self, image):
		self.images.append(image)
		name = self.names.pop(0) if self.names else None
		gestures = [[category(name, 0.9)]] if name else []
		return types.SimpleNamespace(gestures=gestures)

	def close(self):
		self.closed = True


class FakeMp(object):

	ImageFormat = types.SimpleNamespace(SRGB="srgb")

	class Image(object):
		def __init__(self, image_format, data):
			self.image_format = image_format
			self.data = data


class FakeYolo(object):

	names = COCO_NAMES

	def __init__(self, rows):
		self.rows = rows
		self.calls = []

	def predict(self, image, **kwargs):
		self.calls.append((image, kwargs))
		return [fake_result(self.rows)]


def frame_bytes(value=10):

	img = np.zeros((4, 6, 3), dtype=np.uint8)
	img[..., 0] = value      # blue
	img[..., 2] = 200        # red
	return img.tobytes()


def make_node(**overrides):

	sys_ = FakeSystem(load_config(**overrides))
	with contextlib.redirect_stdout(io.StringIO()):
		scene = NodeScene(sys_, start=False)
	return sys_, scene


def with_models(scene, gestures=(), rows=()):

	scene.cv2 = FakeCv2()
	scene.mp = FakeMp()
	scene.recognizer = FakeRecognizer(gestures)
	scene.yolo = FakeYolo(list(rows))
	scene.yolo_ids = class_ids_for(COCO_NAMES, scene.classes)[0]
	return scene


class TestNodeScene(unittest.TestCase):

	def test_topics_and_mode_subscription(self):

		sys_, scene = make_node()
		self.assertIn("core/scene/objects", sys_.pubs)
		self.assertIn("core/scene/gesture", sys_.pubs)
		self.assertEqual(sys_.live_topics(), ["core/scene/mode"])

		# no mode heard: camera stays off
		with contextlib.redirect_stdout(io.StringIO()):
			scene.tick()
		self.assertEqual(scene.mode, "off")

		# mode on: camera subscribed once
		scene.callback_mode(types.SimpleNamespace(data="gestures"))
		with contextlib.redirect_stdout(io.StringIO()):
			scene.tick()
			scene.tick()
		self.assertEqual(scene.mode, "gestures")
		self.assertEqual(sys_.live_topics(), ["core/scene/mode", "sensors/caml/compressed"])

		# frames: only the latest is kept
		cam = [s for s in sys_.sub if s.topic.startswith("sensors/")][0]
		cam.callback(types.SimpleNamespace(data=b"one"))
		cam.callback(types.SimpleNamespace(data=b"two"))
		self.assertEqual(scene.frame[1], b"two")
		self.assertTrue(scene.frame_event.is_set())

		# main goes quiet: stale -> off, camera unsubscribed, frame dropped
		scene.mode_tracker._stamp -= node_scene.MODE_STALE_S + 0.5
		with contextlib.redirect_stdout(io.StringIO()):
			scene.tick()
		self.assertEqual(scene.mode, "off")
		self.assertEqual(sys_.live_topics(), ["core/scene/mode"])
		self.assertNotIn(cam, sys_.sub)
		self.assertIsNone(scene.frame)

	def test_unknown_mode_ignored(self):

		sys_, scene = make_node()
		with contextlib.redirect_stdout(io.StringIO()):
			scene.callback_mode(types.SimpleNamespace(data="faces"))
			scene.tick()
		self.assertEqual(scene.mode, "off")

	def test_gesture_pipeline(self):

		sys_, scene = make_node()
		with_models(scene, gestures=["Victory", "Victory", "Victory", "Thumb_Up"])
		now = time.time()
		for i in range(4):
			scene.process((now, frame_bytes()), "gestures", now=now)
		msgs = sys_.pubs["core/scene/gesture"].json()
		self.assertEqual([m["gesture"] for m in msgs], [None, None, "Victory", "Victory"])
		self.assertEqual(msgs[2]["camera"], "caml")
		self.assertEqual(sys_.pubs["core/scene/objects"].sent, [])
		# mediapipe got RGB: the red channel (200) first
		image = scene.recognizer.images[0]
		self.assertEqual(image.image_format, "srgb")
		self.assertEqual(int(image.data[0, 0, 0]), 200)
		self.assertTrue(image.data.flags["C_CONTIGUOUS"])
		# YOLO is not run in gestures mode
		self.assertEqual(scene.yolo.calls, [])

	def test_object_pipeline(self):

		sys_, scene = make_node(**{"scene.device": "cpu", "identify.confirm": "[2, 3]"})
		with_models(scene, rows=[(41, 0.8, [0.1, 0.1, 0.4, 0.5]), (41, 0.6, [0.5, 0.1, 0.7, 0.5])])
		now = time.time()
		scene.process((now, frame_bytes(7)), "objects", now=now)
		scene.process((now, frame_bytes(7)), "objects", now=now)
		msgs = sys_.pubs["core/scene/objects"].json()
		self.assertEqual(len(msgs), 2)
		self.assertEqual(msgs[0]["confirmed"], [])
		self.assertEqual(msgs[1]["confirmed"], ["cup"])
		self.assertEqual([o["label"] for o in msgs[1]["objects"]], ["cup", "cup"])
		# YOLO got the BGR image (blue channel first) and the filters
		image, kwargs = scene.yolo.calls[0]
		self.assertEqual(int(image[0, 0, 0]), 7)
		self.assertEqual(kwargs["classes"], sorted(HRI_IDS))
		self.assertEqual(kwargs["device"], "cpu")
		self.assertAlmostEqual(kwargs["conf"], 0.4)
		self.assertFalse(kwargs["verbose"])
		self.assertEqual(sys_.pubs["core/scene/gesture"].sent, [])

	def test_both_and_bad_frames(self):

		sys_, scene = make_node()
		with_models(scene, gestures=["Open_Palm"], rows=[])
		now = time.time()
		scene.process((now, b"corrupt"), "both", now=now)
		self.assertEqual(scene.frames_bad, 1)
		# too old: skipped without decoding
		scene.process((now - 5.0, frame_bytes()), "both", now=now)
		self.assertEqual(scene.frames_done, 0)
		scene.process((now, frame_bytes()), "both", now=now)
		self.assertEqual(scene.frames_done, 1)
		self.assertEqual(len(sys_.pubs["core/scene/gesture"].sent), 1)
		self.assertEqual(sys_.pubs["core/scene/objects"].json()[0]["objects"], [])

	def test_missing_models_reported_once(self):

		sys_, scene = make_node(**{"scene.gesture_model": "assets/models/nope.task"})
		scene.cv2 = FakeCv2()
		out = io.StringIO()
		with contextlib.redirect_stdout(out):
			self.assertFalse(scene.ensure_gestures())
			self.assertFalse(scene.ensure_gestures())
			# ultralytics is not installed here: objects unavailable, once
			ok = scene.ensure_yolo()
			scene.ensure_yolo()
		if not ok:
			self.assertEqual(out.getvalue().count("objects unavailable"), 1)
		self.assertEqual(out.getvalue().count("gestures unavailable"), 1)
		# the frame still goes through without models and publishes nothing
		now = time.time()
		with contextlib.redirect_stdout(io.StringIO()):
			scene.process((now, frame_bytes()), "gestures", now=now)
		self.assertEqual(sys_.pubs["core/scene/gesture"].sent, [])

	def test_undistort_forced_off_in_sim(self):

		out = io.StringIO()
		with contextlib.redirect_stdout(out):
			sys_, scene = make_node(**{"scene.undistort": "true", "platform": "sim"})
		self.assertIsNone(scene.undistort)
		sys_, scene = make_node(**{"scene.undistort": "true"})
		self.assertIsNotNone(scene.undistort)
		with_models(scene, gestures=[])
		now = time.time()
		scene.process((now, frame_bytes()), "gestures", now=now)
		self.assertEqual(scene.cv2.undistorted, 1)

	def test_thread_runs_and_shuts_down(self):

		sys_ = FakeSystem(load_config(**{"scene.rate_hz": "50"}))
		with contextlib.redirect_stdout(io.StringIO()):
			scene = NodeScene(sys_)
		with_models(scene, gestures=["Victory"] * 10)
		try:
			scene.callback_mode(types.SimpleNamespace(data="gestures"))
			with contextlib.redirect_stdout(io.StringIO()):
				scene.tick()
			cam = [s for s in sys_.sub if s.topic.startswith("sensors/")][0]
			deadline = time.time() + 5.0
			while len(sys_.pubs["core/scene/gesture"].sent) < 3 and time.time() < deadline:
				cam.callback(types.SimpleNamespace(data=frame_bytes()))
				time.sleep(0.03)
			self.assertGreaterEqual(len(sys_.pubs["core/scene/gesture"].sent), 3)
		finally:
			recognizer = scene.recognizer
			with contextlib.redirect_stdout(io.StringIO()):
				scene.shutdown()
		self.assertFalse(scene.thread.is_alive())
		self.assertTrue(recognizer.closed)

	@unittest.skipIf(cv2 is None, "OpenCV not installed")
	def test_real_jpeg_decode(self):

		sys_, scene = make_node()
		self.assertTrue(scene.ensure_cv2())
		img = np.zeros((36, 64, 3), dtype=np.uint8)
		img[..., 2] = 255
		ok, jpeg = cv2.imencode(".jpg", img)
		self.assertTrue(ok)
		bgr = scene.decode(jpeg.tobytes())
		self.assertEqual(bgr.shape, (36, 64, 3))
		self.assertGreater(int(bgr[0, 0, 2]), 200)



# --------------------------------------------------------------- make_clip

class FakeTTS(object):

	def __init__(self, seconds=0.5):
		self.calls = []
		self.seconds = seconds

	def synthesize(self, text, voice_id=None):
		self.calls.append((text, voice_id))
		n = int(24000 * self.seconds)
		return (np.sin(np.arange(n) * 0.05) * 20000).astype(np.int16)


class FailingTTS(object):

	def synthesize(self, text, voice_id=None):
		from cloud.worker import HttpError
		raise HttpError("tts", 401)


class TestMakeClip(unittest.TestCase):

	def setUp(self):

		self.tmp = tempfile.mkdtemp()

	def tearDown(self):

		shutil.rmtree(self.tmp)

	def run_main(self, argv, tts, environ=None):

		heymiro = heymiro_config.load(environ=environ or {})
		out = io.StringIO()
		with contextlib.redirect_stdout(out):
			code = make_clip.main(argv + ["--out-dir", self.tmp], heymiro=heymiro, tts=tts)
		return code, out.getvalue()

	def test_slugify(self):

		self.assertEqual(make_clip.slugify("Don't touch my ears, please!"), "dont_touch_my_ears_please")
		self.assertEqual(make_clip.slugify("Adiós amigos"), "adios_amigos")
		self.assertEqual(make_clip.slugify("!!!"), "clip")
		self.assertLessEqual(len(make_clip.slugify("word " * 40)), make_clip.SLUG_MAX)

	def test_manifest_line(self):

		manifest = {"greeting": ["hi", "hey"]}
		self.assertEqual(make_clip.manifest_line(manifest, "greeting", "hello"), "greeting: [hi, hey, hello]")
		self.assertEqual(make_clip.manifest_line(manifest, "greeting", "hi"), "greeting: [hi, hey]")
		self.assertEqual(make_clip.manifest_line(manifest, "new", "x"), "new: [x]")

	def test_makes_numbered_takes(self):

		tts = FakeTTS()
		code, out = self.run_main(["Hello", "pal!", "--count", "2", "--category", "greeting", "--voice-id", "v1"], tts)
		self.assertEqual(code, 0)
		folder = os.path.join(self.tmp, "hello_pal")
		self.assertEqual(sorted(os.listdir(folder)), ["0.wav", "1.wav"])
		self.assertEqual(tts.calls, [("Hello pal!", "v1"), ("Hello pal!", "v1")])
		with wave.open(os.path.join(folder, "1.wav"), "rb") as w:
			self.assertEqual((w.getframerate(), w.getnchannels(), w.getsampwidth()), (24000, 1, 2))
			self.assertEqual(w.getnframes(), 12000)
		self.assertIn("greeting: [hmm_inquisitive, ", out)
		self.assertIn(", hello_pal]", out)
		# only TTS / secrets-file warnings are relevant here
		self.assertNotIn("OPENAI_API_KEY", out)

		# numbering continues; default voice from the config; custom slug
		open(os.path.join(folder, "notes.wav"), "wb").close()
		code, out = self.run_main(["Hello again", "--slug", "Hello Pal"], tts)
		self.assertEqual(code, 0)
		self.assertIn("2.wav", os.listdir(folder))
		self.assertEqual(tts.calls[-1][1], heymiro_config.load(environ={}).cfg.tts.voice_id)

	def test_http_error_is_sanitised(self):

		code, out = self.run_main(["Ouch!"], FailingTTS())
		self.assertEqual(code, 1)
		self.assertIn("HttpError 401", out)

	def test_no_key_and_no_secret_printed(self):

		home = tempfile.mkdtemp()
		try:
			# no key anywhere (an empty HOME hides any real secrets file):
			# refuses before any request
			with mock.patch.dict(os.environ, {"HOME": home}):
				code, out = self.run_main(["Ouch!"], None, environ={"HOME": home})
			self.assertEqual(code, 2)
			self.assertIn("ELEVENLABS_API_KEY", out)

			# with a key: the request fails offline (fake session), key never shown
			from cloud import tts as tts_module
			heymiro = heymiro_config.load(environ={"ELEVENLABS_API_KEY": SECRET})

			class Session(object):
				def post(self, url, **kwargs):
					return types.SimpleNamespace(status_code=401, content=b"bad key " + SECRET.encode(), headers={})

			engine = tts_module.TextToSpeech(heymiro.route("tts"), heymiro.cfg.tts, session=Session())
			out = io.StringIO()
			with contextlib.redirect_stdout(out):
				code = make_clip.main(["Ouch!", "--out-dir", self.tmp], heymiro=heymiro, tts=engine)
			self.assertEqual(code, 1)
			self.assertNotIn(SECRET, out.getvalue())
		finally:
			shutil.rmtree(home)

	def test_bad_count(self):

		with contextlib.redirect_stderr(io.StringIO()):
			with self.assertRaises(SystemExit):
				make_clip.parse_args(["hi", "--count", "0"])



if __name__ == "__main__":
	unittest.main()
