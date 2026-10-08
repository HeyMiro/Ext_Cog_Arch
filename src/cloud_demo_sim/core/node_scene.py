#
#	Hey MiRo - NodeScene: scene awareness (the "- scene" client process).
#
#	Runs the two camera models of HRI'25 node_vision.py in a process of
#	their own, so torch/YOLO and mediapipe never compete with the 50 Hz
#	main loop for the GIL:
#
#	  gestures  mediapipe GestureRecognizer (IMAGE mode)    -> core/scene/gesture
#	  objects   ultralytics YOLO, only identify.classes      -> core/scene/objects
#
#	The main client says what to look for on core/scene/mode (off |
#	gestures | objects | both) and re-sends it every second; a mode not
#	heard for MODE_STALE_S counts as "off", so the camera and the models
#	go quiet when main stops. The camera is only subscribed while the
#	mode is not off, because a second copy of the camera stream over
#	WiFi is expensive.
#
#	Threads: ROS callbacks only store the latest message; tick() (10 Hz,
#	from the client loop) applies the mode and (un)subscribes the camera;
#	one inference thread decodes the latest JPEG and runs the models at
#	scene.rate_hz. The models are loaded lazily in that thread on first
#	use; if one fails to load it is reported once and the other keeps
#	going.
#
#	Changes from HRI'25: frames are decoded with cv2.imdecode (BGR, what
#	YOLO expects for numpy input) and converted to RGB for mediapipe only;
#	YOLO class ids are looked up by name in the model's own names table
#	(HRI hardcoded COCO ids); model paths come from the config (HRI used
#	the working directory); undistortion is optional and forced off in
#	the simulator; gestures are confirmed k-of-n (HRI used a history of
#	one frame). DeepFace, defisheye and AprilTag are dropped: facing the
#	user uses the stock camera clients.
#
#	Messages (std_msgs/String holding JSON):
#	  core/scene/objects  {"t", "camera", "objects": [{"label", "score", "box"}],
#	                       "confirmed": [label, ...]}
#	  core/scene/gesture  {"t", "camera", "gesture" (str or null), "score"}
#	t = wall-clock time the frame arrived; box = [x0, y0, x1, y1]
#	normalised to 0..1; confirmed = labels seen in k of the last n frames
#	(identify.confirm, HRI used 4 of 10).
#

import collections
import json
import os
import threading
import time

import numpy as np

import std_msgs.msg
import sensor_msgs.msg

import node


__all__ = ["NodeScene"]

MODES = ("off", "gestures", "objects", "both")

# main publishes the mode every second; three missed means main is gone
MODE_STALE_S = 3.0

# a frame older than this is not worth processing (camera stalled)
FRAME_MAX_AGE_S = 1.0

# mediapipe labels reported as "no gesture": "None" is mediapipe's own
# no-gesture class, and HRI'25 also ignored the fist and both thumbs
# (none of them starts a trick)
IGNORED_GESTURES = ("None", "Closed_Fist", "Thumb_Up", "Thumb_Down")

# a gesture must be seen in 3 of the last 5 frames to be reported
GESTURE_CONFIRM = (3, 5)

# HRI'25 calibration of MiRo's left camera (640 x 360), used when
# scene.undistort is true but no scene.intrinsics are configured
HRI_CAMERA_MATRIX = [
	[1.04358065e+03, 0.0, 3.29969935e+02],
	[0.0, 1.03845278e+03, 1.68243114e+02],
	[0.0, 0.0, 1.0]]
HRI_DIST_COEFFS = [-3.63299415e+00, 1.52661324e+01, -7.23780207e-03, -7.48630198e-04, -3.20700124e+01]



################ pure helpers (unit tested without ROS / models) ################

def mode_wants(mode):

	# (run gestures, run objects) for a scene mode
	return mode in ("gestures", "both"), mode in ("objects", "both")


class ModeTracker(object):

	"""
	The scene mode last heard from main, falling back to "off" once it
	has not been heard for stale_s seconds. Thread-safe: update() runs
	in the ROS callback, current() in tick().
	"""

	def __init__(self, stale_s=MODE_STALE_S):

		self.stale_s = float(stale_s)
		self._lock = threading.Lock()
		self._mode = "off"
		self._stamp = None

	def update(self, mode, now):

		# returns False (and keeps the old mode) for an unknown mode
		mode = str(mode or "").strip().lower()
		if mode not in MODES:
			return False
		with self._lock:
			self._mode = mode
			self._stamp = now
		return True

	def current(self, now):

		with self._lock:
			if self._stamp is None or now - self._stamp > self.stale_s:
				return "off"
			return self._mode


class KofN(object):

	"""
	k-of-n confirmation: a label is confirmed while it was present in
	at least k of the last n frames. update() takes the labels of one
	frame as {label: score} and returns the confirmed labels as a list
	of (label, mean score over the frames it was seen in), most often
	seen first.
	"""

	def __init__(self, k, n):

		self.n = max(1, int(n))
		self.k = min(max(1, int(k)), self.n)
		self.history = collections.deque(maxlen=self.n)

	def reset(self):

		self.history.clear()

	def update(self, scores):

		self.history.append(dict(scores or {}))
		counts = {}
		sums = {}
		for frame in self.history:
			for label, score in frame.items():
				counts[label] = counts.get(label, 0) + 1
				sums[label] = sums.get(label, 0.0) + float(score)
		confirmed = [(label, sums[label] / counts[label]) for label in counts if counts[label] >= self.k]
		confirmed.sort(key=lambda c: (-counts[c[0]], -c[1], str(c[0])))
		return confirmed


class GestureSmoother(object):

	"""
	Turns per-frame (gesture, score) into a stable gesture: ignored
	labels count as "no gesture", and a gesture is only reported while
	it is confirmed k-of-n. Returns (gesture or None, score).
	"""

	def __init__(self, k=GESTURE_CONFIRM[0], n=GESTURE_CONFIRM[1], ignore=IGNORED_GESTURES):

		self.ignore = set(ignore)
		self.kofn = KofN(k, n)

	def reset(self):

		self.kofn.reset()

	def update(self, gesture, score):

		frame = {}
		if gesture and gesture not in self.ignore:
			frame[gesture] = score
		confirmed = self.kofn.update(frame)
		if not confirmed:
			return None, 0.0
		return confirmed[0]


def top_gesture(result, ignore=IGNORED_GESTURES):

	# best (category_name, score) over the hands in a mediapipe
	# GestureRecognizerResult, skipping ignored labels; (None, 0.0) if none
	best = (None, 0.0)
	for hand in (getattr(result, "gestures", None) or []):
		if not hand:
			continue
		top = hand[0]
		name = getattr(top, "category_name", None)
		score = float(getattr(top, "score", 0.0) or 0.0)
		if not name or name in ignore:
			continue
		if best[0] is None or score > best[1]:
			best = (name, score)
	return best


def _names_items(names):

	# YOLO model.names is a dict {id: name} (a list in some versions)
	if isinstance(names, dict):
		return [(int(i), str(n)) for i, n in names.items()]
	return [(i, str(n)) for i, n in enumerate(names or [])]


def class_ids_for(names, wanted):

	"""
	Class ids for the wanted class names (case-insensitive), looked up in
	the model's own names table, so the configured list keeps working
	with any YOLO weights. Returns (sorted ids, names not found).
	"""

	index = {}
	for i, n in _names_items(names):
		index.setdefault(n.strip().lower(), i)
	ids = set()
	missing = []
	for w in (wanted or []):
		key = str(w).strip().lower()
		if key in index:
			ids.add(index[key])
		else:
			missing.append(str(w))
	return sorted(ids), missing


def label_for(names, class_id):

	class_id = int(class_id)
	if isinstance(names, dict):
		return str(names.get(class_id, class_id))
	if names is not None and 0 <= class_id < len(names):
		return str(names[class_id])
	return str(class_id)


def to_numpy(x):

	# torch tensor (possibly on the GPU) or array-like -> numpy array
	if hasattr(x, "cpu"):
		x = x.cpu()
	if hasattr(x, "numpy"):
		x = x.numpy()
	return np.asarray(x)


def objects_from_result(result, names):

	# one ultralytics Results -> list of {"label", "score", "box"}
	boxes = getattr(result, "boxes", None)
	if boxes is None:
		return []
	cls = to_numpy(boxes.cls).reshape(-1)
	conf = to_numpy(boxes.conf).reshape(-1)
	xyxyn = to_numpy(boxes.xyxyn).reshape(-1, 4)
	objects = []
	for c, s, b in zip(cls, conf, xyxyn):
		box = [round(float(min(max(v, 0.0), 1.0)), 4) for v in b]
		objects.append({"label": label_for(names, c), "score": round(float(s), 4), "box": box})
	objects.sort(key=lambda o: -o["score"])
	return objects


def best_scores(objects):

	# {label: best score} over the detections of one frame
	out = {}
	for o in objects:
		if o["score"] > out.get(o["label"], -1.0):
			out[o["label"]] = o["score"]
	return out


def objects_message(t, camera, objects, confirmed):

	return json.dumps({
		"t": round(float(t), 3),
		"camera": str(camera),
		"objects": list(objects),
		"confirmed": [str(label) for label in confirmed],
	})


def gesture_message(t, camera, gesture, score):

	return json.dumps({
		"t": round(float(t), 3),
		"camera": str(camera),
		"gesture": gesture if gesture else None,
		"score": round(float(score), 4) if gesture else 0.0,
	})


def yolo_device(setting):

	# scene.device -> ultralytics device argument (None lets it choose)
	setting = str(setting or "auto").strip().lower()
	if setting in ("", "auto", "none"):
		return None
	if setting.startswith("cuda") or setting == "gpu":
		index = setting.split(":", 1)[1] if ":" in setting else "0"
		return index or "0"
	return setting


def resolve_weights(heymiro, name):

	# a bare file name lives in the download cache; a path (relative to
	# the project root, ~ or absolute) is used as given
	name = str(name)
	if os.path.isabs(os.path.expanduser(name)) or "/" in name or os.sep in name:
		return heymiro.path(name)
	return heymiro.cache_path("weights", name)


def camera_intrinsics(intrinsics):

	"""
	(camera_matrix 3x3, dist_coeffs 1xN) float64 arrays from a
	{camera_matrix, dist_coeffs} dict, or the HRI'25 calibration when
	None. Raises ValueError if the shapes are wrong.
	"""

	if intrinsics is None:
		intrinsics = {"camera_matrix": HRI_CAMERA_MATRIX, "dist_coeffs": HRI_DIST_COEFFS}
	if hasattr(intrinsics, "to_dict"):
		intrinsics = intrinsics.to_dict()
	K = np.asarray(intrinsics.get("camera_matrix"), dtype=np.float64)
	D = np.asarray(intrinsics.get("dist_coeffs"), dtype=np.float64).reshape(1, -1)
	if K.shape != (3, 3) or D.shape[1] not in (4, 5, 8, 12, 14):
		raise ValueError("scene.intrinsics needs a 3x3 camera_matrix and 4/5/8/12/14 dist_coeffs")
	return K, D



################ node ################

class NodeScene(node.Node):

	def __init__(self, sys, start=True):

		node.Node.__init__(self, sys, "scene")

		# the DemoSystem, for its subscribe / publish helpers
		self.sys = sys

		heymiro = self.pars.forebrain
		cfg = heymiro.cfg
		self.camera = str(cfg.get("scene.camera", "caml"))
		self.rate_hz = max(0.2, float(cfg.get("scene.rate_hz", 5.0)))
		self.yolo_conf = float(cfg.get("scene.yolo_conf", 0.4))
		self.device = yolo_device(cfg.get("scene.device", "auto"))
		self.weights = resolve_weights(heymiro, cfg.get("scene.yolo_weights", "yolov10m.pt"))
		self.gesture_model = heymiro.path(cfg.get("scene.gesture_model", "assets/models/gesture_recognizer.task"))
		self.classes = [str(c) for c in (cfg.get("identify.classes") or [])]
		confirm = cfg.get("identify.confirm") or [4, 10]
		self.objects_confirm = (int(confirm[0]), int(confirm[1]))

		# undistortion: off by default, and always off in the simulator
		# (its cameras have no lens distortion to remove)
		self.undistort = None
		if cfg.get("scene.undistort", False):
			if str(cfg.get("platform", "robot")).strip().lower() == "sim":
				print("[scene] undistort forced off in the simulator")
			else:
				try:
					self.undistort = camera_intrinsics(cfg.get("scene.intrinsics"))
				except (ValueError, TypeError, AttributeError) as e:
					print("[scene] undistort disabled: " + str(e))

		# mode (ROS callback -> tick) and effective mode (tick -> thread);
		# mode_gen tells the inference thread to restart its smoothing
		self.mode_tracker = ModeTracker()
		self.mode = "off"
		self.mode_gen = 0

		# latest camera frame: (wall time received, JPEG bytes)
		self.frame_lock = threading.Lock()
		self.frame = None
		self.frame_event = threading.Event()
		self.cam_sub = None

		# models, loaded on first use by the inference thread
		self.cv2 = None
		self.yolo = None
		self.yolo_ids = None
		self.mp = None
		self.recognizer = None
		self.failed = set()

		# smoothing (only touched by the inference thread)
		self.gesture_smoother = GestureSmoother()
		self.objects_kofn = KofN(self.objects_confirm[0], self.objects_confirm[1])

		# counters (for the shutdown report)
		self.frames_done = 0
		self.frames_bad = 0
		self.errors = 0

		# ROS interfaces
		self.pub_objects = sys.publish("core/scene/objects", std_msgs.msg.String)
		self.pub_gesture = sys.publish("core/scene/gesture", std_msgs.msg.String)
		self.mode_sub = sys.subscribe("core/scene/mode", std_msgs.msg.String, self.callback_mode)

		print("[scene] camera " + self.camera + " at " + str(self.rate_hz) + " Hz; objects confirmed " +
			str(self.objects_confirm[0]) + " of " + str(self.objects_confirm[1]) +
			("; undistort on" if self.undistort is not None else ""))

		# inference thread
		self.stop_event = threading.Event()
		self.thread = None
		if start:
			self.thread = threading.Thread(target=self.run, name="scene_inference")
			self.thread.daemon = True
			self.thread.start()

	# ---------------------------------------------------------- ROS callbacks

	def callback_mode(self, msg):

		if not self.mode_tracker.update(msg.data, time.monotonic()):
			if "mode:" + str(msg.data) not in self.failed:
				self.failed.add("mode:" + str(msg.data))
				print("[scene] ignoring unknown mode " + repr(str(msg.data)[:20]))

	def callback_cam(self, msg):

		# store only; decoding happens in the inference thread
		with self.frame_lock:
			self.frame = (time.time(), msg.data)
		self.frame_event.set()

	# ------------------------------------------------------------------- tick

	def tick(self):

		# 10 Hz from the client loop: apply the (possibly stale) mode
		mode = self.mode_tracker.current(time.monotonic())
		if mode == self.mode:
			return
		print("[scene] mode " + self.mode + " -> " + mode)
		self.mode = mode
		self.mode_gen += 1
		if mode == "off":
			self.unsubscribe_camera()
		else:
			self.subscribe_camera()

	def subscribe_camera(self):

		if self.cam_sub is None:
			topic = "sensors/" + self.camera + "/compressed"
			self.cam_sub = self.sys.subscribe(topic, sensor_msgs.msg.CompressedImage, self.callback_cam)

	def unsubscribe_camera(self):

		if self.cam_sub is not None:
			sub = self.cam_sub
			self.cam_sub = None
			sub.unregister()
			subs = getattr(self.sys, "sub", None)
			if isinstance(subs, list) and sub in subs:
				subs.remove(sub)
		# forget the last frame so a later mode never sees an old picture
		with self.frame_lock:
			self.frame = None
		self.frame_event.clear()

	# ------------------------------------------------------- inference thread

	def run(self):

		period = 1.0 / self.rate_hz
		gen = None
		while not self.stop_event.is_set():

			if self.mode == "off":
				self.stop_event.wait(0.2)
				continue

			# wait for a new frame
			if not self.frame_event.wait(0.5):
				continue
			self.frame_event.clear()
			with self.frame_lock:
				frame = self.frame
			mode = self.mode
			if frame is None or mode == "off":
				continue

			# a new mode starts a new game: forget earlier detections
			if gen != self.mode_gen:
				gen = self.mode_gen
				self.gesture_smoother.reset()
				self.objects_kofn.reset()

			t0 = time.monotonic()
			try:
				self.process(frame, mode)
			except Exception as e:
				self.errors += 1
				if self.errors <= 5 or self.errors % 100 == 0:
					print("[scene] inference error (" + str(self.errors) + "): " + e.__class__.__name__ + " " + str(e)[:120])

			# pace to rate_hz (the latest frame is picked up next time)
			rest = period - (time.monotonic() - t0)
			if rest > 0.0:
				self.stop_event.wait(rest)

	def process(self, frame, mode, now=None):

		# one frame through the models the mode asks for
		stamp, data = frame
		now = time.time() if now is None else now
		if now - stamp > FRAME_MAX_AGE_S:
			return
		want_gestures, want_objects = mode_wants(mode)
		if not self.ensure_cv2():
			return
		bgr = self.decode(data)
		if bgr is None:
			self.frames_bad += 1
			return
		self.frames_done += 1

		if want_gestures and self.ensure_gestures():
			name, score = self.detect_gesture(bgr)
			gesture, score = self.gesture_smoother.update(name, score)
			self.publish(self.pub_gesture, gesture_message(stamp, self.camera, gesture, score))

		if want_objects and self.ensure_yolo():
			objects = self.detect_objects(bgr)
			confirmed = self.objects_kofn.update(best_scores(objects))
			self.publish(self.pub_objects, objects_message(stamp, self.camera, objects, [c[0] for c in confirmed]))

	def publish(self, pub, text):

		pub.publish_this(std_msgs.msg.String(data=text))

	def decode(self, data):

		# JPEG bytes -> BGR image (None if corrupt), undistorted if enabled
		cv2 = self.cv2
		buf = np.frombuffer(bytes(data), dtype=np.uint8)
		if buf.size == 0:
			return None
		bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
		if bgr is None:
			return None
		if self.undistort is not None:
			bgr = cv2.undistort(bgr, self.undistort[0], self.undistort[1])
		return bgr

	def detect_gesture(self, bgr):

		# mediapipe wants RGB in a contiguous uint8 buffer
		rgb = np.ascontiguousarray(self.cv2.cvtColor(bgr, self.cv2.COLOR_BGR2RGB))
		image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
		return top_gesture(self.recognizer.recognize(image))

	def detect_objects(self, bgr):

		# YOLO takes the BGR numpy image directly
		kwargs = {"verbose": False, "conf": self.yolo_conf}
		if self.yolo_ids is not None:
			kwargs["classes"] = self.yolo_ids
		if self.device is not None:
			kwargs["device"] = self.device
		objects = []
		for result in self.yolo.predict(bgr, **kwargs):
			objects.extend(objects_from_result(result, self.yolo.names))
		objects.sort(key=lambda o: -o["score"])
		return objects

	# --------------------------------------------------------- model loading

	def report_failure(self, what, reason):

		# print once per model, then carry on without it
		if what not in self.failed:
			self.failed.add(what)
			print("[scene] " + what + " unavailable: " + reason)

	def ensure_cv2(self):

		if self.cv2 is None and "cv2" not in self.failed:
			try:
				import cv2
				self.cv2 = cv2
			except ImportError as e:
				self.report_failure("cv2", "cannot import OpenCV (" + e.__class__.__name__ + ")")
		return self.cv2 is not None

	def ensure_yolo(self):

		if self.yolo is not None:
			return True
		if "objects" in self.failed:
			return False
		try:
			from ultralytics import YOLO
			folder = os.path.dirname(self.weights)
			if folder and not os.path.isdir(folder):
				os.makedirs(folder)
			if not os.path.isfile(self.weights):
				print("[scene] downloading YOLO weights to " + self.weights + " (first use) ...")
			model = YOLO(self.weights)
			ids, missing = class_ids_for(model.names, self.classes)
			if missing:
				print("[scene] WARN classes not known to " + os.path.basename(self.weights) + ": " + ", ".join(missing))
			if self.classes and not ids:
				self.report_failure("objects", "none of identify.classes are known to the model")
				return False
			# an empty identify.classes list means every class
			self.yolo_ids = ids if self.classes else None
			self.yolo = model
			print("[scene] YOLO ready: " + os.path.basename(self.weights) + ", " +
				(str(len(ids)) + " classes" if self.yolo_ids is not None else "all classes"))
		except Exception as e:
			self.report_failure("objects", "YOLO did not load (" + e.__class__.__name__ + " " + str(e)[:120] + ")")
		return self.yolo is not None

	def ensure_gestures(self):

		if self.recognizer is not None:
			return True
		if "gestures" in self.failed:
			return False
		if not os.path.isfile(self.gesture_model):
			self.report_failure("gestures", "model file missing: " + self.gesture_model)
			return False
		try:
			import mediapipe as mp
			from mediapipe.tasks import python as mp_tasks
			from mediapipe.tasks.python import vision
			options = vision.GestureRecognizerOptions(
				base_options=mp_tasks.BaseOptions(model_asset_path=self.gesture_model),
				running_mode=vision.RunningMode.IMAGE)
			self.recognizer = vision.GestureRecognizer.create_from_options(options)
			self.mp = mp
			print("[scene] gesture recognizer ready")
		except Exception as e:
			self.report_failure("gestures", "mediapipe did not load (" + e.__class__.__name__ + " " + str(e)[:120] + ")")
		return self.recognizer is not None

	# --------------------------------------------------------------- shutdown

	def shutdown(self):

		self.stop_event.set()
		self.frame_event.set()
		stopped = True
		if self.thread is not None:
			self.thread.join(2.0)
			stopped = not self.thread.is_alive()
		self.unsubscribe_camera()
		# never close the recognizer under a running inference
		if stopped and self.recognizer is not None:
			try:
				self.recognizer.close()
			except Exception:
				pass
			self.recognizer = None
		print("[scene] frames processed " + str(self.frames_done) + ", undecodable " +
			str(self.frames_bad) + ", errors " + str(self.errors))
