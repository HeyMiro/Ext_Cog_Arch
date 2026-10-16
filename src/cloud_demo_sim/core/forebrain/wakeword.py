#
#	Hey MiRo - wake words (Picovoice Porcupine) and voice activity
#	detection (Picovoice Cobra, or a key-free energy detector).
#
#	HRI'25 created one Porcupine instance per keyword, each with its own
#	access key, plus an unused Koala instance. Here keywords that share an
#	access key share one Porcupine instance (custom .ppn files are tied to
#	the Picovoice account that trained them, so different keys may still
#	be needed). Access keys are never printed or put in exception text.
#
#	All engines take 16 kHz int16 frames of 512 samples
#	(audio_util.HEARING_RATE / HEARING_FRAME).
#
#	Local pipeline (no keys): SileroVad (a small neural VAD, CPU) and
#	AsrKeywordSpotter, which sends each short voiced segment to the local
#	speech-to-text service and matches the text against the wake phrases
#	("hey miro", "dance miro", "stop miro"), tolerating the usual
#	misrecognitions of "MiRo" (mirror, milo, meero, ...).
#

import collections
import difflib
import importlib.util
import os
import re
import threading
import time

import numpy as np

import audio_util



def _scrub(text, secret):

	# exception text from the SDK, with the key masked (just in case)
	text = str(text)
	if secret:
		text = text.replace(secret, "***")
	return text[:160]



class KeywordSpotter(object):

	"""
	keywords: list of dicts {id, path, sensitivity, access_key}.
	process(frame) -> list of keyword ids detected in this frame.
	`create` may be injected for tests (defaults to pvporcupine.create).
	"""

	def __init__(self, keywords, frame_length=audio_util.HEARING_FRAME, create=None):

		if create is None:
			try:
				import pvporcupine
			except Exception as e:
				raise RuntimeError("pvporcupine not available (" + e.__class__.__name__ + ")")
			create = pvporcupine.create

		# group usable keywords by access key, keeping the configured order
		groups = []
		by_key = {}
		for kw in keywords:
			kid = kw.get("id")
			path = kw.get("path")
			key = kw.get("access_key")
			if not path or not os.path.isfile(path):
				print("[wakeword] WARN keyword '" + str(kid) + "' skipped: file not found: " + str(path))
				continue
			if not key:
				print("[wakeword] WARN keyword '" + str(kid) + "' skipped: no Picovoice access key")
				continue
			if key not in by_key:
				by_key[key] = {"key": key, "ids": [], "paths": [], "sens": []}
				groups.append(by_key[key])
			g = by_key[key]
			g["ids"].append(kid)
			g["paths"].append(path)
			g["sens"].append(float(kw.get("sensitivity", 0.5)))
		if not groups:
			raise RuntimeError("no usable wake words")

		# one engine per access key
		self._engines = []
		self.ids = []
		try:
			for g in groups:
				try:
					handle = create(access_key=g["key"], keyword_paths=g["paths"], sensitivities=g["sens"])
				except Exception as e:
					raise RuntimeError("Porcupine create failed for " + ", ".join(g["ids"]) + ": " +
						e.__class__.__name__ + " " + _scrub(e, g["key"]))
				self._engines.append((handle, g["ids"]))
				self.ids += g["ids"]
				if getattr(handle, "frame_length", frame_length) != frame_length:
					raise RuntimeError("Porcupine frame length is not " + str(frame_length))
		except RuntimeError:
			self.delete()
			raise
		self.frame_length = frame_length

	def process(self, frame):

		# Porcupine wants a sequence of ints; tolist() of 512 samples is cheap
		pcm = audio_util.to_int16(frame).tolist()
		found = []
		for handle, ids in self._engines:
			index = handle.process(pcm)
			if index is not None and 0 <= index < len(ids):
				found.append(ids[index])
		return found

	def delete(self):

		for handle, ids in self._engines:
			try:
				handle.delete()
			except Exception:
				pass
		self._engines = []



class EnergyVad(object):

	"""
	Key-free VAD: frame level (dBFS) mapped through a soft step centred on
	threshold_dbfs, then smoothed so single clicks do not count as speech.
	Good enough with a close mic in a quiet room; Cobra is much better in
	a crowd.
	"""

	kind = "energy"

	def __init__(self, threshold_dbfs=-38.0, width_db=3.0, smooth=0.5):

		self.threshold_dbfs = float(threshold_dbfs)
		self.width_db = float(width_db)
		self.smooth = float(smooth)
		self.prob = 0.0

	def process(self, frame):

		db = audio_util.level_dbfs(frame)
		z = (db - self.threshold_dbfs) / self.width_db
		raw = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
		self.prob = self.smooth * self.prob + (1.0 - self.smooth) * raw
		return float(self.prob)

	def delete(self):

		pass



class CobraVad(object):

	kind = "cobra"

	def __init__(self, access_key, create=None):

		if create is None:
			try:
				import pvcobra
			except Exception as e:
				raise RuntimeError("pvcobra not available (" + e.__class__.__name__ + ")")
			create = pvcobra.create
		if not access_key:
			raise RuntimeError("no Picovoice access key")
		try:
			self._handle = create(access_key=access_key)
		except Exception as e:
			raise RuntimeError("Cobra create failed: " + e.__class__.__name__ + " " + _scrub(e, access_key))
		if getattr(self._handle, "frame_length", audio_util.HEARING_FRAME) != audio_util.HEARING_FRAME:
			self.delete()
			raise RuntimeError("Cobra frame length is not " + str(audio_util.HEARING_FRAME))

	def process(self, frame):

		# HRI changed cobra._frame_length to feed whole recordings; Cobra only
		# accepts its own frame length, so we always pass 512-sample frames
		return float(self._handle.process(audio_util.to_int16(frame).tolist()))

	def delete(self):

		handle = getattr(self, "_handle", None)
		self._handle = None
		if handle is not None:
			try:
				handle.delete()
			except Exception:
				pass



class SileroVad(object):

	"""
	Silero VAD v5 (MIT), TorchScript, on the CPU: one 512-sample 16 kHz
	frame in, a speech probability out (it keeps a little state across
	frames). The model file ships inside the silero-vad pip package; it is
	loaded directly, so torchaudio (which that package imports) is not
	needed. `model` may be injected for tests.
	"""

	kind = "silero"

	def __init__(self, model=None, path=None):

		if model is None:
			try:
				import torch
			except Exception as e:
				raise RuntimeError("torch not available (" + e.__class__.__name__ + ")")
			path = path or silero_model_path()
			if not path or not os.path.isfile(path):
				raise RuntimeError("Silero VAD model not found (pip install --no-deps silero-vad)")
			torch.set_num_threads(1)
			model = torch.jit.load(path, map_location="cpu")
			model.eval()
			self._torch = torch
		else:
			self._torch = None
		self._model = model

	def process(self, frame):

		x = audio_util.to_int16(frame).astype(np.float32) / 32768.0
		if self._torch is not None:
			with self._torch.no_grad():
				return float(self._model(self._torch.from_numpy(x), audio_util.HEARING_RATE).item())
		return float(self._model(x, audio_util.HEARING_RATE))

	def reset(self):

		if hasattr(self._model, "reset_states"):
			self._model.reset_states()

	def delete(self):

		self._model = None


def silero_model_path():

	# <site-packages>/silero_vad/data/silero_vad.jit, found without
	# importing the package (its __init__ imports torchaudio)
	try:
		spec = importlib.util.find_spec("silero_vad")
	except Exception:
		return None
	if spec is None or not spec.submodule_search_locations:
		return None
	return os.path.join(list(spec.submodule_search_locations)[0], "data", "silero_vad.jit")



def make_vad(kind, access_key, threshold_dbfs=-38.0):

	# Cobra / Silero when asked for and possible, otherwise the energy VAD
	if kind == "cobra":
		try:
			return CobraVad(access_key)
		except RuntimeError as e:
			print("[wakeword] Cobra VAD unavailable (" + str(e) + "); using the energy VAD")
	if kind == "silero":
		try:
			return SileroVad()
		except RuntimeError as e:
			print("[wakeword] Silero VAD unavailable (" + str(e) + "); using the energy VAD")
	return EnergyVad(threshold_dbfs)



# ------------------------------------------------------- wake words by ASR

# what Whisper writes for "MiRo" (besides miro itself)
MIRO_VARIANTS = ("miro", "mirro", "miroh", "meero", "mero", "mirow", "myro", "mirror", "milo",
	"mira", "nero", "miru", "mi ro", "mieroo")

# the first word(s) of each wake phrase, by keyword id
WAKE_LEADS = {
	"hey_miro": ("hey", "hi", "hello", "hay", "okay", "ok", "heya", "hiya"),
	"dance_miro": ("dance", "dancing", "dans"),
	"stop_miro": ("stop", "stopp"),
}

# biases Whisper towards writing the name the way we match it
WAKE_PROMPT = "Hey MiRo! Dance MiRo! Stop MiRo!"

WORD_RE = re.compile(r"[a-z]+")


def _is_miro(word):

	if word in MIRO_VARIANTS:
		return True
	return difflib.SequenceMatcher(None, word, "miro").ratio() >= 0.75


def match_wake_phrase(text, ids=None, window=3):

	"""
	Keyword ids whose phrase is in text: a lead word ("hey", "dance",
	"stop") followed within `window` words by MiRo (or MiRo then the lead
	word: "MiRo, stop!"). ids limits which phrases are looked for.
	"""

	words = WORD_RE.findall(str(text or "").lower().replace("mi ro", "miro"))
	found = []
	for kid, leads in WAKE_LEADS.items():
		if ids is not None and kid not in ids:
			continue
		for i, w in enumerate(words):
			if w not in leads:
				continue
			near = words[max(0, i - 1):i] + words[i + 1:i + 1 + window]
			if any(_is_miro(x) for x in near):
				found.append(kid)
				break
	return found



class AsrKeywordSpotter(object):

	"""
	Wake words without Porcupine. process(frame, prob) collects voiced
	segments (VAD probability >= threshold); a segment that ends (short
	pause) or reaches max_segment_s is transcribed on a background thread
	by transcribe(wav_bytes) -> (text, no_speech_prob), and the keyword
	ids found in the text are returned by a later process() call.

	Only one transcription runs at a time and at most one waits; anything
	more is dropped (a wake phrase is short, and old audio is stale).
	"""

	needs_prob = True

	def __init__(self, ids, transcribe, threshold=0.5, onset_frames=2, end_silence_s=0.35,
			max_segment_s=2.5, preroll_s=0.3, max_no_speech=0.6, log=print):

		self.ids = [i for i in ids if i in WAKE_LEADS]
		if not self.ids:
			raise RuntimeError("no wake phrases to listen for")
		self.transcribe = transcribe
		self.threshold = float(threshold)
		self.onset_frames = int(onset_frames)
		frame = audio_util.HEARING_FRAME
		rate = float(audio_util.HEARING_RATE)
		self.end_frames = max(1, int(round(end_silence_s * rate / frame)))
		self.max_frames = int(round(max_segment_s * rate / frame))
		self.max_no_speech = float(max_no_speech)
		self.log = log
		self._pre = collections.deque(maxlen=max(1, int(round(preroll_s * rate / frame))))
		self._seg = None
		self._run = 0
		self._quiet = 0
		self._found = []
		self._lock = threading.Lock()
		self._cond = threading.Condition(self._lock)
		self._pending = None
		self._busy = False
		self._running = True
		self.mask = None
		self.last_text = ""
		self.calls = 0
		self.dropped = 0
		self._thread = threading.Thread(target=self._worker, name="wake-asr")
		self._thread.daemon = True
		self._thread.start()

	def process(self, frame, prob=None):

		frame = audio_util.to_int16(frame)
		voiced = prob is not None and prob >= self.threshold
		if self._seg is None:
			self._pre.append(frame)
			self._run = self._run + 1 if voiced else 0
			if self._run >= self.onset_frames:
				self._seg = list(self._pre)
				self._quiet = 0
		else:
			self._seg.append(frame)
			self._quiet = 0 if voiced else self._quiet + 1
			if self._quiet >= self.end_frames or len(self._seg) >= self.max_frames:
				self._submit(np.concatenate(self._seg))
				self._seg = None
				self._run = 0
				self._pre.clear()
		with self._lock:
			found, self._found = self._found, []
		return found

	def reset(self):

		# forget a segment in progress (the mics were muted)
		self._seg = None
		self._run = 0
		self._pre.clear()

	def _submit(self, pcm):

		if self.mask is not None and not self.mask:
			return
		with self._cond:
			if self._pending is not None:
				self.dropped += 1
			self._pending = pcm
			self._cond.notify()

	def _worker(self):

		while True:
			with self._cond:
				while self._running and self._pending is None:
					self._cond.wait(0.5)
				if not self._running:
					return
				pcm, self._pending = self._pending, None
			try:
				t0 = time.monotonic()
				text, no_speech = self.transcribe(audio_util.wav_bytes(pcm, audio_util.HEARING_RATE))
				self.calls += 1
			except Exception as e:
				if self.calls < 3:
					self.log("[wakeword] transcription failed: " + e.__class__.__name__)
				continue
			if no_speech is not None and no_speech > self.max_no_speech:
				continue
			self.last_text = text
			ids = self.mask if self.mask is not None else self.ids
			found = match_wake_phrase(text, ids)
			if found:
				self.log("[wakeword] heard %r -> %s (%.0f ms)" % (text, ",".join(found),
					(time.monotonic() - t0) * 1000))
				with self._lock:
					self._found.extend(found)

	def delete(self):

		with self._cond:
			self._running = False
			self._cond.notify()
