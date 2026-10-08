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

import os

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



def make_vad(kind, access_key, threshold_dbfs=-38.0):

	# Cobra when asked for and possible, otherwise the energy VAD
	if kind == "cobra":
		try:
			return CobraVad(access_key)
		except RuntimeError as e:
			print("[wakeword] Cobra VAD unavailable (" + str(e) + "); using the energy VAD")
	return EnergyVad(threshold_dbfs)
