#
#	Hey MiRo - pre-recorded voice clips.
#
#	config/clips.yaml maps a category (greeting, goodbye, thinking, ...)
#	to phrase folders under assets/clips/; every wav in those folders is
#	a candidate. All clips are decoded once at start-up (they are small)
#	so playing one never touches the disk on the 50 Hz path. HRI'25 built
#	paths with os.getcwd() and a typo ("talkign") crashed the goodbye; a
#	manifest test now checks every category resolves to real files.
#

import glob
import os
import random

import yaml

import audio_util



class ClipLibrary(object):

	def __init__(self, manifest_path, clips_dir, rate=audio_util.VOICE_RATE, rng=None):

		self.rate = int(rate)
		self.clips_dir = clips_dir
		self._rng = rng if rng is not None else random.Random()
		self._clips = {}      # category -> list of (path, int16 pcm)
		self._last = {}       # category -> index picked last time
		self.warnings = []

		with open(manifest_path, "r") as f:
			manifest = yaml.safe_load(f) or {}
		if not isinstance(manifest, dict):
			raise ValueError("clips manifest is not a mapping: " + manifest_path)

		cache = {}
		for category, folders in manifest.items():
			entries = []
			for folder in (folders or []):
				paths = sorted(glob.glob(os.path.join(clips_dir, str(folder), "*.wav")))
				if not paths:
					self.warnings.append("clip folder '" + str(folder) + "' (" + str(category) + ") has no wav files")
				for path in paths:
					if path not in cache:
						cache[path] = self._load(path)
					if cache[path] is not None:
						entries.append((path, cache[path]))
			self._clips[str(category)] = entries
		for w in self.warnings:
			print("[clips] WARN " + w)

	@classmethod
	def from_config(cls, heymiro, rng=None):

		# heymiro: HeyMiroConfig (pars.forebrain)
		cfg = heymiro.cfg
		manifest = heymiro.path(cfg.get("clips.manifest", "config/clips.yaml"))
		clips_dir = heymiro.path(cfg.get("clips.dir", "assets/clips"))
		return cls(manifest, clips_dir, rng=rng)

	def _load(self, path):

		try:
			pcm, rate = audio_util.load_wav(path)
		except Exception as e:
			self.warnings.append("cannot read " + path + " (" + e.__class__.__name__ + ")")
			return None
		if rate != self.rate:
			pcm = audio_util.to_int16(audio_util.resample(pcm, rate, self.rate))
		return pcm

	def categories(self):

		return sorted(self._clips.keys())

	def has(self, category):

		return len(self._clips.get(category, [])) > 0

	def files(self, category):

		return [p for p, pcm in self._clips.get(category, [])]

	def pick(self, category):

		# random clip, never the same one twice in a row (when there is a choice)
		entries = self._clips.get(category, [])
		if not entries:
			return None
		n = len(entries)
		last = self._last.get(category)
		if n == 1:
			k = 0
		else:
			k = self._rng.randrange(n - 1 if last is not None else n)
			if last is not None and k >= last:
				k += 1
		self._last[category] = k
		return entries[k][1]
