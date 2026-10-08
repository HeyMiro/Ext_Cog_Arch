#
#	Hey MiRo - utterance segmenter.
#
#	Turns a stream of 16 kHz frames plus a per-frame voice probability
#	(Cobra or the energy VAD) into one recording per user turn. It
#	replaces the HRI silence_count logic (which re-ran Cobra on whole
#	recordings and counted 640-sample callbacks): here every 512-sample
#	frame is classified once, and all timing is counted in samples so it
#	does not depend on the wall clock or on how the frames arrive.
#
#	arm()            wait for the user to speak
#	feed(frame, p)   -> None, or one event:
#	                    ("voiced", None)        speech onset (drives nods)
#	                    ("utterance", pcm16k)   turn finished (int16, includes pre-roll)
#	                    ("no_speech", None)     nobody spoke within the timeout
#
#	After "utterance" or "no_speech" the segmenter disarms itself.
#

import collections

import numpy as np



class UtteranceSegmenter(object):

	def __init__(self, rate=16000, frame=512, vad_threshold=0.6, onset_frames=3,
			end_silence_s=3.0, no_speech_timeout_s=3.0, max_utterance_s=20.0, preroll_s=0.5):

		self.rate = int(rate)
		self.frame = int(frame)
		self.vad_threshold = float(vad_threshold)
		self.onset_frames = max(1, int(onset_frames))
		self.end_silence = int(round(end_silence_s * self.rate))
		self.no_speech_timeout = int(round(no_speech_timeout_s * self.rate))
		self.max_utterance = int(round(max_utterance_s * self.rate))

		# trailing silence kept on the recording (the rest is not worth uploading)
		self.keep_tail = int(round(0.5 * self.rate))

		# a new "voiced" event during a capture needs this much silence first
		self.reonset_gap = int(round(0.5 * self.rate))

		# pre-roll ring: always filling, so the first syllable is not lost
		n = int(np.ceil(preroll_s * self.rate / float(self.frame)))
		self._ring = collections.deque(maxlen=max(n, self.onset_frames))

		self.armed = False
		self.capturing = False
		self._reset_counters()

	def _reset_counters(self):

		self._run = 0          # consecutive voiced frames
		self._waited = 0       # samples since arm() without an onset
		self._silence = 0      # samples since the last voiced frame (capturing)
		self._length = 0       # samples captured
		self._chunks = []
		self._in_speech = False

	def arm(self):

		self.armed = True
		self.capturing = False
		self._reset_counters()

	def cancel(self):

		self.armed = False
		self.capturing = False
		self._reset_counters()

	def clear_preroll(self):

		# called when the input was muted, so stale audio is not prepended
		self._ring.clear()
		self._run = 0

	def feed(self, frame, prob):

		frame = np.asarray(frame, dtype=np.int16).reshape(-1)
		n = frame.size
		voiced = (prob is not None) and (prob >= self.vad_threshold)
		self._ring.append(frame)
		self._run = self._run + 1 if voiced else 0

		if not self.armed:
			return None

		# waiting for speech
		if not self.capturing:
			if self._run >= self.onset_frames:
				self.capturing = True
				self._in_speech = True
				self._chunks = list(self._ring)
				self._length = sum(c.size for c in self._chunks)
				self._silence = 0
				return ("voiced", None)
			self._waited += n
			if self._waited >= self.no_speech_timeout:
				self.cancel()
				return ("no_speech", None)
			return None

		# capturing
		self._chunks.append(frame)
		self._length += n
		event = None
		if voiced:
			if not self._in_speech and self._run >= self.onset_frames:
				self._in_speech = True
				event = ("voiced", None)
			if self._in_speech:
				self._silence = 0
			else:
				self._silence += n
		else:
			self._silence += n
			if self._silence >= self.reonset_gap:
				self._in_speech = False

		if self._silence >= self.end_silence or self._length >= self.max_utterance:
			return ("utterance", self._finish())
		return event

	def _finish(self):

		pcm = np.concatenate(self._chunks) if self._chunks else np.zeros(0, dtype=np.int16)
		# drop most of the trailing silence
		cut = max(0, self._silence - self.keep_tail)
		if cut > 0 and cut < pcm.size:
			pcm = pcm[:pcm.size - cut]
		pcm = pcm[:self.max_utterance]
		self.cancel()
		return pcm.astype(np.int16)
