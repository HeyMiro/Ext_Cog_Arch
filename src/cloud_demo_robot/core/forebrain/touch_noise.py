#
#	Hey MiRo - ear / tail scratch detection from MiRo's own microphones.
#
#	Touching an ear or the tail rubs right next to a microphone and makes
#	loud, broadband scratching: a high peak AND many zero crossings, which
#	speech (loud but low zero-crossing rate) and hiss (many crossings but
#	quiet) do not have together. Heuristic ported from HRI'25
#	node_affect.affect_from_ear_tail, with the channel mix-up fixed: there
#	the "left" sum was ch1+ch1+ch2 and "right" ch0+ch0+ch1+ch2 (thresholds
#	67000/70000 on ~3 channels, 60000 on 2x the tail), so here each channel
#	is tested on its own with the per-channel equivalents (23000, 30000).
#
#	Channels (MiRo mics): 0 left ear, 1 right ear, 2 centre, 3 tail.
#	Only meaningful on sensors/mics (not a host mic, not the simulator).
#

import numpy as np

import audio_util


EAR_CHANNELS = (0, 1)
TAIL_CHANNEL = 3



class TouchNoiseDetector(object):

	def __init__(self, zcr_min=190, peak_ear=23000, peak_tail=30000, block=640):

		self.zcr_min = int(zcr_min)
		self.peak_ear = float(peak_ear)
		self.peak_tail = float(peak_tail)
		self.block = int(block)
		self._chunker = audio_util.FrameChunker(self.block, audio_util.MIC_CHANNELS)

	def reset(self):

		self._chunker.reset()

	@staticmethod
	def _zero_crossings(x):

		# sign changes (exact zeros count as positive, which is fine for noise)
		s = np.signbit(x)
		return int(np.count_nonzero(s[1:] != s[:-1]))

	def _scratched(self, x, peak):

		x = x.astype(np.int32)
		return np.max(np.abs(x)) > peak and self._zero_crossings(x) > self.zcr_min

	def feed(self, frame):

		# frame: (4, 500) int16 channel-major; works on 640-sample blocks like HRI
		hits = []
		frame = np.asarray(frame)
		for blk in self._chunker.push(frame.T):
			if any(self._scratched(blk[:, ch], self.peak_ear) for ch in EAR_CHANNELS):
				hits.append("ear")
			if self._scratched(blk[:, TAIL_CHANNEL], self.peak_tail):
				hits.append("tail")
		return hits
