#
#	Hey MiRo - audio helpers shared by the hearing and voice nodes,
#	the host audio bridge and the test tools.
#
#	numpy only: no ROS, no miro2, no audio devices, so everything in here
#	can be unit tested. Sample values stay in int16 units (+-32768) even
#	when the arrays are float32, so levels and thresholds mean the same
#	thing everywhere.
#
#	MiRo microphone messages (sensors/mics, and host/mics from the bridge)
#	are Int16MultiArray of 2000 samples: 4 channels x 500 samples at
#	20 kHz, channel-major (ch0 left ear, ch1 right ear, ch2 centre, ch3
#	tail), the layout node_detect_audio_engine.py reshapes to (4, 500).
#

import io
import threading
import wave
from fractions import Fraction

import numpy as np



# MiRo microphones (sensors/mics)
MIC_RATE = 20000
MIC_BLOCK = 500
MIC_CHANNELS = 4

# Porcupine / Cobra want 16 kHz frames of 512 samples
HEARING_RATE = 16000
HEARING_FRAME = 512

# voice clips and ElevenLabs pcm_24000 output
VOICE_RATE = 24000

# MiRo speaker (control/stream), and the stock ActionSpecial pacing
# constants: messages larger than STREAM_MSG_MAX are dropped by the
# robot, and we keep about STREAM_STUFF samples (0.5 s) queued there
SPKR_RATE = 8000
STREAM_MSG_MAX = 4096 - 48
STREAM_STUFF = 4000



def to_int16(x):

	# saturating conversion; plain astype() would wrap around on overflow
	x = np.asarray(x)
	if x.dtype == np.int16:
		return x.copy()
	if np.issubdtype(x.dtype, np.floating):
		x = np.rint(x)
	return np.clip(x, -32768, 32767).astype(np.int16)



class StreamResampler(object):

	"""
	Stateful rational resampler (L/M polyphase, Kaiser-windowed sinc).

	The conversion ratio is reduced with the gcd, e.g. 20000->16000 is
	4/5 and 44100->20000 is 200/441. The last K-1 input samples and the
	output phase are carried between calls, so feeding a signal block by
	block gives exactly the same samples as feeding it in one go, and the
	output length never drifts (after N input samples there are always
	ceil(N * L / M) output samples).

	The filter passes up to 0.4 x the lower rate and is ~60 dB down at
	half the lower rate, which removes the aliasing the HRI np.interp
	decimation let through.
	"""

	def __init__(self, rate_in, rate_out, channels=1, atten_db=60.0):

		rate_in = int(rate_in)
		rate_out = int(rate_out)
		if rate_in <= 0 or rate_out <= 0:
			raise ValueError("sample rates must be positive")
		ratio = Fraction(rate_out, rate_in)
		self.rate_in = rate_in
		self.rate_out = rate_out
		self.channels = int(channels)
		self.up = ratio.numerator
		self.down = ratio.denominator
		self.passthrough = (self.up == 1 and self.down == 1)
		if not self.passthrough:
			self._design(atten_db)
		self.reset()

	def _design(self, atten_db):

		L = self.up
		M = self.down
		fs_up = float(self.rate_in) * L
		f_min = float(min(self.rate_in, self.rate_out))

		# transition band 0.4 .. 0.5 of the lower rate, cutoff in the middle
		f_pass = 0.4 * f_min
		f_stop = 0.5 * f_min
		f_cut = 0.5 * (f_pass + f_stop)
		d_omega = 2.0 * np.pi * (f_stop - f_pass) / fs_up

		# Kaiser formulae for length and beta
		n_taps = int(np.ceil((atten_db - 8.0) / (2.285 * d_omega))) + 1
		K = int(np.ceil(n_taps / float(L)))
		K = max(K, 2)
		n_taps = K * L
		if atten_db > 50.0:
			beta = 0.1102 * (atten_db - 8.7)
		else:
			beta = 0.5842 * (atten_db - 21.0) ** 0.4 + 0.07886 * (atten_db - 21.0)

		# windowed sinc prototype at the upsampled rate; gain L because
		# zero-stuffing by L divides the signal level by L
		n = np.arange(n_taps) - (n_taps - 1) / 2.0
		h = 2.0 * f_cut / fs_up * np.sinc(2.0 * f_cut / fs_up * n)
		h *= np.kaiser(n_taps, beta)
		h *= L / np.sum(h)

		# polyphase table: row p holds taps p, p+L, p+2L, ... so that
		# y[n] = sum_k H[p, k] * x[i - k] with t = n*M, i = t // L, p = t % L
		self.taps = K
		self._H = h.reshape(K, L).T.copy()

	def reset(self):

		# history starts as silence (the filter delay is part of the output)
		if self.passthrough:
			return
		self._hist = np.zeros((self.taps - 1, self.channels))
		# upsampled time of the next output, relative to the start of the next block
		self._t = 0

	def process(self, x):

		x = np.asarray(x)
		mono = (x.ndim == 1)
		if self.passthrough:
			return x.astype(np.float32)
		x2 = x.reshape(-1, 1) if mono else x
		if x2.shape[1] != self.channels:
			raise ValueError("expected " + str(self.channels) + " channel(s)")
		n_in = x2.shape[0]
		L = self.up
		M = self.down
		K = self.taps

		# outputs whose newest input sample is inside this block
		last = n_in * L - 1
		count = 0 if self._t > last else (last - self._t) // M + 1
		buf = np.concatenate((self._hist, x2.astype(np.float64)), axis=0)

		if count > 0:
			y = np.empty((count, self.channels))
			back = (K - 1) - np.arange(K)[None, :]
			# a few thousand outputs at a time keeps the gather table small
			# when a whole file is resampled in one call
			for a in range(0, count, 4096):
				b = min(count, a + 4096)
				t = self._t + M * np.arange(a, b, dtype=np.int64)
				i = t // L
				p = t % L
				# the K input samples for each output, newest first
				idx = i[:, None] + back
				y[a:b] = np.einsum("nk,nkc->nc", self._H[p], buf[idx])
			self._t = self._t + M * count - n_in * L
		else:
			y = np.zeros((0, self.channels))
			self._t -= n_in * L

		# keep the history for the next block
		self._hist = buf[buf.shape[0] - (K - 1):]

		y = y.astype(np.float32)
		return y[:, 0] if mono else y



def resample(x, rate_in, rate_out):

	# one-shot convenience (same filter, so it matches block-wise use)
	x = np.asarray(x)
	ch = 1 if x.ndim == 1 else x.shape[1]
	return StreamResampler(rate_in, rate_out, ch).process(x)



class FrameChunker(object):

	"""
	Cut a stream of arbitrary-sized blocks into frames of exactly `size`
	samples (e.g. 512 for Porcupine, 500 for MiRo mic messages). The
	remainder is kept for the next push.
	"""

	def __init__(self, size, channels=1):

		self.size = int(size)
		self.channels = int(channels)
		self.reset()

	def reset(self):

		self._buf = None

	def push(self, block):

		block = np.asarray(block)
		if self._buf is None or self._buf.shape[0] == 0:
			buf = block
		else:
			buf = np.concatenate((self._buf, block.astype(self._buf.dtype)), axis=0)
		n = buf.shape[0] // self.size
		frames = [buf[k * self.size:(k + 1) * self.size].copy() for k in range(n)]
		self._buf = buf[n * self.size:].copy()
		return frames

	def pending(self):

		return 0 if self._buf is None else self._buf.shape[0]



def miro_frame_from_msg(data):

	# validate a MiRo mics message and return it as (4, 500) int16
	try:
		x = np.asarray(data)
	except Exception:
		return None
	if x.ndim != 1 or x.size != MIC_CHANNELS * MIC_BLOCK:
		return None
	return to_int16(x).reshape((MIC_CHANNELS, MIC_BLOCK))


def mix_channels(frame, channels=(0, 1)):

	# mean (not sum, which overflowed in HRI) of the listed channels
	frame = np.asarray(frame)
	channels = list(channels) if channels else [0]
	return np.mean(frame[channels].astype(np.float32), axis=0).astype(np.float32)


def to_miro_frame(block, gain=1.0):

	"""
	Build a MiRo mics message payload from one 500-sample block.
	mono (500,): copied to all four channels, so whichever convention a
	consumer uses for ch2/ch3 it sees the signal; stereo (500, 2): ch0 = L,
	ch1 = R, ch2 = ch3 = (L + R) / 2. Returns int16 (2000,) channel-major.
	"""

	x = np.asarray(block, dtype=np.float64) * float(gain)
	if x.shape[0] != MIC_BLOCK:
		raise ValueError("block must have " + str(MIC_BLOCK) + " samples")
	out = np.empty((MIC_CHANNELS, MIC_BLOCK))
	if x.ndim == 1:
		out[:] = x[None, :]
	elif x.ndim == 2 and x.shape[1] == 1:
		out[:] = x[:, 0][None, :]
	elif x.ndim == 2 and x.shape[1] == 2:
		out[0] = x[:, 0]
		out[1] = x[:, 1]
		out[2] = out[3] = 0.5 * (x[:, 0] + x[:, 1])
	else:
		raise ValueError("block must be mono or stereo")
	return to_int16(out.reshape(-1))


def level_dbfs(x):

	# RMS level relative to int16 full scale; -120 for digital silence
	x = np.asarray(x, dtype=np.float64)
	if x.size == 0:
		return -120.0
	rms = np.sqrt(np.mean(x * x))
	if rms <= 0.0:
		return -120.0
	return float(max(-120.0, 20.0 * np.log10(rms / 32768.0)))



def load_wav(path):

	# read a PCM wav as int16 mono (stereo is averaged); returns (pcm, rate)
	with wave.open(path, "rb") as w:
		ch = w.getnchannels()
		width = w.getsampwidth()
		rate = w.getframerate()
		raw = w.readframes(w.getnframes())
	if width == 2:
		x = np.frombuffer(raw, dtype="<i2").astype(np.float64)
	elif width == 1:
		x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0) * 256.0
	elif width == 4:
		x = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 65536.0
	else:
		raise ValueError("unsupported wav sample width " + str(width) + " in " + path)
	if ch > 1:
		x = x[:(x.size // ch) * ch].reshape(-1, ch).mean(axis=1)
	return to_int16(x), rate


def wav_bytes(pcm, rate):

	# int16 mono -> complete RIFF/WAV file in memory (for Whisper uploads)
	pcm = to_int16(pcm).reshape(-1)
	f = io.BytesIO()
	with wave.open(f, "wb") as w:
		w.setnchannels(1)
		w.setsampwidth(2)
		w.setframerate(int(rate))
		w.writeframes(pcm.astype("<i2").tobytes())
	return f.getvalue()



class SpeakerBuffer(object):

	"""
	Thread-safe int16 FIFO that behaves like MiRo's speaker buffer: it
	holds at most `total` samples, rejects whole messages longer than
	STREAM_MSG_MAX (the robot drops those) and drops what does not fit.
	Used by the host bridge to emulate control/stream + sensors/stream.
	"""

	def __init__(self, total=8192):

		self.total = int(total)
		self._buf = np.zeros(self.total, dtype=np.int16)
		self._r = 0
		self._n = 0
		self._lock = threading.Lock()
		self.overflows = 0
		self.oversize = 0
		self.underruns = 0

	def write(self, samples):

		x = to_int16(np.asarray(samples).reshape(-1))
		if x.size > STREAM_MSG_MAX:
			self.oversize += 1
			return 0
		with self._lock:
			n = min(x.size, self.total - self._n)
			if n < x.size:
				self.overflows += 1
			w = (self._r + self._n) % self.total
			first = min(n, self.total - w)
			self._buf[w:w + first] = x[:first]
			self._buf[:n - first] = x[first:n]
			self._n += n
		return n

	def read(self, n):

		n = int(n)
		out = np.zeros(n, dtype=np.int16)
		with self._lock:
			m = min(n, self._n)
			first = min(m, self.total - self._r)
			out[:first] = self._buf[self._r:self._r + first]
			out[first:m] = self._buf[:m - first]
			self._r = (self._r + m) % self.total
			self._n -= m
			if m < n and m > 0:
				self.underruns += 1
		return out

	def occupancy(self):

		with self._lock:
			return self._n

	def space(self):

		with self._lock:
			return self.total - self._n



def stream_send_count(space, total, stuff=STREAM_STUFF, max_msg=STREAM_MSG_MAX):

	# stock ActionSpecial.audio_play: top the robot buffer up to `stuff`
	n = int(stuff) - (int(total) - int(space))
	return max(0, min(n, int(max_msg)))
