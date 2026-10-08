#
#	Hey MiRo - NodeVoice: everything MiRo says or plays.
#
#	Two channels, both 24 kHz mono int16, mixed with their own gain:
#	  speech  clips (greeting, "hmm", ouch, ...) and TTS replies
#	  music   the thinking/loading tune and dance songs
#
#	Backends (voice.backend):
#	  host  a sounddevice OutputStream on the computer running the demo
#	        (laptop / Bluetooth speaker, as in the HRI'25 study); the
#	        PortAudio callback pulls from the mixer. Without a usable
#	        device it falls back to a silent clock-driven drain, so
#	        durations, is_speaking() and music positions still advance.
#	  miro  MiRo's own speaker: in tick(), resample 24 kHz to 8 kHz and
#	        keep the robot's buffer topped up using its sensors/stream
#	        feedback, exactly like the stock ActionSpecial.audio_play (HRI
#	        published 1000 samples at 10 Hz regardless, overrunning it).
#
#	tick() sets state.in_speaking, which NodeLoop uses for the stock
#	reafference gating and NodeHearing for its self-hearing guard.
#

import threading
import time

import numpy as np

import node
import audio_util
from forebrain.clips import ClipLibrary


__all__ = ["NodeVoice", "Mixer"]

RATE = audio_util.VOICE_RATE



class _Channel(object):

	def __init__(self, gain):

		self.gain = float(gain)
		self.pcm = None
		self.pos = 0
		self.loop = False

	def active(self):

		return self.pcm is not None and (self.loop or self.pos < self.pcm.size)

	def remaining(self):

		if self.pcm is None:
			return 0
		return max(0, self.pcm.size - self.pos)



class Mixer(object):

	"""
	Pull-based two-channel mixer (thread-safe). The output thread (or the
	tick, for the miro backend) pulls samples; positions advance only by
	what was actually pulled, so music_position follows real playback.
	"""

	def __init__(self, speech_gain=1.0, music_gain=0.6):

		self._lock = threading.Lock()
		self.ch = {"speech": _Channel(speech_gain), "music": _Channel(music_gain)}

	def play(self, name, pcm, loop=False):

		pcm = audio_util.to_int16(np.asarray(pcm).reshape(-1))
		with self._lock:
			c = self.ch[name]
			c.pcm = pcm if pcm.size else None
			c.pos = 0
			c.loop = bool(loop) and pcm.size > 0
		return pcm.size / float(RATE)

	def stop(self, name=None):

		with self._lock:
			for key, c in self.ch.items():
				if name is None or key == name:
					c.pcm = None
					c.pos = 0
					c.loop = False

	def active(self, name=None):

		with self._lock:
			if name is None:
				return any(c.active() for c in self.ch.values())
			return self.ch[name].active()

	def position(self, name):

		with self._lock:
			c = self.ch[name]
			return c.pos if c.pcm is not None else 0

	def remaining(self, name):

		with self._lock:
			return self.ch[name].remaining()

	def _take(self, c, n):

		# n samples from one channel (zero padded), advancing its position
		out = np.zeros(n, dtype=np.float32)
		if c.pcm is None:
			return out
		k = 0
		while k < n:
			if c.pos >= c.pcm.size:
				if not c.loop:
					break
				c.pos = 0
			m = min(n - k, c.pcm.size - c.pos)
			out[k:k + m] = c.pcm[c.pos:c.pos + m]
			c.pos += m
			k += m
		if not c.loop and c.pos >= c.pcm.size:
			c.pcm = None
			c.pos = 0
		return out * c.gain

	def pull(self, n):

		with self._lock:
			mix = self._take(self.ch["speech"], n) + self._take(self.ch["music"], n)
		return audio_util.to_int16(mix)

	def advance(self, n):

		# silent playback: move positions on without making samples
		with self._lock:
			for c in self.ch.values():
				if c.pcm is None:
					continue
				if c.loop:
					c.pos = (c.pos + n) % c.pcm.size
				else:
					c.pos += n
					if c.pos >= c.pcm.size:
						c.pcm = None
						c.pos = 0



class NodeVoice(node.Node):

	def __init__(self, sys, clock=None, open_device=True):

		node.Node.__init__(self, sys, "voice")

		heymiro = self.pars.forebrain
		cfg = heymiro.cfg.voice
		self.backend = str(cfg.get("backend", "host"))
		self.clock = clock if clock is not None else time.monotonic

		self.mixer = Mixer(cfg.get("volume", 1.0), cfg.get("music_volume", 0.6))
		self.clips = ClipLibrary.from_config(heymiro)

		# host backend
		self._stream = None
		self._silent = False
		self._silent_t = None
		self._silent_frac = 0.0

		# miro backend
		self._resampler = audio_util.StreamResampler(RATE, audio_util.SPKR_RATE)
		self._robot_buffered = 0          # samples queued on the robot (last feedback)
		self._robot_tail_until = 0.0      # speech still coming out of the robot until then
		self._last_feedback = None

		if self.backend == "host":
			if not (open_device and self._open_device(cfg.get("device", None))):
				self._go_silent("no output device")
		elif self.backend != "miro":
			print("[voice] WARN unknown voice.backend '" + self.backend + "'; using host")
			self.backend = "host"
			if not (open_device and self._open_device(cfg.get("device", None))):
				self._go_silent("no output device")

	# ----------------------------------------------------------- host backend

	def _open_device(self, device):

		try:
			import sounddevice as sd
		except (ImportError, OSError) as e:
			print("[voice] sounddevice unavailable (" + e.__class__.__name__ + "); install libportaudio2")
			return False
		try:
			self._stream = sd.OutputStream(samplerate=RATE, channels=1, dtype="int16",
				blocksize=480, device=device, callback=self._callback)
			self._stream.start()
		except Exception as e:
			print("[voice] cannot open output device " + str(device) + " (" + e.__class__.__name__ + ")")
			self._stream = None
			return False
		print("[voice] playing on host device " + str(device if device is not None else "(default)"))
		return True

	def _callback(self, outdata, frames, time_info, status):

		# PortAudio thread: pull from the mixer only
		outdata[:, 0] = self.mixer.pull(frames)

	def _go_silent(self, reason):

		if not self._silent:
			print("[voice] " + reason + ": audio is not played, a silent clock keeps time")
		self._silent = True
		self._silent_t = None

	def _silent_drain(self):

		now = self.clock()
		if self._silent_t is None:
			self._silent_t = now
			return
		# cap catch-up after a stall at 1 s; carry the fraction so time never drifts
		elapsed = min(max(0.0, now - self._silent_t), 1.0)
		self._silent_t = now
		samples = elapsed * RATE + self._silent_frac
		n = int(samples)
		self._silent_frac = samples - n
		if n > 0:
			self.mixer.advance(n)

	# ----------------------------------------------------------- miro backend

	def _miro_pace(self):

		now = self.clock()
		stream = self.input.stream
		if stream:
			# consume the feedback so we never send twice on stale data
			self.input.stream = None
			space, total = int(stream[0]), int(stream[1])
			self._last_feedback = now
			self._robot_buffered = max(0, total - space)
			n8k = audio_util.stream_send_count(space, total)
			if n8k > 0 and self.mixer.active():
				speech = self.mixer.active("speech")
				out = audio_util.to_int16(self._resampler.process(self.mixer.pull(3 * n8k)))
				if out.size:
					self.output.stream = out.tolist()
					self._robot_buffered += out.size
				if speech:
					self._robot_tail_until = now + self._robot_buffered / float(audio_util.SPKR_RATE)
			elif not self.mixer.active():
				self._resampler.reset()
		if self._silent and not self.mixer.active():
			# idle: restart the silent clock when the next sound starts,
			# so it does not "catch up" over the idle gap
			self._silent_t = None
			self._silent_frac = 0.0
		elif self.mixer.active() and (self._last_feedback is None or now - self._last_feedback > 1.0):
			# no sensors/stream feedback (no robot, or the bridge is not
			# running): keep time silently so the demo does not hang
			if not self._silent:
				self._go_silent("no sensors/stream feedback")
			self._silent_drain()
			return
		if self._silent and stream:
			print("[voice] sensors/stream feedback back; streaming to MiRo")
			self._silent = False

	# -------------------------------------------------------------------- API

	def play_clip(self, category, channel="speech"):

		pcm = self.clips.pick(category)
		if pcm is None:
			print("[voice] WARN no clip for '" + str(category) + "'")
			return 0.0
		return self.mixer.play(channel, pcm)

	def say_pcm(self, pcm24k):

		return self.mixer.play("speech", pcm24k)

	def play_music(self, pcm24k, loop=False):

		return self.mixer.play("music", pcm24k, loop)

	def stop(self, channel=None):

		self.mixer.stop(channel)
		if channel in (None, "speech"):
			self._robot_tail_until = 0.0

	def is_speaking(self):

		if self.mixer.active("speech"):
			return True
		return self.backend == "miro" and self.clock() < self._robot_tail_until

	def is_music_playing(self):

		return self.mixer.active("music")

	def _latency_s(self):

		# audio already sent to the robot but not yet heard
		if self.backend == "miro" and not self._silent:
			return self._robot_buffered / float(audio_util.SPKR_RATE)
		return 0.0

	def music_position_s(self):

		return max(0.0, self.mixer.position("music") / float(RATE) - self._latency_s())

	def speech_remaining_s(self):

		rem = self.mixer.remaining("speech") / float(RATE)
		if self.backend == "miro":
			rem = max(rem, self._robot_tail_until - self.clock())
		return max(0.0, rem)

	# ------------------------------------------------------------------- tick

	def tick(self):

		if self.backend == "miro":
			self._miro_pace()
		elif self._silent:
			self._silent_drain()
		self.state.in_speaking = 1.0 if self.is_speaking() else 0.0

	def shutdown(self):

		self.mixer.stop()
		stream = self._stream
		self._stream = None
		if stream is not None:
			try:
				stream.stop()
				stream.close()
			except Exception:
				pass
