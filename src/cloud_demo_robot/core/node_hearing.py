#
#	Hey MiRo - NodeHearing: the linguistic side of MiRo's ears.
#
#	Mic frames (4 x 500 @ 20 kHz, from sensors/mics or the host bridge's
#	host/mics) arrive in a ROS callback that only enqueues them. A worker
#	thread then does, per frame:
#
#	  touch-noise detector (ear/tail scratching, MiRo mics only)
#	  -> mix ear channels -> 20 kHz to 16 kHz (anti-aliased) -> 512 frames
#	  -> per 512 frame: wake words (Porcupine) -> VAD -> utterance segmenter
#
#	and posts events to a queue that the forebrain drains in its tick:
#	  ("wake", keyword_id) ("voiced", None) ("utterance", int16 16 kHz)
#	  ("no_speech", None) ("touch", "ear" | "tail")
#
#	While MiRo is speaking (state.in_speaking, from NodeVoice) and for
#	self_hearing_guard_s afterwards the mics are ignored, so MiRo does not
#	wake itself up or record its own voice (there is no echo cancellation).
#
#	This replaces HRI'25 node_detect_word (three Porcupine instances, an
#	unfiltered np.interp decimation, an int16-overflowing channel sum) and
#	the recording logic inside action_llm.audio_cb.
#

import queue
import threading
import time

import numpy as np

import node
import audio_util
from forebrain.wakeword import KeywordSpotter, make_vad
from forebrain.segmenter import UtteranceSegmenter
from forebrain.touch_noise import TouchNoiseDetector


__all__ = ["NodeHearing"]

# about 1.25 s of mic messages (40 per second)
INPUT_QUEUE_SIZE = 50
EVENT_QUEUE_SIZE = 200



class NodeHearing(node.Node):

	def __init__(self, sys, spotter=None, vad=None, start=True):

		node.Node.__init__(self, sys, "hearing")

		heymiro = self.pars.forebrain
		cfg = heymiro.cfg.hearing
		features = heymiro.features
		self.cfg = cfg

		# mixing
		self.channels = list(cfg.get("miro_channels", [0, 1]))
		self.gain = float(cfg.get("gain", 1.0))
		self.guard_s = float(cfg.get("self_hearing_guard_s", 0.3))

		# wake words (injected for tests, else Porcupine if enabled and possible)
		self.spotter = spotter
		if self.spotter is None and features.get("wake_word", False):
			self.spotter = self._make_spotter(heymiro, cfg)
		self.keyword_ids = list(self.spotter.ids) if self.spotter is not None else []

		# voice activity
		self.vad = vad
		if self.vad is None:
			kind = "cobra" if features.get("vad_cobra", False) else "energy"
			self.vad = make_vad(kind, heymiro.picovoice_key(), cfg.get("energy_threshold_dbfs", -38.0))
		self.vad_kind = getattr(self.vad, "kind", "custom")

		# utterances
		self.segmenter = UtteranceSegmenter(
			rate=audio_util.HEARING_RATE,
			frame=audio_util.HEARING_FRAME,
			vad_threshold=cfg.get("vad_threshold", 0.6),
			onset_frames=cfg.get("onset_frames", 3),
			end_silence_s=cfg.get("end_silence_s", 3.0),
			no_speech_timeout_s=cfg.get("no_speech_timeout_s", 3.0),
			max_utterance_s=cfg.get("max_utterance_s", 20.0),
			preroll_s=cfg.get("preroll_s", 0.5))

		# ear / tail scratching (only valid on MiRo's own mics)
		self.touch = None
		if features.get("touch_reactions", False):
			tn = cfg.get("touch_noise")
			self.touch = TouchNoiseDetector(
				zcr_min=tn.get("zcr_min", 190) if tn else 190,
				peak_ear=tn.get("peak_ear", 23000) if tn else 23000,
				peak_tail=tn.get("peak_tail", 30000) if tn else 30000)

		# DSP
		self.resampler = audio_util.StreamResampler(audio_util.MIC_RATE, audio_util.HEARING_RATE)
		self.chunker = audio_util.FrameChunker(audio_util.HEARING_FRAME)

		# control (set from the tick, read by the worker)
		self._mask = None                  # None = every keyword
		self._commands = queue.Queue()     # "arm" / "cancel" for the segmenter
		self._mute_forced = False
		self._speaking = False
		self._guard_until = 0.0
		self._was_muted = False

		# status
		self.last_prob = 0.0
		self.level_dbfs = -120.0
		self.frames = 0
		self.drops = 0
		self.errors = 0

		# threads
		self._input = queue.Queue(maxsize=INPUT_QUEUE_SIZE)
		self._events = queue.Queue(maxsize=EVENT_QUEUE_SIZE)
		self._running = True
		self._thread = threading.Thread(target=self._worker, name="hearing")
		self._thread.daemon = True
		if start:
			self._thread.start()

		print("[hearing] wake words: " + (", ".join(self.keyword_ids) if self.keyword_ids else "none") +
			"; VAD: " + self.vad_kind + "; touch: " + ("on" if self.touch else "off"))

	def _make_spotter(self, heymiro, cfg):

		keywords = []
		for w in (cfg.get("wake_words") or []):
			keywords.append({
				"id": w.get("id"),
				"path": heymiro.path(w.get("file")),
				"sensitivity": w.get("sensitivity", 0.6),
				"access_key": heymiro.picovoice_key(w.get("access_key_env")),
			})
		try:
			return KeywordSpotter(keywords)
		except RuntimeError as e:
			print("[hearing] wake words disabled: " + str(e))
			return None

	# ----------------------------------------------------- ROS callback thread

	def on_mics(self, msg):

		# validate and enqueue only; dropping the oldest keeps latency bounded
		frame = audio_util.miro_frame_from_msg(msg.data)
		if frame is None:
			return
		try:
			self._input.put_nowait(frame)
		except queue.Full:
			try:
				self._input.get_nowait()
			except queue.Empty:
				pass
			self.drops += 1
			try:
				self._input.put_nowait(frame)
			except queue.Full:
				pass

	# ------------------------------------------------------- control (tick)

	def set_keyword_mask(self, ids):

		# reference swap, so the worker sees either the old or the new set
		self._mask = None if ids is None else frozenset(ids)

	def arm_utterance(self):

		self._commands.put("arm")

	def cancel_utterance(self):

		self._commands.put("cancel")

	def mute(self, on):

		self._mute_forced = bool(on)

	def muted(self):

		return self._mute_forced or self._speaking or time.monotonic() < self._guard_until

	def poll_events(self):

		events = []
		while True:
			try:
				events.append(self._events.get_nowait())
			except queue.Empty:
				return events

	def tick(self):

		# self-hearing guard: muted while speaking and a little after
		speaking = bool(getattr(self.state, "in_speaking", 0.0))
		if self._speaking and not speaking:
			self._guard_until = time.monotonic() + self.guard_s
		self._speaking = speaking

	def shutdown(self):

		self._running = False
		try:
			self._input.put_nowait(None)
		except queue.Full:
			pass
		if self._thread.is_alive():
			self._thread.join(1.0)
		for engine in (self.spotter, self.vad):
			if engine is not None:
				try:
					engine.delete()
				except Exception:
					pass
		self.spotter = None

	# ---------------------------------------------------------------- worker

	def _post(self, name, payload=None):

		try:
			self._events.put_nowait((name, payload))
		except queue.Full:
			self.drops += 1

	def _worker(self):

		while self._running:
			try:
				frame = self._input.get(timeout=0.2)
			except queue.Empty:
				self._apply_commands()
				continue
			if frame is None:
				break
			try:
				self.process_frame(frame)
			except Exception as e:
				# keep hearing alive; report the first few problems only
				self.errors += 1
				if self.errors <= 3:
					print("[hearing] error in worker: " + e.__class__.__name__ + " " + str(e)[:120])

	def _apply_commands(self):

		while True:
			try:
				cmd = self._commands.get_nowait()
			except queue.Empty:
				return
			if cmd == "arm":
				self.segmenter.arm()
			elif cmd == "cancel":
				self.segmenter.cancel()

	def process_frame(self, frame):

		# one (4, 500) int16 mic frame (worker thread; public for tests/tools)
		self.frames += 1
		self._apply_commands()
		muted = self.muted()

		if self.touch is not None and not muted:
			for hit in self.touch.feed(frame):
				self._post("touch", hit)

		mono = audio_util.mix_channels(frame, self.channels) * self.gain
		pcm = audio_util.to_int16(self.resampler.process(mono))
		self.level_dbfs = audio_util.level_dbfs(pcm)

		for f in self.chunker.push(pcm):
			if muted:
				# forget what was heard before/while MiRo spoke
				if not self._was_muted:
					self.segmenter.clear_preroll()
					if self.touch is not None:
						self.touch.reset()
				self._was_muted = True
				continue
			self._was_muted = False

			if self.spotter is not None:
				mask = self._mask
				for kid in self.spotter.process(f):
					if mask is None or kid in mask:
						self._post("wake", kid)

			prob = float(self.vad.process(f))
			self.last_prob = prob
			event = self.segmenter.feed(f, prob)
			if event is not None:
				self._post(event[0], event[1])
