#
#	Hey MiRo - text-to-speech (ElevenLabs REST API).
#
#	synthesize() asks for raw 16-bit PCM at 24 kHz, the rate NodeVoice
#	mixes at, so normally no decoding is needed at all. If the server
#	answers with mp3 anyway (some plans/proxies do) it is decoded with
#	ffmpeg. The HRI'25 code used the ElevenLabs SDK plus pydub inside
#	the 50 Hz tick; call this on the CloudWorker instead.
#
#	The API key travels only in the xi-api-key header and is never
#	printed; errors carry the HTTP status only.
#

from urllib.parse import quote

import numpy as np

from .music import ffmpeg_decode
from .worker import HttpError



TTS_RATE = 24000

# normalise the speech peak to this fraction of full scale (headroom
# for the mixer and the resampler)
PEAK_LEVEL = 0.9



def _cfg_get(cfg, key, default):

	# cfg_tts may be a heymiro_config Section or a plain dict
	if cfg is None:
		return default
	value = cfg.get(key, default)
	return default if value is None else value


def normalise_peak(pcm, level=PEAK_LEVEL):

	# scale so the loudest sample sits at level * full scale; silence
	# stays silence (the HRI code divided by max(abs) unguarded)
	x = np.asarray(pcm, dtype=np.float32)
	peak = float(np.max(np.abs(x))) if x.size else 0.0
	if peak <= 0.0:
		return np.zeros(x.shape, dtype=np.int16)
	x = x * (level * 32767.0 / peak)
	return np.clip(np.round(x), -32768, 32767).astype(np.int16)


def is_mp3(content_type, data):

	ctype = (content_type or "").lower().strip()
	if "mpeg" in ctype or "mp3" in ctype:
		return True
	if ctype:
		# the server said what it is (pcm, octet-stream, ...): trust it;
		# raw s16le PCM can start with bytes that look like an MPEG sync
		return False
	# no content type: an ID3 tag, or a plausible MPEG frame header
	# (sync, a defined version/layer, and a bitrate index that is not
	# "free" or "bad")
	if data[:3] == b"ID3":
		return True
	if len(data) < 3 or data[0] != 0xFF or (data[1] & 0xE0) != 0xE0:
		return False
	version = (data[1] >> 3) & 0x03
	layer = (data[1] >> 1) & 0x03
	bitrate = (data[2] >> 4) & 0x0F
	return version != 1 and layer != 0 and bitrate not in (0, 15)



class TextToSpeech(object):

	def __init__(self, route, cfg_tts=None, session=None):

		import requests
		self.route = route
		self.session = session or requests.Session()
		self.model_id = _cfg_get(cfg_tts, "model_id", "eleven_multilingual_v2")
		self.voice_id = _cfg_get(cfg_tts, "voice_id", None)
		self.stability = float(_cfg_get(cfg_tts, "stability", 0.5))
		self.similarity_boost = float(_cfg_get(cfg_tts, "similarity_boost", 0.75))
		timeout = float(_cfg_get(cfg_tts, "timeout_s", getattr(route, "timeout_s", 20.0)))
		self.timeout = (5.0, timeout)

	def request(self, text, voice_id=None):

		# (url, params, headers, json) of the REST call; kept separate
		# so the tests can check the request without a network
		voice = voice_id or self.voice_id
		if not voice:
			raise ValueError("no ElevenLabs voice_id configured (tts.voice_id)")
		url = self.route.base_url + "/v1/text-to-speech/" + quote(str(voice), safe="")
		params = {"output_format": "pcm_24000"}
		headers = {
			"xi-api-key": self.route.api_key or "",
			"Content-Type": "application/json",
			"Accept": "audio/pcm, audio/mpeg;q=0.5",
		}
		body = {
			"text": text,
			"model_id": self.model_id,
			"voice_settings": {
				"stability": self.stability,
				"similarity_boost": self.similarity_boost,
			},
		}
		return url, params, headers, body

	def synthesize(self, text, voice_id=None):

		"""
		Speech for text as int16 mono at 24 kHz (empty array for empty
		text). Raises HttpError (status only) or RuntimeError.
		"""

		text = (text or "").strip()
		if not text:
			return np.zeros(0, dtype=np.int16)
		url, params, headers, body = self.request(text, voice_id)
		r = self.session.post(url, params=params, headers=headers, json=body, timeout=self.timeout)
		if r.status_code != 200:
			raise HttpError("tts", r.status_code)
		data = r.content or b""
		if is_mp3(r.headers.get("Content-Type"), data):
			pcm = ffmpeg_decode(data=data, rate=TTS_RATE)
		else:
			# raw little-endian s16 (drop a stray odd byte)
			pcm = np.frombuffer(data[:len(data) // 2 * 2], dtype="<i2")
		return normalise_peak(pcm)
