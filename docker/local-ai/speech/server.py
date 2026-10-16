"""
Hey MiRo local speech service.

OpenAI-compatible endpoints, so the robot code reaches them through the
same clients it uses for the cloud (only the base URL changes):

  POST /v1/audio/transcriptions   faster-whisper (Whisper large-v3-turbo)
  POST /v1/audio/speech           Kokoro-82M; response_format=pcm streams
                                  s16le 24 kHz mono, one chunk per sentence
  POST /v1/turn                   smart-turn v3: has the speaker finished?
                                  body = s16le 16 kHz mono PCM
  GET  /health                    models loaded, device, GPU memory

Extensions to /v1/audio/speech (ignored by OpenAI clients): "pitch"
(factor, 1.0 = natural; MiRo uses ~1.15 for a smaller-sounding voice) and
"stream" (default true for pcm).

Every model is read from MODELS_DIR (the heymiro-models volume); nothing is
downloaded at run time (HF_HUB_OFFLINE=1).
"""

import io
import logging
import os
import re
import threading
import time
from fractions import Fraction

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from scipy.signal import resample_poly

log = logging.getLogger("speech")
logging.basicConfig(level=logging.INFO, format="[speech] %(message)s")

MODELS = os.environ.get("MODELS_DIR", "/models")
STT_DIR = os.environ.get("STT_DIR", MODELS + "/deepdml/faster-whisper-large-v3-turbo-ct2")
STT_COMPUTE = os.environ.get("STT_COMPUTE", "int8_float16")
STT_BEAM = int(os.environ.get("STT_BEAM", "1"))
TTS_DIR = os.environ.get("TTS_DIR", MODELS + "/hexgrad/Kokoro-82M")
TTS_LANG = os.environ.get("TTS_LANG", "a")          # a = American English, b = British
TTS_VOICE = os.environ.get("TTS_VOICE", "af_heart")
TURN_MODEL = os.environ.get("TURN_MODEL", MODELS + "/pipecat-ai/smart-turn-v3/smart-turn-v3.2-cpu.onnx")

STT_RATE = 16000
TTS_RATE = 24000
TURN_SECONDS = 8

# sentence boundary for streaming TTS: after . ! ? (and runs of them)
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

app = FastAPI(title="heymiro-speech")
_stt_lock = threading.Lock()
_tts_lock = threading.Lock()
_state = {"stt": None, "tts": None, "turn": None, "turn_fe": None, "device": "cpu", "errors": {}}


# --- loading ---------------------------------------------------------------------------------

def _device():

	try:
		import torch
		return "cuda" if torch.cuda.is_available() else "cpu"
	except Exception:
		return "cpu"


def _load_stt(device):

	from faster_whisper import WhisperModel
	compute = STT_COMPUTE if device == "cuda" else "int8"
	model = WhisperModel(STT_DIR, device=device, compute_type=compute)
	model.transcribe(np.zeros(STT_RATE, dtype=np.float32), language="en", beam_size=1)
	return model


def _load_tts(device):

	from kokoro import KModel, KPipeline
	kmodel = KModel(repo_id="hexgrad/Kokoro-82M", config=TTS_DIR + "/config.json",
		model=TTS_DIR + "/kokoro-v1_0.pth").to(device).eval()
	pipeline = KPipeline(lang_code=TTS_LANG, repo_id="hexgrad/Kokoro-82M", model=kmodel)
	for _ in pipeline("Hi!", voice=_voice_path(TTS_VOICE)):
		pass
	return pipeline


def _load_turn():

	import onnxruntime as ort
	from transformers import WhisperFeatureExtractor
	so = ort.SessionOptions()
	so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
	so.inter_op_num_threads = 1
	so.intra_op_num_threads = 2
	so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
	session = ort.InferenceSession(TURN_MODEL, sess_options=so, providers=["CPUExecutionProvider"])
	return session, WhisperFeatureExtractor(chunk_length=TURN_SECONDS)


@app.on_event("startup")
def _startup():

	device = _device()
	_state["device"] = device
	log.info("device: %s", device)
	for name, loader in (("stt", lambda: _load_stt(device)), ("tts", lambda: _load_tts(device)),
			("turn", _load_turn)):
		t0 = time.time()
		try:
			value = loader()
			if name == "turn":
				_state["turn"], _state["turn_fe"] = value
			else:
				_state[name] = value
			log.info("%s ready (%.1f s)", name, time.time() - t0)
		except Exception as e:
			# keep serving what did load; /health reports the rest
			_state["errors"][name] = type(e).__name__ + ": " + str(e)[:200]
			log.error("%s failed: %s", name, _state["errors"][name])


# --- helpers ---------------------------------------------------------------------------------

def _voice_path(voice):

	name = re.sub(r"[^a-z0-9_]", "", (voice or TTS_VOICE).lower()) or TTS_VOICE
	path = TTS_DIR + "/voices/" + name + ".pt"
	if not os.path.isfile(path):
		path = TTS_DIR + "/voices/" + TTS_VOICE + ".pt"
	return path


def _decode_audio(data):

	# any soundfile-readable container (wav/flac/ogg) -> mono float32 16 kHz
	try:
		audio, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
	except Exception:
		raise HTTPException(400, "unreadable audio (send wav)")
	audio = audio.mean(axis=1)
	if rate != STT_RATE:
		r = Fraction(STT_RATE, rate).limit_denominator(1000)
		audio = resample_poly(audio, r.numerator, r.denominator).astype(np.float32)
	return audio


def _pitch_shift(audio, factor):

	# synthesized at speed/factor (so it is factor x longer), then played
	# factor x faster: pitch rises by factor, duration is back to normal
	if abs(factor - 1.0) < 1e-3:
		return audio
	r = Fraction(factor).limit_denominator(40)
	return resample_poly(audio, r.denominator, r.numerator)


def _to_s16(audio):

	return (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def _synthesize(text, voice, speed, pitch):

	# yields s16le chunks, one per sentence, as soon as each is ready
	pipeline = _state["tts"]
	vpath = _voice_path(voice)
	for sentence in SENTENCE_SPLIT.split(text.strip()):
		if not sentence.strip():
			continue
		with _tts_lock:
			parts = [np.asarray(a, dtype=np.float32) for _, _, a in
				pipeline(sentence, voice=vpath, speed=speed / pitch, split_pattern=None)
				if a is not None]
		if parts:
			yield _to_s16(_pitch_shift(np.concatenate(parts), pitch))


# --- endpoints -------------------------------------------------------------------------------

@app.get("/health")
def health():

	info = {
		"ok": all(_state[k] is not None for k in ("stt", "tts", "turn")),
		"device": _state["device"],
		"stt": _state["stt"] is not None,
		"tts": _state["tts"] is not None,
		"turn": _state["turn"] is not None,
		"errors": _state["errors"],
	}
	if _state["device"] == "cuda":
		import torch
		info["torch_vram_mb"] = round(torch.cuda.memory_allocated() / 1e6)
	return JSONResponse(info, status_code=200 if info["ok"] else 503)


@app.get("/v1/models")
def models():

	return {"object": "list", "data": [
		{"id": "whisper-large-v3-turbo", "object": "model"},
		{"id": "kokoro", "object": "model"},
		{"id": "smart-turn-v3", "object": "model"},
	]}


@app.post("/v1/audio/transcriptions")
async def transcriptions(file: UploadFile = File(...), model: str = Form("whisper-1"),
		language: str = Form(None), prompt: str = Form(None),
		response_format: str = Form("json"), temperature: float = Form(0.0)):

	if _state["stt"] is None:
		raise HTTPException(503, "stt not loaded")
	audio = _decode_audio(await file.read())
	t0 = time.time()
	with _stt_lock:
		segments, info = _state["stt"].transcribe(audio, language=language or None,
			initial_prompt=prompt or None, beam_size=STT_BEAM, temperature=temperature or 0.0,
			vad_filter=False, condition_on_previous_text=False, without_timestamps=True)
		segments = list(segments)
	text = " ".join(s.text.strip() for s in segments).strip()
	ms = round((time.time() - t0) * 1000)
	if response_format == "text":
		return Response(text, media_type="text/plain")
	if response_format == "verbose_json":
		return {"text": text, "language": info.language, "duration": info.duration, "ms": ms,
			"segments": [{"text": s.text, "no_speech_prob": s.no_speech_prob,
				"avg_logprob": s.avg_logprob} for s in segments]}
	return {"text": text}


@app.post("/v1/audio/speech")
async def speech(request: Request):

	if _state["tts"] is None:
		raise HTTPException(503, "tts not loaded")
	body = await request.json()
	text = str(body.get("input") or "").strip()[:2000]
	if not text:
		raise HTTPException(400, "empty input")
	voice = body.get("voice") or TTS_VOICE
	speed = min(max(float(body.get("speed") or 1.0), 0.5), 2.0)
	pitch = min(max(float(body.get("pitch") or 1.0), 0.7), 1.6)
	fmt = (body.get("response_format") or "pcm").lower()
	headers = {"X-Sample-Rate": str(TTS_RATE)}

	if fmt == "pcm" and body.get("stream", True):
		return StreamingResponse(_synthesize(text, voice, speed, pitch), media_type="audio/pcm",
			headers=headers)
	pcm = b"".join(_synthesize(text, voice, speed, pitch))
	if fmt == "pcm":
		return Response(pcm, media_type="audio/pcm", headers=headers)
	if fmt == "wav":
		buf = io.BytesIO()
		sf.write(buf, np.frombuffer(pcm, dtype="<i2"), TTS_RATE, format="WAV", subtype="PCM_16")
		return Response(buf.getvalue(), media_type="audio/wav", headers=headers)
	raise HTTPException(400, "response_format must be pcm or wav")


@app.post("/v1/turn")
async def turn(request: Request):

	if _state["turn"] is None:
		raise HTTPException(503, "turn model not loaded")
	data = await request.body()
	audio = np.frombuffer(data[:len(data) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0
	n = TURN_SECONDS * STT_RATE
	audio = audio[-n:] if len(audio) > n else np.pad(audio, (n - len(audio), 0))
	t0 = time.time()
	feats = _state["turn_fe"](audio, sampling_rate=STT_RATE, return_tensors="np", padding="max_length",
		max_length=n, truncation=True, do_normalize=True).input_features.astype(np.float32)
	prob = float(_state["turn"].run(None, {"input_features": feats})[0].reshape(-1)[0])
	return {"complete": prob > 0.5, "probability": prob, "ms": round((time.time() - t0) * 1000)}
