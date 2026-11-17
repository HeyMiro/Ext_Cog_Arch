#!/usr/bin/env python3
#
#	Hey MiRo - benchmark every stage of the local pipeline, separately.
#
#	  python3 tools/bench_pipeline.py                      # everything
#	  python3 tools/bench_pipeline.py --only stt,vad       # some parts
#	  python3 tools/bench_pipeline.py --images DIR         # photos for YOLO
#	  python3 tools/bench_pipeline.py --json out.json
#
#	Needs the docker/local-ai services (speech + llm) running; YOLO and
#	mediapipe run in this process (start the container with the GPU).
#	The speech used for the audio tests is synthesized by the speech
#	service (Kokoro, voices different from MiRo's), so the ground-truth
#	text is known; noise is added at known signal-to-noise ratios:
#
#	  white    broadband hiss (fans, electronics)
#	  rumble   < 150 Hz motor / wheel / handling noise, like MiRo's body
#	  babble   other voices talking at the same time (an exhibition)
#
#	Parts: vad, turn, wake, stt, llm, tts, yolo, gesture, vision
#	Results are printed as Markdown tables (docs/TECHNICAL.md uses them).
#

import argparse
import base64
import json
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "core"))

import audio_util                                   # noqa: E402
import heymiro_config                               # noqa: E402
from forebrain.wakeword import EnergyVad, SileroVad, match_wake_phrase, WAKE_PROMPT   # noqa: E402

RATE = audio_util.HEARING_RATE
FRAME = audio_util.HEARING_FRAME

SENTENCES = [
	"What is your favourite colour?",
	"Can you dance to Dancing Queen for me?",
	"I went to the Winter Gardens with my grandmother yesterday.",
	"Do you like going to school?",
	"My brother has a little brown dog called Biscuit.",
	"Can you spin around and wiggle your ears?",
	"What time is it right now?",
	"I think robots are really cool, especially you.",
	"Have you ever been to the seaside?",
	"Tell me a story about a dragon.",
]
WAKE_LINES = ["Hey MiRo!", "Hey MiRo, how are you?", "Hello MiRo!", "Stop MiRo!", "Dance MiRo!"]
NOT_WAKE = ["Hey there, how are you?", "I like mirrors.", "Hello everyone!", "Stop it please.",
	"Let's dance together.", "My friend Milo is here."]
VOICES = ["am_michael", "bf_emma"]
BABBLE_VOICES = ["af_bella", "am_michael", "bf_emma"]
SNRS = [None, 20, 10, 5, 0]


# ------------------------------------------------------------------ helpers

def pct(xs, p):

	xs = sorted(xs)
	if not xs:
		return float("nan")
	k = (len(xs) - 1) * p / 100.0
	f = int(k)
	c = min(f + 1, len(xs) - 1)
	return xs[f] + (xs[c] - xs[f]) * (k - f)


def table(title, header, rows):

	print()
	print("### " + title)
	print()
	print("| " + " | ".join(header) + " |")
	print("|" + "|".join("---" for _ in header) + "|")
	for r in rows:
		print("| " + " | ".join(str(c) for c in r) + " |")


def words(text):

	return re.findall(r"[a-z0-9']+", str(text).lower().replace("mirror", "miro"))


def wer(ref, hyp):

	r, h = words(ref), words(hyp)
	d = list(range(len(h) + 1))
	for i in range(1, len(r) + 1):
		prev, d[0] = d[0], i
		for j in range(1, len(h) + 1):
			cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
			prev, d[j] = d[j], cur
	return d[len(h)] / float(max(1, len(r)))


class Speech(object):

	def __init__(self, base_url):

		import requests
		self.base = base_url.rstrip("/")
		self.s = requests.Session()
		self.cache = {}

	def tts(self, text, voice, pitch=1.0):

		key = (text, voice, pitch)
		if key not in self.cache:
			t0 = time.monotonic()
			r = self.s.post(self.base + "/audio/speech", json={"input": text, "voice": voice,
				"response_format": "pcm", "stream": False, "pitch": pitch}, timeout=60)
			r.raise_for_status()
			dt = time.monotonic() - t0
			pcm = np.frombuffer(r.content[:len(r.content) // 2 * 2], dtype="<i2").astype(np.float32)
			self.cache[key] = (audio_util.resample(pcm, 24000, RATE).astype(np.float32), dt, len(pcm) / 24000.0)
		return self.cache[key]

	def stt(self, pcm16k, prompt=None, verbose=False):

		wav = audio_util.wav_bytes(audio_util.to_int16(pcm16k), RATE)
		data = {"language": "en", "response_format": "verbose_json"}
		if prompt:
			data["prompt"] = prompt
		t0 = time.monotonic()
		r = self.s.post(self.base + "/audio/transcriptions", files={"file": ("a.wav", wav, "audio/wav")},
			data=data, timeout=60)
		r.raise_for_status()
		dt = time.monotonic() - t0
		j = r.json()
		nsp = max([s["no_speech_prob"] for s in j.get("segments") or []] or [0.0])
		return j.get("text", ""), dt, nsp

	def turn(self, pcm16k):

		t0 = time.monotonic()
		r = self.s.post(self.base + "/turn", data=audio_util.to_int16(pcm16k).astype("<i2").tobytes(),
			headers={"Content-Type": "application/octet-stream"}, timeout=10)
		r.raise_for_status()
		return r.json()["probability"], time.monotonic() - t0


def rms(x):

	return float(np.sqrt(np.mean(np.square(np.asarray(x, dtype=np.float64))) + 1e-12))


def noise(kind, n, rng, speech=None):

	if kind == "white":
		return rng.standard_normal(n)
	if kind == "rumble":
		t = np.arange(n) / float(RATE)
		x = sum(np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28)) * a for f, a in ((32, 1.0), (55, 0.7), (90, 0.5), (130, 0.3)))
		low = np.cumsum(rng.standard_normal(n))
		low = low - np.convolve(low, np.ones(400) / 400.0, mode="same")
		return x + 0.3 * low / (rms(low) + 1e-9)
	if kind == "babble":
		return speech
	raise ValueError(kind)


def mix(clean, kind, snr, rng, babble=None):

	if snr is None:
		return clean.copy()
	nz = noise(kind, clean.size, rng, babble)
	if nz.size < clean.size:
		nz = np.tile(nz, int(np.ceil(clean.size / float(nz.size))))[:clean.size]
	nz = nz[:clean.size]
	g = rms(clean) / (rms(nz) * 10 ** (snr / 20.0))
	return clean + g * nz


def pad(pcm, before=0.5, after=0.8):

	return np.concatenate([np.zeros(int(before * RATE)), pcm, np.zeros(int(after * RATE))]).astype(np.float32)


def babble_track(sp, n):

	parts = [sp.tts(s, v)[0] for s, v in zip(SENTENCES[::-1], BABBLE_VOICES * 4)]
	track = np.zeros(n)
	for k, p in enumerate(parts):
		off = (k * 7919) % max(1, n)
		seg = np.tile(p, int(np.ceil((n + off) / float(p.size))))[off:off + n]
		track += seg
	return track


# ------------------------------------------------------------------- parts

def bench_vad(sp, rng, out):

	clean = [pad(sp.tts(s, VOICES[i % 2])[0]) for i, s in enumerate(SENTENCES)]
	rows = []
	for kind in ("white", "rumble", "babble"):
		for snr in SNRS:
			if snr is None and kind != "white":
				continue
			res = {}
			for name in ("silero", "energy"):
				tp = fp = fn = tn = 0
				times = []
				for c in clean:
					vad = SileroVad() if name == "silero" else EnergyVad()
					bab = babble_track(sp, c.size) if kind == "babble" else None
					x = mix(c, kind, snr, rng, bab)
					for i in range(0, x.size - FRAME + 1, FRAME):
						f = audio_util.to_int16(x[i:i + FRAME])
						truth = audio_util.level_dbfs(audio_util.to_int16(c[i:i + FRAME])) > -45.0
						t0 = time.perf_counter()
						p = vad.process(f)
						times.append(time.perf_counter() - t0)
						v = p >= 0.6
						tp += v and truth
						fp += v and not truth
						fn += (not v) and truth
						tn += (not v) and not truth
				prec = tp / float(max(1, tp + fp))
				rec = tp / float(max(1, tp + fn))
				res[name] = (2 * prec * rec / max(1e-9, prec + rec), fp / float(max(1, fp + tn)), 1000 * np.mean(times))
			rows.append([kind, "clean" if snr is None else "%d dB" % snr,
				"%.2f" % res["silero"][0], "%.0f%%" % (100 * res["silero"][1]),
				"%.2f" % res["energy"][0], "%.0f%%" % (100 * res["energy"][1])])
			out.setdefault("vad", []).append({"noise": kind, "snr": snr, "silero_f1": res["silero"][0],
				"silero_fpr": res["silero"][1], "energy_f1": res["energy"][0], "energy_fpr": res["energy"][1]})
	ms = out["vad_ms"] = {"silero": res["silero"][2], "energy": res["energy"][2]}
	table("Voice activity detection: frame F1 and false alarms (32 ms frames, threshold 0.6)",
		["noise", "SNR", "Silero F1", "Silero false alarms", "energy F1", "energy false alarms"], rows)
	print("\nCPU time per 32 ms frame: Silero %.2f ms, energy %.3f ms" % (ms["silero"], ms["energy"]))


def bench_turn(sp, out):

	full, cut, lat = [], [], []
	for i, s in enumerate(SENTENCES):
		pcm = sp.tts(s, VOICES[i % 2])[0]
		p, dt = sp.turn(np.concatenate([pcm, np.zeros(int(0.3 * RATE))]))
		full.append(p)
		lat.append(dt)
		# the same sentence stopped mid-way, as if the speaker paused to think
		k = int(pcm.size * 0.55)
		p2, dt = sp.turn(np.concatenate([pcm[:k], np.zeros(int(0.3 * RATE))]))
		cut.append(p2)
		lat.append(dt)
	acc_full = np.mean([p > 0.5 for p in full])
	acc_cut = np.mean([p <= 0.5 for p in cut])
	out["turn"] = {"finished_ok": acc_full, "unfinished_ok": acc_cut, "p50_ms": 1000 * pct(lat, 50)}
	table("End of turn (smart-turn v3.2, CPU): is the speaker finished, 0.3 s after they stop?",
		["case", "correct", "mean probability 'finished'"],
		[["finished sentences", "%d/%d" % (round(acc_full * len(full)), len(full)), "%.2f" % np.mean(full)],
		 ["stopped mid-sentence", "%d/%d" % (round(acc_cut * len(cut)), len(cut)), "%.2f" % np.mean(cut)]])
	print("\nLatency per check (HTTP included): p50 %.0f ms" % (1000 * pct(lat, 50)))


def bench_wake(sp, rng, out):

	rows = []
	for snr in (None, 10, 5, 0):
		hit = n_hit = false = n_false = 0
		lat = []
		for kind in ("white", "babble"):
			if snr is None and kind == "babble":
				continue
			for i, line in enumerate(WAKE_LINES + NOT_WAKE):
				pcm = pad(sp.tts(line, VOICES[i % 2])[0], 0.2, 0.3)
				bab = babble_track(sp, pcm.size) if kind == "babble" else None
				text, dt, nsp = sp.stt(mix(pcm, kind, snr, rng, bab), prompt=WAKE_PROMPT)
				got = match_wake_phrase(text) if nsp <= 0.6 else []
				lat.append(dt)
				if line in WAKE_LINES:
					n_hit += 1
					hit += got == match_wake_phrase(line)
				else:
					n_false += 1
					false += bool(got)
		rows.append(["clean" if snr is None else "%d dB" % snr, "%d/%d" % (hit, n_hit),
			"%d/%d" % (false, n_false), "%.0f ms" % (1000 * pct(lat, 50))])
		out.setdefault("wake", []).append({"snr": snr, "hits": hit, "of": n_hit, "false": false, "of_false": n_false})
	table("Wake phrases by local ASR (white noise and babble pooled)",
		["SNR", "wake phrases recognised", "false wakes (similar sentences)", "latency p50"], rows)


def bench_stt(sp, rng, out):

	# latency against utterance length
	rows = []
	for secs in (1, 3, 6, 10):
		text = " ".join(SENTENCES)
		pcm = sp.tts(text, "am_michael")[0][:secs * RATE]
		times = [sp.stt(pcm)[1] for _ in range(5)]
		rows.append(["%d s" % secs, "%.0f ms" % (1000 * pct(times, 50)), "%.3f" % (pct(times, 50) / secs)])
	out["stt_latency"] = rows
	table("Speech-to-text latency (Whisper large-v3-turbo, int8_float16, beam 1)",
		["utterance", "latency p50", "real-time factor"], rows)

	# accuracy in noise, with and without the 80 Hz high-pass
	rows = []
	for kind in ("white", "rumble", "babble"):
		for snr in SNRS:
			if snr is None and kind != "white":
				continue
			errs, errs_hp = [], []
			for i, s in enumerate(SENTENCES):
				c = pad(sp.tts(s, VOICES[i % 2])[0], 0.3, 0.3)
				bab = babble_track(sp, c.size) if kind == "babble" else None
				x = mix(c, kind, snr, rng, bab)
				errs.append(wer(s, sp.stt(x)[0]))
				hp = audio_util.HighPass(RATE, 80.0).process(x)
				errs_hp.append(wer(s, sp.stt(hp)[0]))
			rows.append([kind, "clean" if snr is None else "%d dB" % snr,
				"%.1f%%" % (100 * np.mean(errs)), "%.1f%%" % (100 * np.mean(errs_hp))])
			out.setdefault("stt_wer", []).append({"noise": kind, "snr": snr, "wer": np.mean(errs), "wer_hp": np.mean(errs_hp)})
	table("Speech-to-text word error rate in noise (10 sentences, 2 voices)",
		["noise", "SNR", "WER", "WER with 80 Hz high-pass"], rows)


def bench_llm(heymiro, out):

	from cloud import llm, tools
	from cloud.music import SongLibrary
	cfg = heymiro.cfg
	songs = tools.song_choices(SongLibrary(heymiro).songs)
	persona = llm.build_system_prompt(heymiro.persona(), cfg.get("llm.event_name", ""), songs)
	examples = llm.few_shot_messages(songs)
	import requests
	url = heymiro.route("llm").base_url + "/chat/completions"
	body_base = {"model": "heymiro", "max_tokens": 200, "temperature": 0.7, "top_p": 0.8, "top_k": 20,
		"cache_prompt": True, "chat_template_kwargs": {"enable_thinking": False},
		"response_format": tools.response_format(songs)}
	ctx = llm.context_message(0.6, 0.6, "Thursday 15:00")
	questions = ["What's your favourite colour?", "Can you dance to Dancing Queen?", "Tell me about yourself!",
		"Do you like the Winter Gardens?", "Can you spin around?"]

	def run(q, cache):
		msgs = llm.ChatMemory(0).messages(persona, ctx, q, examples, True)
		body = dict(body_base, messages=msgs, cache_prompt=cache)
		t0 = time.monotonic()
		r = requests.post(url, json=body, timeout=120).json()
		total = time.monotonic() - t0
		tm = r.get("timings", {})
		return total, tm, r.get("usage", {})

	run("Hi!", True)
	cold = [run(q, False) for q in questions[:2]]
	warm = [run(q, True) for q in questions]
	pp = [tm.get("prompt_per_second", 0) for _, tm, _ in cold]
	dec = [tm.get("predicted_per_second", 0) for _, tm, _ in warm]
	cached_pp_ms = [tm.get("prompt_ms", 0) for _, tm, _ in warm]
	cold_pp_ms = [tm.get("prompt_ms", 0) for _, tm, _ in cold]
	ptoks = [u.get("prompt_tokens", 0) for _, _, u in warm]
	gtoks = [u.get("completion_tokens", 0) for _, _, u in warm]
	out["llm"] = {"prompt_tokens": np.mean(ptoks), "prompt_ms_cold": np.mean(cold_pp_ms),
		"prompt_ms_cached": np.mean(cached_pp_ms), "decode_tok_s": np.mean(dec), "prefill_tok_s": np.mean(pp),
		"reply_tokens": np.mean(gtoks), "total_s": np.mean([t for t, _, _ in warm])}
	table("LLM (Qwen3.5 via llama.cpp, JSON-schema constrained)",
		["measure", "value"],
		[["prompt size (system + examples + turn)", "%.0f tokens" % np.mean(ptoks)],
		 ["prompt processing without cache", "%.0f ms (%.0f tokens/s)" % (np.mean(cold_pp_ms), np.mean(pp))],
		 ["prompt processing with cache (only the new turn)", "%.0f ms" % np.mean(cached_pp_ms)],
		 ["generation speed", "%.0f tokens/s" % np.mean(dec)],
		 ["reply length", "%.0f tokens" % np.mean(gtoks)],
		 ["whole reply (not streamed)", "%.2f s" % np.mean([t for t, _, _ in warm])]])


def bench_tts(sp, out):

	rows = []
	for text in ("Hi there!", "My favourite band is ABBA! Do you like them too?",
			"I was created in 2017 at the University of Sheffield, and I love to dance and talk with people like you!"):
		times = []
		dur = 0
		for k in range(5):
			sp.cache.clear()
			_, dt, dur = sp.tts(text, "af_heart", 1.15)
			times.append(dt)
		rows.append(["%d words" % len(text.split()), "%.2f s" % dur, "%.0f ms" % (1000 * pct(times, 50)),
			"%.3f" % (pct(times, 50) / dur)])
	out["tts"] = rows
	table("Text-to-speech (Kokoro-82M, pitch 1.15, one request)",
		["sentence", "audio length", "synthesis p50", "real-time factor"], rows)


def load_images(folder):

	import cv2
	imgs = []
	for name in sorted(os.listdir(folder)):
		img = cv2.imread(os.path.join(folder, name))
		if img is not None:
			# MiRo's camera frames are 640 x 360
			imgs.append(cv2.resize(img, (640, 360)))
	return imgs


def bench_yolo(images, out, weights):

	from ultralytics import YOLO
	import torch
	rows = []
	for dev in (["0"] if torch.cuda.is_available() else []) + ["cpu"]:
		for w in weights:
			model = YOLO(w)
			for _ in range(3):
				model.predict(images[0], device=dev, verbose=False, conf=0.4)
			times = []
			found = 0
			for _ in range(4):
				for img in images:
					t0 = time.perf_counter()
					r = model.predict(img, device=dev, verbose=False, conf=0.4)[0]
					times.append(time.perf_counter() - t0)
					found += len(r.boxes)
			rows.append([os.path.basename(w), "GPU" if dev != "cpu" else "CPU", "%.1f ms" % (1000 * pct(times, 50)),
				"%.0f" % (1.0 / pct(times, 50)), "%.1f" % (found / float(4 * len(images)))])
			out.setdefault("yolo", []).append({"weights": w, "device": dev, "ms": 1000 * pct(times, 50)})
	table("YOLO object detection (640 x 360 frame, all 80 classes, conf 0.4)",
		["model", "device", "per frame p50", "frames/s", "objects per frame"], rows)


def bench_gesture(images, out, model_path):

	import mediapipe as mp
	from mediapipe.tasks import python as mpt
	from mediapipe.tasks.python import vision
	import cv2
	rec = vision.GestureRecognizer.create_from_options(vision.GestureRecognizerOptions(
		base_options=mpt.BaseOptions(model_asset_path=model_path)))
	times = []
	for _ in range(4):
		for img in images:
			im = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
			t0 = time.perf_counter()
			rec.recognize(im)
			times.append(time.perf_counter() - t0)
	out["gesture_ms"] = 1000 * pct(times, 50)
	table("Hand gestures (mediapipe GestureRecognizer, CPU)", ["per frame p50"], [["%.1f ms" % out["gesture_ms"]]])


def bench_vision(heymiro, images, out):

	# the cost of giving the LLM the camera image vs YOLO's sentence
	import cv2
	import requests
	url = heymiro.route("llm").base_url + "/chat/completions"
	scene = "Your camera sees: a person in front of you, a cup on your left."
	rows = []

	def ask(content):
		body = {"model": "heymiro", "max_tokens": 1, "cache_prompt": False,
			"chat_template_kwargs": {"enable_thinking": False},
			"messages": [{"role": "user", "content": content}]}
		t0 = time.monotonic()
		r = requests.post(url, json=body, timeout=120).json()
		return r.get("usage", {}).get("prompt_tokens", 0), time.monotonic() - t0

	base_t, base_s = ask("What do you see?")
	ok, jpg = cv2.imencode(".jpg", images[0], [cv2.IMWRITE_JPEG_QUALITY, 85])
	data_url = "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode()
	img = [ask([{"type": "image_url", "image_url": {"url": data_url}}, {"type": "text", "text": "What do you see?"}])
		for _ in range(3)]
	txt = [ask("[" + scene + "]\nWhat do you see?") for _ in range(3)]
	img_t = np.mean([t for t, _ in img]) - base_t
	txt_t = np.mean([t for t, _ in txt]) - base_t
	out["vision"] = {"image_tokens": img_t, "image_s": np.mean([s for _, s in img]),
		"text_tokens": txt_t, "text_s": np.mean([s for _, s in txt])}
	table("Scene context: one camera frame as image tokens vs YOLO's sentence as text",
		["what the LLM gets", "extra prompt tokens", "time to process the prompt"],
		[["the 640 x 360 camera frame (vision encoder + image tokens)", "%.0f" % img_t, "%.0f ms" % (1000 * np.mean([s for _, s in img]))],
		 ["YOLO's sentence: \"" + scene + "\"", "%.0f" % txt_t, "%.0f ms" % (1000 * np.mean([s for _, s in txt]))]])


def main():

	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--only", default="vad,turn,wake,stt,llm,tts,yolo,gesture,vision")
	ap.add_argument("--images", default=None, help="folder of photos for yolo/gesture/vision")
	ap.add_argument("--weights", default="yolov10m.pt", help="YOLO weights (comma list)")
	ap.add_argument("--json", default=None)
	args = ap.parse_args()
	parts = set(args.only.split(","))

	heymiro = heymiro_config.load()
	sp = Speech(heymiro.route("stt").base_url)
	rng = np.random.default_rng(7)
	out = {}
	if "vad" in parts:
		bench_vad(sp, rng, out)
	if "turn" in parts:
		bench_turn(sp, out)
	if "wake" in parts:
		bench_wake(sp, rng, out)
	if "stt" in parts:
		bench_stt(sp, rng, out)
	if "llm" in parts:
		bench_llm(heymiro, out)
	if "tts" in parts:
		bench_tts(sp, out)
	images = load_images(args.images) if args.images else []
	if images:
		weights = []
		for w in args.weights.split(","):
			path = os.path.join(heymiro.cache_dir, "weights", w)
			weights.append(path if os.path.isfile(path) else w)
		if "yolo" in parts:
			bench_yolo(images, out, weights)
		if "gesture" in parts:
			bench_gesture(images, out, heymiro.path(heymiro.cfg.get("scene.gesture_model")))
		if "vision" in parts:
			bench_vision(heymiro, images, out)
	if args.json:
		with open(args.json, "w") as f:
			json.dump(out, f, indent=1, default=float)


if __name__ == "__main__":
	main()
