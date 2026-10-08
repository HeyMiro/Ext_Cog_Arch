#!/usr/bin/env python3
#
#	Hey MiRo - make a pre-recorded voice clip with ElevenLabs.
#
#	Synthesises a phrase in MiRo's voice (tts.voice_id unless --voice-id)
#	and writes it as assets/clips/<slug>/<n>.wav (24 kHz mono 16-bit, the
#	format NodeVoice preloads), continuing the numbering of any takes
#	already there. ElevenLabs gives a slightly different reading every
#	time, so --count N makes N takes; ClipLibrary picks one at random.
#	It then prints the config/clips.yaml line that puts the new phrase
#	folder into a category.
#
#	The key and the route come from the environment / secrets file via
#	heymiro_config (ELEVENLABS_API_KEY, ELEVENLABS_BASE_URL); the key is
#	never printed. No ROS needed. Replaces HRI'25 core/make_clip.py
#	(hardcoded key, pydub, interactive ROS loop).
#
#	usage:
#	  make_clip.py "Hi there!" [--slug hi_there] [--count 3] [--category greeting]
#	  make_clip.py "Ouch!" --voice-id <id> --play
#

import argparse
import glob
import os
import re
import sys
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
CORE = os.path.join(os.path.dirname(HERE), "core")
if CORE not in sys.path:
	sys.path.insert(0, CORE)

import yaml

import audio_util
import heymiro_config
from cloud.worker import describe_error


CLIP_RATE = audio_util.VOICE_RATE
SLUG_MAX = 48



def slugify(text):

	# "Don't touch my ears, please!" -> "dont_touch_my_ears_please"
	text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("ascii")
	text = text.lower().replace("'", "")
	slug = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
	return slug[:SLUG_MAX].rstrip("_") or "clip"


def next_index(folder):

	# takes are 0.wav, 1.wav, ...; continue after the highest one
	highest = -1
	for path in glob.glob(os.path.join(folder, "*.wav")):
		stem = os.path.splitext(os.path.basename(path))[0]
		if stem.isdigit():
			highest = max(highest, int(stem))
	return highest + 1


def manifest_line(manifest, category, slug):

	# the clips.yaml line for category with slug appended (once)
	folders = [str(f) for f in ((manifest or {}).get(category) or [])]
	if slug not in folders:
		folders.append(slug)
	return category + ": [" + ", ".join(folders) + "]"


def load_manifest(path):

	try:
		with open(path, "r") as f:
			data = yaml.safe_load(f) or {}
		return data if isinstance(data, dict) else {}
	except (IOError, OSError, yaml.YAMLError):
		return {}


def play(pcm, rate):

	# optional preview on the host speaker (PortAudio may be missing)
	try:
		import sounddevice
		sounddevice.play(pcm, rate, blocking=True)
	except (ImportError, OSError) as e:
		print("[make_clip] cannot play (" + e.__class__.__name__ + "), listen to the wav instead")
	except Exception as e:
		print("[make_clip] playback failed (" + e.__class__.__name__ + ")")


def make_takes(tts, text, folder, count, voice_id=None, preview=False):

	# synthesise count takes into folder; returns [(path, seconds)]
	if not os.path.isdir(folder):
		os.makedirs(folder)
	written = []
	n = next_index(folder)
	for _ in range(count):
		pcm = tts.synthesize(text, voice_id)
		if pcm is None or len(pcm) == 0:
			raise RuntimeError("the TTS returned no audio")
		path = os.path.join(folder, str(n) + ".wav")
		with open(path, "wb") as f:
			f.write(audio_util.wav_bytes(pcm, CLIP_RATE))
		seconds = len(pcm) / float(CLIP_RATE)
		print("[make_clip] wrote " + path + " (%.2f s)" % seconds)
		if preview:
			play(pcm, CLIP_RATE)
		written.append((path, seconds))
		n += 1
	return written


def parse_args(argv):

	p = argparse.ArgumentParser(description="Make a Hey MiRo voice clip with ElevenLabs (24 kHz mono wav).")
	p.add_argument("text", nargs="+", help="what MiRo should say")
	p.add_argument("--slug", help="phrase folder name under the clips dir (default: from the text)")
	p.add_argument("--voice-id", help="ElevenLabs voice id (default: tts.voice_id from the config)")
	p.add_argument("--count", type=int, default=1, help="number of takes to make (default 1)")
	p.add_argument("--category", help="clips.yaml category to print the manifest line for (e.g. greeting)")
	p.add_argument("--out-dir", help="clips dir (default: clips.dir from the config, assets/clips)")
	p.add_argument("--play", action="store_true", help="play each take on the host speaker")
	args = p.parse_args(argv)
	if args.count < 1:
		p.error("--count must be at least 1")
	return args


def main(argv=None, heymiro=None, tts=None):

	args = parse_args(argv)
	text = " ".join(args.text).strip()
	slug = slugify(args.slug) if args.slug else slugify(text)

	heymiro = heymiro or heymiro_config.load()
	for w in heymiro.warnings:
		# only what matters here (the conversation / wake word ones do not)
		if "secrets" in w or w.startswith("tts"):
			print("WARN [heymiro] " + w)
	cfg = heymiro.cfg
	clips_dir = os.path.abspath(os.path.expanduser(args.out_dir)) if args.out_dir else \
		heymiro.path(cfg.get("clips.dir", "assets/clips"))

	if tts is None:
		route = heymiro.route("tts")
		if not route.api_key:
			print("[make_clip] ELEVENLABS_API_KEY is not set (environment or secrets file)")
			return 2
		from cloud.tts import TextToSpeech
		tts = TextToSpeech(route, cfg.get("tts"))

	voice_id = args.voice_id or cfg.get("tts.voice_id")
	print("[make_clip] \"" + text + "\" -> " + os.path.join(clips_dir, slug) + " (" + str(args.count) + " take(s))")
	try:
		make_takes(tts, text, os.path.join(clips_dir, slug), args.count, voice_id, args.play)
	except (ValueError, RuntimeError) as e:
		# raised locally (no voice id, empty audio, ffmpeg): safe to show
		print("[make_clip] failed: " + e.__class__.__name__ + " " + str(e)[:120])
		return 1
	except Exception as e:
		# HTTP / network errors: class and status only, never the message
		print("[make_clip] failed: " + describe_error(e))
		return 1

	# how to use it
	manifest = load_manifest(heymiro.path(cfg.get("clips.manifest", "config/clips.yaml")))
	print("\nadd the phrase folder to a category in config/clips.yaml, e.g.:")
	if args.category:
		print("  " + manifest_line(manifest, args.category, slug))
	else:
		print("  <category>: [..., " + slug + "]    (categories: " + ", ".join(sorted(manifest.keys())) + ")")
	return 0



if __name__ == "__main__":

	sys.exit(main())
