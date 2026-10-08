#!/usr/bin/python3
#
#	Estimate the tempo (BPM) of the songs in assets/songs so that
#	config/songs.yaml can carry a tempo for every bundled track.
#	(Spotify's audio-analysis endpoint is no longer available to new
#	apps, so the tempo is measured once, offline, and stored.)
#
#	usage: tools/estimate_bpm.py [file.mp3 ...]
#
#	Needs ffmpeg on the PATH and numpy; runs without ROS.
#

import os
import subprocess
import sys

import numpy as np

RATE = 11025
HOP = 256


def decode(path, seconds=90):

	# decode to mono 16-bit at RATE using ffmpeg
	cmd = ["ffmpeg", "-v", "error", "-i", path, "-t", str(seconds),
		"-f", "s16le", "-ac", "1", "-ar", str(RATE), "-"]
	raw = subprocess.check_output(cmd)
	return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def onset_envelope(x):

	# spectral flux of a short-time magnitude spectrum
	win = np.hanning(1024).astype(np.float32)
	n = 1 + (len(x) - 1024) // HOP
	frames = np.stack([x[i * HOP:i * HOP + 1024] * win for i in range(n)])
	mag = np.log1p(np.abs(np.fft.rfft(frames, axis=1)))
	flux = np.maximum(np.diff(mag, axis=0), 0.0).sum(axis=1)
	flux -= flux.mean()
	return flux


def estimate_bpm(x, lo=70.0, hi=180.0):

	env = onset_envelope(x)
	ac = np.correlate(env, env, mode="full")[len(env) - 1:]
	fps = float(RATE) / HOP
	lags = np.arange(len(ac))
	bpm = np.zeros(len(ac))
	bpm[1:] = 60.0 * fps / lags[1:]
	mask = (bpm >= lo) & (bpm <= hi)

	# weight towards ~120 BPM (log-gaussian), as perceptual tempo
	# estimators do, to resolve half/double tempo ambiguity
	weight = np.exp(-0.5 * (np.log2(np.maximum(bpm, 1e-6) / 120.0) / 0.9) ** 2)
	score = np.where(mask, ac * weight, -np.inf)
	lag = int(np.argmax(score))

	# refine the peak with parabolic interpolation (lags are coarse)
	a, b, c = ac[lag - 1], ac[lag], ac[lag + 1]
	den = a - 2.0 * b + c
	frac = 0.5 * (a - c) / den if den != 0 else 0.0
	return 60.0 * fps / (lag + frac)


def main():

	files = sys.argv[1:]
	if not files:
		here = os.path.dirname(os.path.abspath(__file__))
		songs = os.path.join(here, "..", "assets", "songs")
		files = sorted(os.path.join(songs, f) for f in os.listdir(songs) if f.endswith(".mp3"))
	for path in files:
		bpm = estimate_bpm(decode(path))
		print("%-28s %6.1f" % (os.path.basename(path), bpm))


if __name__ == "__main__":
	main()
