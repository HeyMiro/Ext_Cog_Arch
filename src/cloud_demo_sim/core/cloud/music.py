#
#	Hey MiRo - songs to dance to.
#
#	SongLibrary  the songs in config/songs.yaml (file, title, artist,
#	             aliases, bpm, genre) with fuzzy matching of what the
#	             person asked for ("dance to Dancing Queen", "ABBA")
#	decode_mp3   mp3 -> int16 mono PCM via an ffmpeg subprocess (call it
#	             on the CloudWorker: decoding a song takes ~0.5 s)
#	SpotifyClient optional song lookup (client credentials) for title,
#	             artist, genre and, where the app still may, the tempo
#
#	The HRI'25 dance looked every song up on Spotify and downloaded its
#	30 s preview. Spotify no longer gives new apps previews or audio
#	analysis, so the songs and their BPMs live in songs.yaml and the
#	Spotify lookup is optional (features.spotify, off by default); it
#	never downloads audio.
#

import difflib
import os
import random
import re
import shutil
import subprocess
import threading
import time

import numpy as np
import yaml

from .worker import HttpError, describe_error



# fuzzy match score needed to accept a song (difflib ratio, 0..1)
MATCH_THRESHOLD = 0.6

# words that carry no information about which song is meant
FILLER_WORDS = set(("the", "song", "by", "please", "play", "dance", "to", "a", "some",
	"music", "track", "tune", "called", "can", "you", "lets", "let's", "something"))

# Spotify genre words -> our light/dance palettes (choreography.lights);
# checked in the HRI decide_genre() order (our extra disco/dance/house
# words first, so "dance pop" stays pop); within one name the last match wins
GENRE_WORDS = (
	("soul", "soul"),
	("disco", "disco"),
	("dance", "dance"),
	("house", "dance"),
	("pop", "pop"),
	("rock", "rock"),
	("metal", "rock"),
	("hip-hop", "dance"),
	("hip hop", "dance"),
	("classical", "default"),
	("electr", "dance"),
)



def normalise(text):

	# lower case, letters/digits only, single spaces
	text = str(text or "").lower().replace("&", " and ")
	text = re.sub(r"[^a-z0-9]+", " ", text)
	return " ".join(text.split())


def ffmpeg_decode(data=None, path=None, rate=24000, max_s=None, timeout_s=60.0):

	"""
	Decode audio bytes (data) or a file (path) to int16 mono at rate,
	using ffmpeg. Raises RuntimeError if ffmpeg is missing or fails.
	"""

	exe = shutil.which("ffmpeg")
	if exe is None:
		raise RuntimeError("ffmpeg not found (apt install ffmpeg)")
	cmd = [exe, "-nostdin", "-v", "error", "-i", path if path is not None else "pipe:0"]
	if max_s is not None:
		cmd += ["-t", "%.3f" % float(max_s)]
	cmd += ["-f", "s16le", "-ac", "1", "-ar", str(int(rate)), "pipe:1"]
	try:
		proc = subprocess.run(cmd, input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
			timeout=timeout_s)
	except subprocess.TimeoutExpired:
		raise RuntimeError("ffmpeg timed out")
	if proc.returncode != 0:
		raise RuntimeError("ffmpeg failed (exit " + str(proc.returncode) + ")")
	out = proc.stdout[:len(proc.stdout) // 2 * 2]
	return np.frombuffer(out, dtype="<i2").astype(np.int16)


def decode_mp3(path, rate=24000, max_s=None):

	return ffmpeg_decode(path=path, rate=rate, max_s=max_s)


def genre_from_spotify(genres):

	# port of HRI action_dance.decide_genre(): the first Spotify genre
	# string that mentions a known style decides, e.g. Sam Cooke's
	# ["classic soul", "soul", "vocal jazz"] -> "soul"
	for name in genres or []:
		name = str(name).lower()
		found = None
		for word, genre in GENRE_WORDS:
			if word in name:
				found = genre
		if found is not None:
			return found
	return "default"



class Song(object):

	def __init__(self, id, path, title, artist="", aliases=None, bpm=None, genre="default"):

		self.id = id
		self.path = path
		self.title = title
		self.artist = artist
		self.aliases = list(aliases or [])
		self.bpm = bpm
		self.genre = genre

	def names(self):

		# everything a person might call this song, normalised
		names = [self.title, self.artist, self.id.replace("_", " ")] + self.aliases
		names.append(self.title + " " + self.artist)
		return [n for n in (normalise(x) for x in names) if n]

	def __repr__(self):

		return "Song(%s: %r by %r, %s bpm, %s)" % (self.id, self.title, self.artist, self.bpm, self.genre)



class SongLibrary(object):

	def __init__(self, cfg, manifest=None):

		# cfg is the HeyMiroConfig; manifest overrides dance.songs_manifest
		self.songs = []
		default_bpm = float(cfg.cfg.get("dance.default_bpm", 120.0))
		path = cfg.path(manifest or cfg.cfg.get("dance.songs_manifest", "config/songs.yaml"))
		with open(path, "r") as f:
			data = yaml.safe_load(f) or {}
		for entry in data.get("songs") or []:
			try:
				song = Song(
					str(entry["id"]),
					cfg.path(entry["file"]),
					str(entry.get("title") or entry["id"]),
					str(entry.get("artist") or ""),
					[str(a) for a in (entry.get("aliases") or [])],
					float(entry.get("bpm") or default_bpm),
					str(entry.get("genre") or "default"))
			except (KeyError, TypeError, ValueError):
				print("[music] skipping a malformed entry in " + path)
				continue
			if not os.path.isfile(song.path):
				# skip it, so find() never returns a song we cannot play
				print("[music] song file missing, skipped: " + song.path)
				continue
			self.songs.append(song)

	def find(self, query):

		"""
		Best fuzzy match for what the person asked for, or None.
		A name contained in the query ("dancing queen by abba please")
		wins outright; otherwise difflib similarity over the title,
		artist, aliases and id must reach MATCH_THRESHOLD.
		"""

		q = normalise(query)
		if not q:
			return None
		core = " ".join(w for w in q.split() if w not in FILLER_WORDS)
		padded = " " + q + " "
		best_score = 0.0
		best = None
		for song in self.songs:
			for name in song.names():
				score = difflib.SequenceMatcher(None, core or q, name).ratio()
				score = max(score, difflib.SequenceMatcher(None, q, name).ratio())
				if len(name) >= 4 and (" " + name + " ") in padded:
					# whole name inside the query; longer names more specific
					score = max(score, 0.9 + 0.1 * min(1.0, len(name) / float(len(q))))
				if score > best_score:
					best_score = score
					best = song
		return best if best_score >= MATCH_THRESHOLD else None

	def random(self, rng=None):

		if not self.songs:
			return None
		return (rng or random).choice(self.songs)

	def get(self, song_id):

		for song in self.songs:
			if song.id == song_id:
				return song
		return None



class SpotifyClient(object):

	"""
	Optional song lookup with the client-credentials flow (no user
	login). Call lookup() on the CloudWorker. The audio-features
	endpoint (tempo) answers 403/404 for apps created after Nov 2024;
	the first such answer is remembered and the endpoint not asked again.
	"""

	def __init__(self, route, session=None):

		import requests
		self.route = route
		self.session = session or requests.Session()
		self.timeout = (5.0, float(getattr(route, "timeout_s", 10.0) or 10.0))
		self.lock = threading.Lock()
		self.token = None
		self.token_expiry = 0.0
		self.features_blocked = False
		self.cache = {}

	@property
	def available(self):

		return bool(getattr(self.route, "client_id", None) and getattr(self.route, "client_secret", None))

	def _get_token(self):

		with self.lock:
			if self.token and time.monotonic() < self.token_expiry:
				return self.token
			r = self.session.post(self.route.token_url, data={"grant_type": "client_credentials"},
				auth=(self.route.client_id, self.route.client_secret), timeout=self.timeout)
			if r.status_code != 200:
				raise HttpError("spotify token", r.status_code)
			data = r.json()
			self.token = data["access_token"]
			# renew a minute early
			self.token_expiry = time.monotonic() + max(0.0, float(data.get("expires_in", 3600)) - 60.0)
			return self.token

	def _get(self, path, params=None):

		# GET an API path; a 401 means the token expired early: renew once
		for attempt in (0, 1):
			headers = {"Authorization": "Bearer " + self._get_token()}
			r = self.session.get(self.route.base_url + path, params=params, headers=headers,
				timeout=self.timeout)
			if r.status_code == 401 and attempt == 0:
				with self.lock:
					self.token = None
				continue
			return r

	def lookup(self, query):

		"""
		Find a track: {title, artist, genre, bpm or None}, or None if
		nothing was found or the lookup failed (failures are printed
		as class name + HTTP status only).
		"""

		key = normalise(query)
		if not key or not self.available:
			return None
		if key in self.cache:
			return self.cache[key]
		try:
			result = self._lookup(query)
		except Exception as e:
			print("[music] spotify lookup failed: " + describe_error(e))
			return None
		self.cache[key] = result
		return result

	def _lookup(self, query):

		r = self._get("/search", {"q": query, "type": "track", "limit": 1})
		if r.status_code != 200:
			raise HttpError("spotify search", r.status_code)
		items = (r.json().get("tracks") or {}).get("items") or []
		if not items:
			return None
		track = items[0]
		artists = track.get("artists") or [{}]
		info = {
			"title": track.get("name") or "",
			"artist": artists[0].get("name") or "",
			"genre": "default",
			"bpm": None,
		}

		# genre from the (first) artist, as the HRI code did
		artist_id = artists[0].get("id")
		if artist_id:
			r = self._get("/artists/" + artist_id)
			if r.status_code == 200:
				info["genre"] = genre_from_spotify(r.json().get("genres"))

		# tempo, where Spotify still allows it
		track_id = track.get("id")
		if track_id and not self.features_blocked:
			r = self._get("/audio-features/" + track_id)
			if r.status_code == 200:
				tempo = r.json().get("tempo")
				info["bpm"] = float(tempo) if tempo else None
			elif r.status_code in (403, 404):
				self.features_blocked = True
				print("[music] spotify audio-features unavailable (HTTP " + str(r.status_code) + "); using songs.yaml BPMs")

		return info
