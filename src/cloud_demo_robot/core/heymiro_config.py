#
#	Hey MiRo - settings, secrets and API routes for the forebrain.
#
#	Non-secret settings come from YAML (config/heymiro.yaml and overlays);
#	secrets and API routes (base URLs) come from the environment or from a
#	secrets file that lives OUTSIDE git. Secret values are never printed,
#	logged, stored on pars, or put on the ROS parameter server.
#
#	Precedence (highest first)
#
#	  settings: $HEYMIRO_SET ("a.b=c;d.e=f") > $HEYMIRO_CONFIG (YAML file)
#	            > config/heymiro.local.yaml > config/heymiro.yaml
#	  secrets:  environment > $HEYMIRO_SECRETS_FILE > ~/.config/heymiro/secrets.env
#
#	An empty value counts as unset. This module has no ROS dependency.
#

import copy
import os
import stat

import yaml



# project root is the folder holding config/, core/ and assets/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# every secret / route variable we understand (used for the status banner)
SECRET_NAMES = [
	"OPENAI_API_KEY",
	"OPENAI_BASE_URL",
	"HEYMIRO_LLM_API_KEY",
	"HEYMIRO_LLM_BASE_URL",
	"HEYMIRO_STT_API_KEY",
	"HEYMIRO_STT_BASE_URL",
	"ELEVENLABS_API_KEY",
	"ELEVENLABS_BASE_URL",
	"PICOVOICE_ACCESS_KEY",
	"PICOVOICE_ACCESS_KEY_HEY_MIRO",
	"PICOVOICE_ACCESS_KEY_DANCE_MIRO",
	"PICOVOICE_ACCESS_KEY_STOP_MIRO",
	"SPOTIFY_CLIENT_ID",
	"SPOTIFY_CLIENT_SECRET",
	"SPOTIFY_API_BASE_URL",
	"SPOTIFY_TOKEN_URL",
]

# default API routes
DEFAULT_ROUTES = {
	"OPENAI_BASE_URL": "https://api.openai.com/v1",
	"ELEVENLABS_BASE_URL": "https://api.elevenlabs.io",
	"SPOTIFY_API_BASE_URL": "https://api.spotify.com/v1",
	"SPOTIFY_TOKEN_URL": "https://accounts.spotify.com/api/token",
}

DEFAULT_SECRETS_FILE = os.path.join("~", ".config", "heymiro", "secrets.env")



class Section(object):

	"""
	Read-only attribute view of a nested dict, e.g. cfg.hearing.vad_threshold.
	Missing keys raise AttributeError; use get() for optional keys.
	"""

	def __init__(self, data):

		object.__setattr__(self, "_data", data if data is not None else {})

	def __getattr__(self, name):

		try:
			value = self._data[name]
		except KeyError:
			raise AttributeError("config has no key '" + name + "'")
		return Section(value) if isinstance(value, dict) else value

	def __setattr__(self, name, value):

		raise AttributeError("config is read-only")

	def __getitem__(self, name):

		return self.__getattr__(name)

	def __contains__(self, name):

		return name in self._data

	def get(self, key, default=None):

		# dotted keys are allowed: cfg.get("hearing.touch_noise.zcr_min")
		node = self._data
		for part in key.split("."):
			if not isinstance(node, dict) or part not in node:
				return default
			node = node[part]
		return Section(node) if isinstance(node, dict) else node

	def keys(self):

		return self._data.keys()

	def items(self):

		for k in self._data:
			yield k, self.__getattr__(k)

	def to_dict(self):

		return copy.deepcopy(self._data)

	def __repr__(self):

		return "Section(" + repr(self._data) + ")"



class Route(object):

	"""
	Where (and with which key) to reach one cloud service.
	The key is never included in repr() / str().
	"""

	def __init__(self, name, base_url, api_key, timeout_s=20.0, is_default_url=True):

		self.name = name
		self.base_url = base_url
		self.api_key = api_key
		self.timeout_s = timeout_s
		self.is_default_url = is_default_url

	@property
	def available(self):

		# a key is required for the public endpoints; a custom route
		# (e.g. a local OpenAI-compatible server) may not need one
		return bool(self.api_key) or not self.is_default_url

	def __repr__(self):

		key = "set" if self.api_key else "missing"
		return "Route(" + self.name + ", " + str(self.base_url) + ", key " + key + ")"

	__str__ = __repr__



def deep_merge(base, overlay):

	# merge overlay into a copy of base (dicts merge, everything else replaces)
	out = copy.deepcopy(base)
	for k, v in (overlay or {}).items():
		if isinstance(v, dict) and isinstance(out.get(k), dict):
			out[k] = deep_merge(out[k], v)
		else:
			out[k] = copy.deepcopy(v)
	return out


def parse_set_overrides(text):

	# "hearing.input_topic=host/mics;voice.volume=0.5" -> nested dict
	out = {}
	for item in (text or "").split(";"):
		item = item.strip()
		if not item or "=" not in item:
			continue
		key, value = item.split("=", 1)
		value = yaml.safe_load(value.strip()) if value.strip() else ""
		node = out
		parts = [p for p in key.strip().split(".") if p]
		for part in parts[:-1]:
			node = node.setdefault(part, {})
		if parts:
			node[parts[-1]] = value
	return out


def read_env_file(path):

	"""
	Read a Docker-style env file (KEY=value per line, # comments on their
	own line). Quotes and a leading "export " are tolerated but reported.
	Returns (values, problem_line_numbers). Values are never printed.
	"""

	values = {}
	problems = []
	with open(path, "r") as f:
		for n, line in enumerate(f, 1):
			line = line.strip()
			if not line or line.startswith("#"):
				continue
			if line.startswith("export "):
				line = line[len("export "):].strip()
				problems.append(n)
			if "=" not in line:
				problems.append(n)
				continue
			key, value = line.split("=", 1)
			key = key.strip()
			value = value.strip()
			if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
				value = value[1:-1]
				problems.append(n)
			values[key] = value
	return values, problems



class HeyMiroConfig(object):

	def __init__(self, root=None, environ=None):

		self.root = os.path.abspath(root or PROJECT_ROOT)
		self.environ = os.environ if environ is None else environ
		self.warnings = []

		# settings
		self.cfg = Section(self._load_settings())

		# secrets (file part; environment is consulted live)
		self._file_secrets = self._load_secret_file()

		# cache for downloaded weights etc.
		cache = self._env("HEYMIRO_CACHE_DIR") or os.path.join("~", ".cache", "heymiro")
		self.cache_dir = os.path.abspath(os.path.expanduser(cache))

		# effective feature switches
		self.features = Section(self._compute_features())

	# ---------------------------------------------------------------- settings

	def _load_settings(self):

		data = {}
		layers = [
			os.path.join(self.root, "config", "heymiro.yaml"),
			os.path.join(self.root, "config", "heymiro.local.yaml"),
		]
		extra = self._env("HEYMIRO_CONFIG")
		if extra:
			layers.append(os.path.expanduser(extra))
		for path in layers:
			if os.path.isfile(path):
				with open(path, "r") as f:
					layer = yaml.safe_load(f) or {}
				if not isinstance(layer, dict):
					raise ValueError("config file is not a mapping: " + path)
				data = deep_merge(data, layer)
			elif path == layers[0]:
				raise IOError("missing settings file: " + path)
			elif path == extra:
				self.warnings.append("HEYMIRO_CONFIG file not found: " + path)
		data = deep_merge(data, parse_set_overrides(self._env("HEYMIRO_SET")))
		return data

	def path(self, p):

		# resolve a configured path: absolute and ~ paths as-is,
		# anything else relative to the project root
		if p is None:
			return None
		p = os.path.expanduser(str(p))
		if os.path.isabs(p):
			return p
		return os.path.normpath(os.path.join(self.root, p))

	def cache_path(self, *parts):

		return os.path.join(self.cache_dir, *parts)

	def persona(self):

		# raw persona template text (string.Template, $vars)
		with open(self.path(self.cfg.llm.persona_file), "r") as f:
			return f.read()

	# ----------------------------------------------------------------- secrets

	def _env(self, name):

		value = self.environ.get(name)
		if value is None:
			return None
		value = value.strip()
		return value if value else None

	def _load_secret_file(self):

		# ~/.config/heymiro/secrets.env, overlaid key by key with the file
		# named by $HEYMIRO_SECRETS_FILE (the one mounted by run_docker.bash)
		merged = {}
		named = self._env("HEYMIRO_SECRETS_FILE")
		candidates = [os.path.expanduser(DEFAULT_SECRETS_FILE)]
		if named:
			candidates.append(os.path.expanduser(named))
		for path in candidates:
			merged.update(self._read_one_secret_file(path, path == candidates[-1] and bool(named)))
		return merged

	def _read_one_secret_file(self, path, required):

		if not os.path.isfile(path):
			if required:
				self.warnings.append("HEYMIRO_SECRETS_FILE not found: " + path)
			return {}
		try:
			mode = os.stat(path).st_mode
			if mode & (stat.S_IRWXG | stat.S_IRWXO):
				self.warnings.append("secrets file " + path + " is readable by other users; run: chmod 600 " + path)
			values, problems = read_env_file(path)
		except (IOError, OSError) as e:
			self.warnings.append("cannot read secrets file " + path + " (" + e.__class__.__name__ + ")")
			return {}
		if problems:
			self.warnings.append("secrets file " + path + ": quotes/export on line(s) " +
				", ".join(str(n) for n in sorted(set(problems))) + " (docker --env-file keeps them literally)")
		return dict((k, v.strip()) for k, v in values.items() if v.strip())

	def _lookup(self, name):

		# environment first, then the secrets file; empty means unset
		value = self._env(name)
		if value is None:
			value = self._file_secrets.get(name)
		return value

	def secret(self, name):

		# as _lookup(), but routes fall back to their public default
		value = self._lookup(name)
		if value is None:
			value = DEFAULT_ROUTES.get(name)
		return value

	def has_secret(self, name):

		return self._lookup(name) is not None

	def secret_status(self):

		return dict((name, "set" if self.has_secret(name) else "missing") for name in SECRET_NAMES)

	def picovoice_key(self, specific_env=None):

		# a keyword-specific key (custom .ppn files are tied to the
		# Picovoice account that trained them) or the generic key
		if specific_env and self.has_secret(specific_env):
			return self.secret(specific_env)
		if self.has_secret("PICOVOICE_ACCESS_KEY"):
			return self.secret("PICOVOICE_ACCESS_KEY")
		return None

	def route(self, name):

		cfg = self.cfg
		if name in ("llm", "stt"):
			# per-route override (HEYMIRO_LLM_* / HEYMIRO_STT_*) or the shared OPENAI_* route
			prefix = "HEYMIRO_" + name.upper() + "_"
			custom_base = self._lookup(prefix + "BASE_URL")
			base = (custom_base or self.secret("OPENAI_BASE_URL")).rstrip("/")
			key = self._lookup(prefix + "API_KEY")
			if key is None and (custom_base is None or custom_base.rstrip("/") == self.secret("OPENAI_BASE_URL").rstrip("/")):
				# share the OpenAI key only with the OpenAI route itself, never
				# with a different (e.g. local) server given by HEYMIRO_*_BASE_URL
				key = self._lookup("OPENAI_API_KEY")
			timeout = cfg.get(name + ".timeout_s", 20.0)
			return Route(name, base, key, timeout, base == DEFAULT_ROUTES["OPENAI_BASE_URL"])
		if name == "tts":
			base = self.secret("ELEVENLABS_BASE_URL")
			key = self.secret("ELEVENLABS_API_KEY")
			return Route(name, base.rstrip("/"), key, cfg.get("tts.timeout_s", 20.0), base.rstrip("/") == DEFAULT_ROUTES["ELEVENLABS_BASE_URL"])
		if name == "spotify":
			base = self.secret("SPOTIFY_API_BASE_URL")
			route = Route(name, base.rstrip("/"), None, 10.0, True)
			route.token_url = self.secret("SPOTIFY_TOKEN_URL")
			route.client_id = self.secret("SPOTIFY_CLIENT_ID")
			route.client_secret = self.secret("SPOTIFY_CLIENT_SECRET")
			route.api_key = route.client_secret
			return route
		raise KeyError("unknown route " + name)

	# ---------------------------------------------------------------- features

	def _compute_features(self):

		want = self.cfg.get("features")
		want = want.to_dict() if want is not None else {}
		have = dict(want)

		def off(feature, reason):
			if have.get(feature):
				self.warnings.append(feature + " disabled: " + reason)
			have[feature] = False

		if not self.route("llm").available or not self.route("stt").available:
			off("conversation", "OPENAI_API_KEY not set (or HEYMIRO_LLM_*/HEYMIRO_STT_* route)")

		keys = [self.picovoice_key(w.get("access_key_env")) for w in (self.cfg.get("hearing.wake_words") or [])]
		if not any(keys):
			off("wake_word", "PICOVOICE_ACCESS_KEY not set (use the core/forebrain/command topic instead)")

		have["tts"] = self.cfg.get("tts.provider", "none") == "elevenlabs"
		if have["tts"] and not self.route("tts").available:
			off("tts", "ELEVENLABS_API_KEY not set (replies are printed, clips still play)")

		have["vad_cobra"] = self.cfg.get("hearing.vad", "energy") == "cobra"
		if have["vad_cobra"] and not self.picovoice_key():
			have["vad_cobra"] = False
			self.warnings.append("Cobra VAD unavailable without PICOVOICE_ACCESS_KEY; using the energy VAD")

		if have.get("spotify"):
			r = self.route("spotify")
			if not (r.client_id and r.client_secret):
				off("spotify", "SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET not set (song manifest only)")

		if have.get("touch_reactions"):
			if not self.cfg.get("hearing.touch_from_mics", False):
				have["touch_reactions"] = False
			elif self.cfg.get("hearing.input_topic") != "sensors/mics":
				have["touch_reactions"] = False

		return have

	def log_warnings(self, printer=print):

		for w in self.warnings:
			printer("WARN [heymiro] " + w)

	def status_lines(self):

		lines = []
		for name, status in sorted(self.secret_status().items()):
			lines.append("  %-34s %s" % (name, status))
		on = [k for k, v in self.features.items() if v is True]
		lines.append("  features on: " + ", ".join(sorted(on)))
		return lines



def load(root=None, environ=None):

	return HeyMiroConfig(root, environ)
