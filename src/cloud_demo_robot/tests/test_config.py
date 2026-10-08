#
#	Tests for core/heymiro_config.py: settings layers and HEYMIRO_SET,
#	secret precedence (environment > $HEYMIRO_SECRETS_FILE >
#	~/.config/heymiro/secrets.env, empty = unset), API route overrides,
#	feature switches, and that no warning or status line ever shows a
#	secret value. Uses temporary directories, a fake environment dict
#	and a temporary $HOME; the real environment is never read.
#

import _testpath

import contextlib
import io
import os
import shutil
import tempfile
import unittest
from unittest import mock

import heymiro_config
from heymiro_config import HeyMiroConfig


# distinctive values, so a leak is easy to spot
# the robot copy's hearing settings (the sim copy's config differs; the
# same tests run in both copies)
ROBOT_MICS = "hearing.input_topic=sensors/mics;hearing.touch_from_mics=true"

SECRETS = {
	"OPENAI_API_KEY": "sk-openai-SECRET-1111",
	"HEYMIRO_LLM_API_KEY": "llm-SECRET-2222",
	"HEYMIRO_STT_API_KEY": "stt-SECRET-3333",
	"ELEVENLABS_API_KEY": "eleven-SECRET-4444",
	"PICOVOICE_ACCESS_KEY": "pico-SECRET-5555",
	"PICOVOICE_ACCESS_KEY_HEY_MIRO": "pico-hey-SECRET-6666",
	"PICOVOICE_ACCESS_KEY_DANCE_MIRO": "pico-dance-SECRET-7777",
	"PICOVOICE_ACCESS_KEY_STOP_MIRO": "pico-stop-SECRET-8888",
	"SPOTIFY_CLIENT_ID": "spotify-id-SECRET-9999",
	"SPOTIFY_CLIENT_SECRET": "spotify-SECRET-0000",
}



class ConfigTestCase(unittest.TestCase):

	def setUp(self):

		# a private project root (copy of the real heymiro.yaml) and a
		# private $HOME, so neither a developer's heymiro.local.yaml nor
		# their ~/.config/heymiro/secrets.env can affect the tests
		self.tmp = tempfile.mkdtemp(prefix="heymiro_cfg_")
		self.root = os.path.join(self.tmp, "project")
		os.makedirs(os.path.join(self.root, "config"))
		shutil.copy(os.path.join(_testpath.ROOT, "config", "heymiro.yaml"),
			os.path.join(self.root, "config", "heymiro.yaml"))
		self.home = os.path.join(self.tmp, "home")
		os.makedirs(self.home)
		patcher = mock.patch.dict(os.environ, {"HOME": self.home})
		patcher.start()
		self.addCleanup(patcher.stop)
		self.addCleanup(shutil.rmtree, self.tmp, True)

	def write(self, path, text, mode=0o600):

		d = os.path.dirname(path)
		if not os.path.isdir(d):
			os.makedirs(d)
		with open(path, "w") as f:
			f.write(text)
		os.chmod(path, mode)
		return path

	def default_secrets_file(self, text, mode=0o600):

		return self.write(os.path.join(self.home, ".config", "heymiro", "secrets.env"), text, mode)

	def load(self, environ=None):

		return HeyMiroConfig(self.root, dict(environ or {}))



class TestSecretPrecedence(ConfigTestCase):

	def setUp(self):

		ConfigTestCase.setUp(self)
		self.default_secrets_file("# default file\nOPENAI_API_KEY=from-default-file\n")
		self.named = self.write(os.path.join(self.tmp, "named.env"), "OPENAI_API_KEY=from-named-file\n")

	def key(self, environ):

		return self.load(environ).route("llm").api_key

	def test_environment_wins(self):

		self.assertEqual(self.key({"OPENAI_API_KEY": "from-env", "HEYMIRO_SECRETS_FILE": self.named}), "from-env")

	def test_named_file_beats_default_file(self):

		self.assertEqual(self.key({"HEYMIRO_SECRETS_FILE": self.named}), "from-named-file")

	def test_default_file(self):

		self.assertEqual(self.key({}), "from-default-file")

	def test_empty_environment_value_is_unset(self):

		self.assertEqual(self.key({"OPENAI_API_KEY": "", "HEYMIRO_SECRETS_FILE": self.named}), "from-named-file")
		self.assertEqual(self.key({"OPENAI_API_KEY": "   "}), "from-default-file")
		# an empty HEYMIRO_SECRETS_FILE is unset too (no warning)
		cfg = self.load({"HEYMIRO_SECRETS_FILE": ""})
		self.assertEqual(cfg.route("llm").api_key, "from-default-file")
		self.assertFalse([w for w in cfg.warnings if "HEYMIRO_SECRETS_FILE" in w])

	def test_missing_named_file_warns_and_falls_back(self):

		cfg = self.load({"HEYMIRO_SECRETS_FILE": os.path.join(self.tmp, "nope.env")})
		self.assertEqual(cfg.route("llm").api_key, "from-default-file")
		self.assertTrue([w for w in cfg.warnings if "HEYMIRO_SECRETS_FILE not found" in w])

	def test_empty_value_in_file_is_unset(self):

		named = self.write(os.path.join(self.tmp, "empty.env"), "ELEVENLABS_API_KEY=\nOPENAI_API_KEY=  \n")
		cfg = self.load({"HEYMIRO_SECRETS_FILE": named})
		self.assertFalse(cfg.has_secret("ELEVENLABS_API_KEY"))
		# the files are merged key by key and an empty value is "unset",
		# so the value from ~/.config/heymiro/secrets.env shows through
		self.assertEqual(cfg.route("llm").api_key, "from-default-file")
		self.assertEqual(cfg.secret_status()["ELEVENLABS_API_KEY"], "missing")

	def test_custom_route_never_gets_the_openai_key(self):

		cfg = self.load({"HEYMIRO_SECRETS_FILE": self.named, "HEYMIRO_LLM_BASE_URL": "http://localhost:11434/v1"})
		self.assertEqual(cfg.route("llm").base_url, "http://localhost:11434/v1")
		self.assertIsNone(cfg.route("llm").api_key)
		self.assertTrue(cfg.route("llm").available)
		self.assertEqual(cfg.route("stt").api_key, "from-named-file")

	def test_secret_status_only_says_set_or_missing(self):

		cfg = self.load({"HEYMIRO_SECRETS_FILE": self.named, "ELEVENLABS_API_KEY": "x"})
		status = cfg.secret_status()
		self.assertEqual(set(status), set(heymiro_config.SECRET_NAMES))
		self.assertEqual(status["OPENAI_API_KEY"], "set")
		self.assertEqual(status["ELEVENLABS_API_KEY"], "set")
		self.assertEqual(status["SPOTIFY_CLIENT_ID"], "missing")
		self.assertTrue(set(status.values()) <= set(["set", "missing"]))

	def test_picovoice_specific_key_beats_generic(self):

		cfg = self.load({"PICOVOICE_ACCESS_KEY": "generic", "PICOVOICE_ACCESS_KEY_HEY_MIRO": "hey"})
		self.assertEqual(cfg.picovoice_key("PICOVOICE_ACCESS_KEY_HEY_MIRO"), "hey")
		self.assertEqual(cfg.picovoice_key("PICOVOICE_ACCESS_KEY_STOP_MIRO"), "generic")
		self.assertEqual(cfg.picovoice_key(None), "generic")
		self.assertIsNone(self.load({}).picovoice_key("PICOVOICE_ACCESS_KEY_HEY_MIRO"))



class TestSettings(ConfigTestCase):

	def test_defaults_from_yaml(self):

		cfg = self.load().cfg
		self.assertEqual(cfg.llm.model, "gpt-4o")
		# robot copy: sensors/mics; sim copy: host/mics (the tests run in both)
		self.assertIn(cfg.hearing.input_topic, ["sensors/mics", "host/mics"])
		self.assertEqual(cfg.get("hearing.touch_noise.zcr_min"), 190)
		self.assertIsNone(cfg.get("hearing.no_such_key"))
		self.assertEqual(cfg.get("no.such.key", 3), 3)
		with self.assertRaises(AttributeError):
			cfg.no_such_section
		with self.assertRaises(AttributeError):
			cfg.llm = None

	def test_layers_and_set_overrides(self):

		self.write(os.path.join(self.root, "config", "heymiro.local.yaml"),
			"voice:\n  volume: 0.7\nllm:\n  model: local-model\n")
		overlay = self.write(os.path.join(self.tmp, "overlay.yaml"),
			"voice:\n  volume: 0.6\n  music_volume: 0.3\n")
		cfg = self.load({
			"HEYMIRO_CONFIG": overlay,
			"HEYMIRO_SET": "voice.volume=0.5; hearing.input_topic=host/mics ;llm.event_name=Open Day;bad-item",
		}).cfg
		self.assertEqual(cfg.voice.volume, 0.5)			# HEYMIRO_SET
		self.assertEqual(cfg.voice.music_volume, 0.3)	# HEYMIRO_CONFIG over local
		self.assertEqual(cfg.llm.model, "local-model")	# local over heymiro.yaml
		self.assertEqual(cfg.llm.max_history_turns, 10)	# untouched default
		self.assertEqual(cfg.hearing.input_topic, "host/mics")
		self.assertEqual(cfg.llm.event_name, "Open Day")
		self.assertEqual(cfg.voice.backend, "host")

	def test_missing_overlay_warns(self):

		cfg = self.load({"HEYMIRO_CONFIG": os.path.join(self.tmp, "nope.yaml")})
		self.assertTrue([w for w in cfg.warnings if "HEYMIRO_CONFIG" in w])

	def test_parse_set_overrides(self):

		out = heymiro_config.parse_set_overrides("a.b=1;a.c=true;d=host/mics;e=;;x")
		self.assertEqual(out, {"a": {"b": 1, "c": True}, "d": "host/mics", "e": ""})
		self.assertEqual(heymiro_config.parse_set_overrides(None), {})

	def test_paths(self):

		cfg = self.load({"HEYMIRO_CACHE_DIR": os.path.join(self.tmp, "cache")})
		self.assertEqual(cfg.path("config/songs.yaml"), os.path.join(self.root, "config", "songs.yaml"))
		self.assertEqual(cfg.path("/abs/file"), "/abs/file")
		self.assertEqual(cfg.path("~/x"), os.path.join(self.home, "x"))
		self.assertIsNone(cfg.path(None))
		self.assertEqual(cfg.cache_path("weights", "y.pt"), os.path.join(self.tmp, "cache", "weights", "y.pt"))

	def test_missing_settings_file_raises(self):

		os.remove(os.path.join(self.root, "config", "heymiro.yaml"))
		with self.assertRaises(IOError):
			self.load()



class TestRoutes(ConfigTestCase):

	def test_default_routes(self):

		cfg = self.load({"OPENAI_API_KEY": "k"})
		llm = cfg.route("llm")
		self.assertEqual(llm.base_url, "https://api.openai.com/v1")
		self.assertTrue(llm.is_default_url)
		self.assertTrue(llm.available)
		self.assertEqual(llm.timeout_s, 25)
		self.assertEqual(cfg.route("stt").timeout_s, 20)
		self.assertEqual(cfg.route("tts").base_url, "https://api.elevenlabs.io")
		sp = cfg.route("spotify")
		self.assertEqual(sp.base_url, "https://api.spotify.com/v1")
		self.assertEqual(sp.token_url, "https://accounts.spotify.com/api/token")
		with self.assertRaises(KeyError):
			cfg.route("nope")

	def test_custom_openai_route_needs_no_key(self):

		cfg = self.load({"OPENAI_BASE_URL": "http://localhost:8000/v1/"})
		for name in ("llm", "stt"):
			r = cfg.route(name)
			self.assertEqual(r.base_url, "http://localhost:8000/v1")
			self.assertFalse(r.is_default_url)
			self.assertTrue(r.available)
			self.assertIsNone(r.api_key)
		self.assertTrue(cfg.features.conversation)
		# the default URL with a trailing slash is still the default
		self.assertTrue(self.load({"OPENAI_BASE_URL": "https://api.openai.com/v1/"}).route("llm").is_default_url)

	def test_per_route_overrides(self):

		cfg = self.load({
			"OPENAI_API_KEY": "shared",
			"HEYMIRO_LLM_BASE_URL": "http://llm.local/v1",
			"HEYMIRO_LLM_API_KEY": "llm-key",
			"HEYMIRO_STT_API_KEY": "stt-key",
		})
		llm = cfg.route("llm")
		stt = cfg.route("stt")
		self.assertEqual((llm.base_url, llm.api_key), ("http://llm.local/v1", "llm-key"))
		self.assertEqual((stt.base_url, stt.api_key), ("https://api.openai.com/v1", "stt-key"))

	def test_llm_override_without_stt_key_disables_conversation(self):

		cfg = self.load({"HEYMIRO_LLM_BASE_URL": "http://llm.local/v1"})
		self.assertTrue(cfg.route("llm").available)
		self.assertFalse(cfg.route("stt").available)
		self.assertFalse(cfg.features.conversation)

	def test_other_route_overrides(self):

		cfg = self.load({
			"ELEVENLABS_BASE_URL": "http://tts.local/",
			"SPOTIFY_API_BASE_URL": "http://sp.local/v1/",
			"SPOTIFY_TOKEN_URL": "http://sp.local/token",
			"SPOTIFY_CLIENT_ID": "id",
			"SPOTIFY_CLIENT_SECRET": "secret",
		})
		tts = cfg.route("tts")
		self.assertEqual(tts.base_url, "http://tts.local")
		self.assertFalse(tts.is_default_url)
		self.assertTrue(tts.available)
		sp = cfg.route("spotify")
		self.assertEqual(sp.base_url, "http://sp.local/v1")
		self.assertEqual(sp.token_url, "http://sp.local/token")
		self.assertEqual((sp.client_id, sp.client_secret), ("id", "secret"))
		self.assertTrue(sp.available)

	def test_routes_from_secrets_file(self):

		self.default_secrets_file("OPENAI_BASE_URL=http://file.local/v1\nOPENAI_API_KEY=file-key\n")
		r = self.load().route("llm")
		self.assertEqual((r.base_url, r.api_key), ("http://file.local/v1", "file-key"))



class TestFeatures(ConfigTestCase):

	def test_everything_off_without_keys(self):

		cfg = self.load({"HEYMIRO_SET": "features.spotify=true;" + ROBOT_MICS})
		f = cfg.features
		self.assertFalse(f.conversation)
		self.assertFalse(f.wake_word)
		self.assertFalse(f.tts)
		self.assertFalse(f.vad_cobra)
		self.assertFalse(f.spotify)
		# features that need no key stay as configured
		self.assertTrue(f.dance)
		self.assertTrue(f.tricks)
		self.assertTrue(f.touch_reactions)
		text = "\n".join(cfg.warnings)
		for word in ("conversation disabled", "wake_word disabled", "tts disabled", "spotify disabled", "Cobra"):
			self.assertIn(word, text)

	def test_on_with_keys(self):

		f = self.load({"OPENAI_API_KEY": "a", "PICOVOICE_ACCESS_KEY": "b", "ELEVENLABS_API_KEY": "c"}).features
		self.assertTrue(f.conversation)
		self.assertTrue(f.wake_word)
		self.assertTrue(f.tts)
		self.assertTrue(f.vad_cobra)
		self.assertFalse(f.spotify)	# off in heymiro.yaml

	def test_keyword_specific_picovoice_key(self):

		cfg = self.load({"PICOVOICE_ACCESS_KEY_HEY_MIRO": "b"})
		self.assertTrue(cfg.features.wake_word)
		self.assertFalse(cfg.features.vad_cobra)	# Cobra needs the generic key

	def test_disabled_in_yaml_gives_no_warning(self):

		cfg = self.load({"HEYMIRO_SET": "features.conversation=false;features.wake_word=false;tts.provider=none;hearing.vad=energy"})
		self.assertFalse(cfg.features.conversation)
		self.assertFalse(cfg.features.tts)
		self.assertFalse(cfg.features.vad_cobra)
		self.assertEqual(cfg.warnings, [])

	def test_touch_reactions_need_miro_mics(self):

		self.assertFalse(self.load({"HEYMIRO_SET": "hearing.input_topic=host/mics"}).features.touch_reactions)
		self.assertFalse(self.load({"HEYMIRO_SET": "hearing.touch_from_mics=false"}).features.touch_reactions)

	def test_spotify_on_with_credentials(self):

		cfg = self.load({"HEYMIRO_SET": "features.spotify=true", "SPOTIFY_CLIENT_ID": "i", "SPOTIFY_CLIENT_SECRET": "s"})
		self.assertTrue(cfg.features.spotify)



class TestNoSecretLeaks(ConfigTestCase):

	def everything_printed(self, cfg):

		out = io.StringIO()
		with contextlib.redirect_stdout(out):
			cfg.log_warnings()
		parts = [out.getvalue()] + list(cfg.warnings) + cfg.status_lines()
		for name in ("llm", "stt", "tts", "spotify"):
			r = cfg.route(name)
			parts += [repr(r), str(r)]
		parts.append(repr(cfg.cfg))
		parts.append(repr(cfg.features))
		return "\n".join(parts)

	def assertNoSecret(self, text):

		for name, value in SECRETS.items():
			self.assertNotIn(value, text, name)

	def test_secrets_from_environment(self):

		env = dict(SECRETS)
		env["HEYMIRO_SET"] = "features.spotify=true"
		self.assertNoSecret(self.everything_printed(self.load(env)))

	def test_secrets_from_a_messy_file(self):

		# quotes, export, an unreadable-by-others warning and a broken
		# line all produce warnings; none of them may echo a value
		lines = ["# my keys", "export OPENAI_API_KEY=" + SECRETS["OPENAI_API_KEY"]]
		lines.append('ELEVENLABS_API_KEY="' + SECRETS["ELEVENLABS_API_KEY"] + '"')
		lines.append("PICOVOICE_ACCESS_KEY='" + SECRETS["PICOVOICE_ACCESS_KEY"] + "'")
		lines.append(SECRETS["SPOTIFY_CLIENT_SECRET"])	# no "=": a pasted bare key
		path = self.default_secrets_file("\n".join(lines) + "\n", mode=0o644)
		cfg = self.load({"HEYMIRO_SET": "features.spotify=true"})
		text = self.everything_printed(cfg)
		self.assertNoSecret(text)
		self.assertIn("chmod 600", text)
		self.assertIn("line(s) 2, 3, 4, 5", text)
		# the values were still read, with quotes and export removed
		self.assertEqual(cfg.route("llm").api_key, SECRETS["OPENAI_API_KEY"])
		self.assertEqual(cfg.route("tts").api_key, SECRETS["ELEVENLABS_API_KEY"])
		self.assertEqual(cfg.picovoice_key(), SECRETS["PICOVOICE_ACCESS_KEY"])
		values, problems = heymiro_config.read_env_file(path)
		self.assertEqual(problems, [2, 3, 4, 5])
		self.assertNotIn(SECRETS["SPOTIFY_CLIENT_SECRET"], values.values())

	def test_secrets_never_reach_settings(self):

		cfg = self.load(dict(SECRETS))
		self.assertNoSecret(repr(cfg.cfg.to_dict()))



if __name__ == "__main__":
	unittest.main()
