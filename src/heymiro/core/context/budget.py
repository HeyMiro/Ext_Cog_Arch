#
#	Hey MiRo - TokenBudget: keep every SLM request inside its context
#	window, with room for the person's words and the reply.
#
#	llama.cpp is started with --ctx-size 16384 --parallel 2, so each
#	request has n_ctx = 8192 tokens. A request is
#
#	  [system prompt + examples]   fixed, prompt-cached       ~1,900
#	  [history]                    this conversation's turns  grows
#	  [note + the person's words]  new every turn             capped
#	  [reply]                      generated                  llm.max_tokens
#
#	so the only part that grows is the history. The budget gives it
#
#	  n_ctx - prefix - reserve_reply - reserve_user - reserve_note - margin
#
#	and the person's words and the note are capped to their reserves.
#
#	No model calls and no network on the turn: the window size (GET
#	/props) and the exact prefix size (POST /apply-template + /tokenize)
#	are measured ONCE at start-up; that also gives this tokenizer's
#	characters per token, so every later count is an instant estimate.
#
#	History is trimmed with hysteresis (cloud/llm.py ChatMemory): when it
#	passes the high-water mark, the oldest turns go in one step down to
#	the low-water mark. Dropping one turn per turn instead would change
#	the start of the history every turn, and llama.cpp's prompt cache
#	would then re-process the whole history each time (seconds on a CPU);
#	this way the cache misses once per ~8 turns.
#

import math

DEFAULT_N_CTX = 8192
DEFAULT_CHARS_PER_TOKEN = 3.0     # safe for English (Qwen measures ~4)
ELLIPSIS = "…"


class TokenBudget(object):

	def __init__(self, n_ctx=DEFAULT_N_CTX, prefix_tokens=None, chars_per_token=DEFAULT_CHARS_PER_TOKEN,
			reserve_reply=200, reserve_user=150, reserve_note=250, margin=0.05, high_turns=16, low_turns=8,
			prefix_chars=0):

		self.n_ctx = int(n_ctx)
		self.chars_per_token = max(1.0, float(chars_per_token))
		# without a measured prefix, estimate it from its characters
		self.prefix_tokens = int(prefix_tokens) if prefix_tokens else self.estimate_chars(prefix_chars)
		self.reserve_reply = int(reserve_reply)
		self.reserve_user = int(reserve_user)
		self.reserve_note = int(reserve_note)
		self.margin = float(margin)
		self.high_turns = max(1, int(high_turns))
		self.low_turns = max(1, min(int(low_turns), self.high_turns))
		self.measured = prefix_tokens is not None

	# ------------------------------------------------------------ counting

	def estimate_chars(self, n):

		return int(math.ceil(max(0, int(n)) / self.chars_per_token))

	def estimate(self, text):

		return self.estimate_chars(len(str(text or "")))

	def messages_tokens(self, messages):

		# a few tokens per message for the chat template's role markers
		return sum(self.estimate(m.get("content", "")) + 4 for m in messages or ())

	# -------------------------------------------------------------- limits

	@property
	def history_tokens(self):

		"""Tokens the conversation history may use (the high-water mark)."""

		free = self.n_ctx - self.prefix_tokens - self.reserve_reply - self.reserve_user - self.reserve_note
		return max(0, int(free - self.margin * self.n_ctx))

	@property
	def low_tokens(self):

		return self.history_tokens // 2

	def over_high(self, n_turns, tokens):

		return n_turns > self.high_turns or tokens > self.history_tokens

	def under_low(self, n_turns, tokens):

		return n_turns <= self.low_turns and tokens <= self.low_tokens

	def fits(self, prompt_tokens):

		return prompt_tokens + self.reserve_reply <= self.n_ctx

	# ------------------------------------------------------------- capping

	def cap(self, text, tokens):

		"""text cut to about `tokens` tokens at a word boundary, with "…"."""

		text = " ".join(str(text or "").split())
		limit = int(tokens * self.chars_per_token)
		if len(text) <= limit:
			return text
		cut = text[:max(1, limit - 1)].rsplit(" ", 1)[0]
		return cut + ELLIPSIS

	def cap_user(self, text):

		return self.cap(text, self.reserve_user)

	def describe(self):

		return "n_ctx %d per request, prefix %d%s, history up to %d tokens / %d turns (trim to %d turns), " \
			"user %d, note %d, reply %d" % (self.n_ctx, self.prefix_tokens, "" if self.measured else " (estimated)",
			self.history_tokens, self.high_turns, self.low_turns, self.reserve_user, self.reserve_note,
			self.reserve_reply)


def pick_turns(turns, head=4, tail=8):

	"""
	The turns a summary should read from a long conversation: the first
	`head` (introductions, names) and the last `tail` (what is fresh).
	"""

	turns = list(turns)
	if len(turns) <= head + tail:
		return turns
	return turns[:head] + turns[-tail:]


def _base(url):

	url = str(url).rstrip("/")
	return url[:-3] if url.endswith("/v1") else url


def fetch_n_ctx(llm_url, timeout=2.0, session=None):

	"""llama.cpp's per-request context (GET /props), or None."""

	try:
		import requests
		http = session or requests
		props = http.get(_base(llm_url) + "/props", timeout=timeout).json()
		n = (props.get("default_generation_settings") or {}).get("n_ctx") or props.get("n_ctx")
		return int(n) if n else None
	except Exception:
		return None


def calibrate(llm_url, prefix_messages, timeout=3.0, session=None):

	"""
	Measure the window and the fixed prefix once: (n_ctx or None,
	prefix tokens or None, chars per token or None). Never raises.
	"""

	n_ctx = fetch_n_ctx(llm_url, timeout, session)
	try:
		import requests
		http = session or requests
		base = _base(llm_url)
		prompt = http.post(base + "/apply-template", json={"messages": list(prefix_messages)},
			timeout=timeout).json().get("prompt")
		tokens = http.post(base + "/tokenize", json={"content": prompt}, timeout=timeout).json().get("tokens")
		if prompt and tokens:
			return n_ctx, len(tokens), len(prompt) / float(len(tokens))
	except Exception:
		pass
	return n_ctx, None, None


def from_config(ccfg, llm_max_tokens=200, n_ctx=None, prefix_tokens=None, chars_per_token=None, prefix_chars=0):

	"""A TokenBudget from ai.local.context.budget (a config Section or dict) and measured values."""

	from context.harness import _setting
	get = lambda k, d: _setting(ccfg, "budget." + k, d)
	return TokenBudget(
		n_ctx=n_ctx or get("n_ctx", DEFAULT_N_CTX),
		prefix_tokens=prefix_tokens,
		chars_per_token=chars_per_token or DEFAULT_CHARS_PER_TOKEN,
		reserve_reply=int(llm_max_tokens or 200),
		reserve_user=get("reserve_user", 150),
		reserve_note=get("reserve_note", 250),
		high_turns=get("high_turns", 16),
		low_turns=get("low_turns", 8),
		prefix_chars=prefix_chars)
