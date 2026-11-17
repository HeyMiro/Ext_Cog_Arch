#
#	Hey MiRo - EventLog: what MiRo's body did recently, for the SLM.
#
#	The forebrain and the actions add one short line when something
#	happens ("you danced to Dancing Queen by ABBA", "someone scratched
#	your tail"); the context harness renders the recent ones into the
#	turn's context, e.g.
#
#	  "Recently: you danced to Dancing Queen by ABBA (2 minutes ago);
#	   someone stroked you (3 times, just now)."
#
#	Lines are written in the second person, as the SLM reads them.
#	Thread-safe: added on the tick, read on the worker.
#

import collections
import threading
import time

DEFAULT_MAX_AGE_S = 600.0
DEFAULT_MAX_ITEMS = 4


def ago(seconds):

	"""A spoken-style age: "just now", "40 seconds ago", "2 minutes ago"."""

	s = max(0.0, float(seconds))
	if s < 25:
		return "just now"
	if s < 55:
		return "%d seconds ago" % int(round(s / 10.0) * 10)
	m = int(round(s / 60.0))
	if m < 60:
		return "a minute ago" if m <= 1 else "%d minutes ago" % m
	h = int(round(s / 3600.0))
	if h < 48:
		return "an hour ago" if h <= 1 else "%d hours ago" % h
	return "%d days ago" % int(round(s / 86400.0))


class EventLog(object):

	def __init__(self, maxlen=32, clock=None):

		self.clock = clock if clock is not None else time.time
		self.items = collections.deque(maxlen=int(maxlen))
		self.lock = threading.Lock()

	def add(self, kind, text, now=None):

		text = " ".join(str(text or "").split())
		if not text:
			return
		now = self.clock() if now is None else float(now)
		with self.lock:
			self.items.append((now, str(kind), text))

	def clear(self):

		with self.lock:
			self.items.clear()

	def recent(self, now=None, max_age_s=DEFAULT_MAX_AGE_S, max_items=DEFAULT_MAX_ITEMS):

		"""
		The newest distinct events within max_age_s, newest first, as
		rendered lines. Repeats of the same line are merged into one
		("..., 3 times, just now"), dated by the latest.
		"""

		now = self.clock() if now is None else float(now)
		with self.lock:
			items = list(self.items)
		merged = collections.OrderedDict()
		for t, kind, text in reversed(items):
			if now - t > max_age_s:
				break
			key = (kind, text)
			if key in merged:
				merged[key][1] += 1
			else:
				merged[key] = [t, 1]
		out = []
		for (kind, text), (t, n) in merged.items():
			when = ago(now - t)
			out.append(text + (" (%d times, %s)" % (n, when) if n > 1 else " (" + when + ")"))
			if len(out) >= max_items:
				break
		return out

	def describe(self, now=None, max_age_s=DEFAULT_MAX_AGE_S, max_items=DEFAULT_MAX_ITEMS):

		lines = self.recent(now, max_age_s, max_items)
		if not lines:
			return None
		return "Recently: " + "; ".join(lines) + "."

	def __len__(self):

		with self.lock:
			return len(self.items)
