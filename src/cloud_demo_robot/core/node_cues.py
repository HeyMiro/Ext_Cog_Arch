#
#	Hey MiRo - NodeCues: attentional cues through the eyelids and ears.
#
#	While the forebrain is busy, the eyelids blink at a rate that suits
#	what MiRo is doing (more often while listening, less while watching
#	for hand signs) instead of the stock express routine; ears perk up
#	when MiRo hears its name. Replaces HRI'25 node_blink.py, whose modes
#	fell through into each other (every mode also ran the generic one)
#	and which wrote the cosmetic joints directly, fighting NodeExpress.
#	Here each mode is handled once and the eyelids go through
#	NodeExpress.override(), which expires on its own when we stop.
#
#	Modes (set by NodeForebrain every tick):
#	  ""            no override: the stock express eyelids run
#	  "listening"   eyes wide open, frequent blinks
#	  "thinking"    eyes a little narrowed, ordinary blinks
#	  "responding"  eyes open, frequent blinks
#	  "dancing"     the dance's eyelid openness, occasional blinks
#	  "tricks"      eyes wide open, rare blinks (watching the hands)
#

import random

import node


# per mode: (eyelid closure between blinks, chance of a blink per tick);
# the chances are HRI'25's randint(0, n) == 0 values
MODES = {
	"listening": (0.0, 1.0 / 21.0),
	"thinking": (0.3, 1.0 / 16.0),
	"responding": (0.0, 1.0 / 26.0),
	"dancing": (0.0, 1.0 / 31.0),
	"tricks": (0.0, 1.0 / 51.0),
}

# a blink keeps the lids closed for this many ticks, then open until
# BLINK_TICKS (HRI'25 timing at 50 Hz)
BLINK_CLOSED_TICKS = 3
BLINK_TICKS = 5

# ears forward for this long when MiRo hears its name
PERK_TICKS = 25

OWNER = "cues"



class NodeCues(node.Node):

	def __init__(self, sys, rng=None):

		node.Node.__init__(self, sys, "cues")

		self.rng = rng if rng is not None else random.Random()
		self.mode = ""
		self.blink_t = None          # ticks into the current blink, None = not blinking
		self.double = False          # a second blink follows this one
		self.dance_lids = None       # [l, r] closure from the dance, if any
		self.perk_ticks = 0

		try:
			self.double_blink_prob = float(self.pars.express.double_blink_prob)
		except AttributeError:
			self.double_blink_prob = 0.2

	def set_mode(self, mode):

		mode = mode or ""
		if mode not in MODES and mode != "":
			print("[cues] unknown mode '" + str(mode) + "'")
			mode = ""
		if mode != self.mode:
			self.mode = mode
			self.blink_t = None
			if mode != "dancing":
				self.dance_lids = None

	def set_dance_eyelids(self, lids):

		# the dance choreography's eyelid closure [l, r] for this tick
		self.dance_lids = [float(lids[0]), float(lids[1])]

	def perk(self):

		# "Hey MiRo": ears forward for a moment
		self.perk_ticks = PERK_TICKS

	def _blink_closure(self, base, chance):

		# returns this tick's closure from the base openness and blink state
		if self.blink_t is None:
			if self.rng.random() < chance:
				self.blink_t = 0
				self.double = self.rng.random() < self.double_blink_prob
			else:
				return base
		closed = self.blink_t < BLINK_CLOSED_TICKS
		self.blink_t += 1
		if self.blink_t >= BLINK_TICKS:
			if self.double:
				# second blink straight after the first
				self.double = False
				self.blink_t = 0
			else:
				self.blink_t = None
		return 1.0 if closed else base

	def eyelids(self):

		# [l, r] closure for this tick, or None in mode ""
		if self.mode == "":
			return None
		base, chance = MODES[self.mode]
		if self.mode == "dancing" and self.dance_lids is not None:
			left, right = self.dance_lids
		else:
			left = right = base
		closure = self._blink_closure(0.0, chance)
		if closure >= 1.0:
			return [1.0, 1.0]
		return [left, right]

	def tick(self):

		express = getattr(self.nodes, "express", None)
		if express is None:
			return

		lids = self.eyelids()
		if lids is not None:
			express.override("eyelids", lids, ttl_ticks=3, owner=OWNER)

		if self.perk_ticks > 0:
			self.perk_ticks -= 1
			# ear 0 = forward (the stock routine maps high valence there)
			express.override("ears", [0.0, 0.0], ttl_ticks=2, owner=OWNER)
