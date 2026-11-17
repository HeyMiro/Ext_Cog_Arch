#!/usr/bin/env python3
#
# Hey MiRo - paper figures, drawn as plain SVG (no dependencies).
#
#   python3 docs/figures/make_figures.py          # writes docs/figures/*.svg
#   docs/figures/render.bash                      # SVG -> PDF + PNG (headless Chrome)
#
# Every figure is laid out on a grid by hand, so the result is stable and
# editable: change a label here and re-run.

import os

HERE = os.path.dirname(os.path.abspath(__file__))
FONT = "Helvetica, Arial, 'Liberation Sans', sans-serif"

# palette (prints in greyscale: the fills differ in lightness too)
INK = "#222222"
MUTED = "#5b5b5b"
STOCK_FILL, STOCK_LINE = "#f1f1f1", "#8c8c8c"
NEW_FILL, NEW_LINE = "#e6effa", "#2c64ad"
NEW_BAND = "#f5f8fd"
OFF_FILL, OFF_LINE = "#e3f3ec", "#23845f"
ON_FILL, ON_LINE = "#fcefe1", "#bf6316"
BAND = "#fafafa"
BAND_LINE = "#d9d9d9"


def esc(s):
	return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class Svg(object):

	def __init__(self, w, h):
		self.w, self.h = w, h
		self.items = []

	def add(self, s):
		self.items.append(s)

	def rect(self, x, y, w, h, fill="#fff", stroke=INK, sw=1.0, rx=4, dash=None):
		d = ' stroke-dasharray="%s"' % dash if dash else ""
		self.add('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" rx="%s" fill="%s" stroke="%s" stroke-width="%s"%s/>'
			% (x, y, w, h, rx, fill, stroke, sw, d))

	def text(self, x, y, s, size=10, weight="normal", anchor="middle", color=INK, italic=False, rotate=None):
		st = ' font-style="italic"' if italic else ""
		rot = ' transform="rotate(%s %.1f %.1f)"' % (rotate, x, y) if rotate is not None else ""
		self.add('<text x="%.1f" y="%.1f" font-size="%s" font-weight="%s" text-anchor="%s" fill="%s"%s%s>%s</text>'
			% (x, y, size, weight, anchor, color, st, rot, esc(s)))

	def lines(self, x, y, rows, size=9.5, gap=None, color=INK, anchor="middle", weight="normal"):
		gap = gap or size * 1.3
		for i, r in enumerate(rows):
			self.text(x, y + i * gap, r, size=size, color=color, anchor=anchor, weight=weight)

	def box(self, x, y, w, h, title, rows=(), kind="stock", tsize=11, rsize=9.3):
		fill, line = {"stock": (STOCK_FILL, STOCK_LINE), "new": (NEW_FILL, NEW_LINE),
			"plain": ("#ffffff", INK), "off": (OFF_FILL, OFF_LINE), "on": (ON_FILL, ON_LINE)}[kind]
		self.rect(x, y, w, h, fill=fill, stroke=line, sw=1.1)
		n = len(rows)
		block = tsize + n * rsize * 1.3
		ty = y + (h - block) / 2 + tsize * 0.9
		self.text(x + w / 2, ty, title, size=tsize, weight="bold")
		self.lines(x + w / 2, ty + tsize * 0.55 + rsize * 1.15, rows, size=rsize, color="#333")

	def tag(self, x, y, label, kind):
		fill, line = (OFF_FILL, OFF_LINE) if kind == "off" else (ON_FILL, ON_LINE)
		w = 9 + len(label) * 5.4
		self.rect(x, y - 8.5, w, 11.5, fill=fill, stroke=line, sw=0.8, rx=5.5)
		self.text(x + w / 2, y, label, size=7.6, weight="bold", color=line)
		return w

	def arrow(self, pts, color=INK, dash=None, sw=1.2, head=True, tail=False):
		d = "M " + " L ".join("%.1f %.1f" % p for p in pts)
		da = ' stroke-dasharray="%s"' % dash if dash else ""
		mid = self._marker(color)
		me = ' marker-end="url(#%s)"' % mid if head else ""
		ms = ' marker-start="url(#%s)"' % mid if tail else ""
		self.add('<path d="%s" fill="none" stroke="%s" stroke-width="%s"%s%s%s stroke-linejoin="round"/>'
			% (d, color, sw, da, me, ms))

	_markers = {}

	def _marker(self, color):
		mid = "ah" + color.strip("#")
		self._markers[mid] = color
		return mid

	def label(self, x, y, s, size=8.4, color=MUTED, anchor="middle", bg=True, italic=True):
		if bg:
			w = len(s) * size * 0.5 + 6
			x0 = x - w / 2 if anchor == "middle" else (x - 3 if anchor == "start" else x - w + 3)
			self.add('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="#ffffff" fill-opacity="0.92"/>'
				% (x0, y - size + 1, w, size + 3))
		self.text(x, y, s, size=size, color=color, anchor=anchor, italic=italic)

	def save(self, name):
		defs = "".join(
			'<marker id="%s" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
			'orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="%s"/></marker>' % (k, v)
			for k, v in sorted(self._markers.items()))
		out = ('<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d" '
			'font-family="%s">\n<defs>%s</defs>\n<rect width="100%%" height="100%%" fill="#ffffff"/>\n%s\n</svg>\n'
			% (self.w, self.h, self.w, self.h, FONT, defs, "\n".join(self.items)))
		with open(os.path.join(HERE, name + ".svg"), "w") as f:
			f.write(out)
		print("wrote", name + ".svg")


# ---------------------------------------------------------------------------
# Figure 1: the layered architecture
# ---------------------------------------------------------------------------

def fig_architecture():

	s = Svg(1000, 668)
	Svg._markers = {}
	W = 155
	A, B, C = 138, 307, 476
	D, E = 657, 826
	cx = lambda x: x + W / 2
	CH = C + W + 13     # free vertical channel between C and D

	bands = [
		("P4", "Off-board", 18, 142, BAND),
		("P3", "Forebrain", 152, 350, NEW_BAND),
		("P2", "Brainstem", 360, 476, BAND),
		("P1", "Spinal cord", 486, 552, BAND),
		("", "Body", 562, 648, "#ffffff"),
	]
	for code, name, y0, y1, fill in bands:
		line = NEW_LINE if code == "P3" else BAND_LINE
		s.rect(10, y0, 980, y1 - y0, fill=fill, stroke=line, sw=0.8 if code != "P3" else 1.0, rx=6)
		my = (y0 + y1) / 2
		if code:
			s.text(56, my - 3, code, size=17, weight="bold", color=NEW_LINE if code == "P3" else MUTED)
			s.text(56, my + 13, name, size=9.5, color=NEW_LINE if code == "P3" else MUTED)
		else:
			s.text(56, my + 4, name, size=11, weight="bold", color=MUTED)
	s.text(56, 335, "Hey MiRo", size=8.5, weight="bold", color=NEW_LINE, italic=True)
	s.text(56, 462, "stock MiRo", size=8.5, color=MUTED, italic=True)

	# --- P4: the swappable model layer ---
	y4, h4 = 46, 82
	for x, title, off, on in ((A, "Speech-to-text", "Whisper v3-turbo", "OpenAI whisper-1"),
			(B, "Language model", "Qwen3.5-4B", "GPT-4o"),
			(C, "Text-to-speech", "Kokoro-82M", "ElevenLabs")):
		s.rect(x, y4, W, h4, fill="#ffffff", stroke=INK, sw=1.1)
		s.text(cx(x), y4 + 21, title, size=11, weight="bold")
		tw = s.tag(x + 10, y4 + 45, "offline", "off")
		s.text(x + 16 + tw, y4 + 45, off, size=9.2, anchor="start")
		tw = s.tag(x + 10, y4 + 66, "online", "on")
		s.text(x + 16 + tw, y4 + 66, on, size=9.2, anchor="start")
	s.arrow([(A + W, y4 + 30), (B, y4 + 30)])
	s.arrow([(B + W, y4 + 30), (C, y4 + 30)])

	# legend
	lx, ly = D + 8, 34
	s.text(lx, ly + 10, "Legend", size=9.5, weight="bold", anchor="start")
	s.rect(lx, ly + 20, 22, 13, fill=STOCK_FILL, stroke=STOCK_LINE, sw=1)
	s.text(lx + 30, ly + 30, "stock MiRo MDK module", size=9, anchor="start")
	s.rect(lx, ly + 41, 22, 13, fill=NEW_FILL, stroke=NEW_LINE, sw=1)
	s.text(lx + 30, ly + 51, "added by Hey MiRo", size=9, anchor="start")
	s.tag(lx, ly + 73, "offline", "off")
	s.text(lx + 50, ly + 73, "local model", size=9, anchor="start")
	s.tag(lx, ly + 93, "online", "on")
	s.text(lx + 50, ly + 93, "cloud service", size=9, anchor="start")
	x2 = lx + 160
	s.arrow([(x2, ly + 26), (x2 + 30, ly + 26)])
	s.text(x2 + 38, ly + 30, "data", size=9, anchor="start")
	s.arrow([(x2, ly + 47), (x2 + 30, ly + 47)], color=NEW_LINE, dash="4 3")
	s.text(x2 + 38, ly + 51, "modulation", size=9, anchor="start")
	s.arrow([(x2, ly + 68), (x2 + 30, ly + 68)], color=NEW_LINE, sw=1.6)
	s.text(x2 + 38, ly + 72, "action bid", size=9, anchor="start")

	# --- P3: the forebrain ---
	r1, r2, hr = 168, 264, 68
	s.box(A, r1, W, hr, "Hearing", ["VAD · wake words", "end-of-turn detection"], "new")
	s.box(B, r1, W, hr, "Dialogue", ["conversation state", "machine, streaming"], "new")
	s.box(C, r1, W, hr, "Voice", ["speech, clips, music", "mixed on one output"], "new")
	s.box(A, r2, W, hr, "Scene awareness", ["YOLOv10m objects", "MediaPipe gestures"], "new")
	s.box(B, r2, C + W - B, hr, "Forebrain integration",
		["events → intentions · one task slot", "top-down modulation of the brainstem"], "new")
	# behaviours, a tall box with four actions
	bh = r2 + hr - r1
	s.rect(D, r1, E + W - D, bh, fill="#ffffff", stroke=NEW_LINE, sw=1.1)
	s.text((D + E + W) / 2, r1 + 18, "Forebrain behaviours", size=11, weight="bold")
	s.text((D + E + W) / 2, r1 + 31, "new basal-ganglia actions", size=8.8, color=MUTED, italic=True)
	acts = [("Converse", "talk, face the user"), ("Dance", "music + choreography"),
		("Tricks", "gesture → trick"), ("Identify", "object game")]
	aw, ah = 147, 50
	for i, (t, sub) in enumerate(acts):
		ax = D + 10 + (i % 2) * (aw + 10)
		ay = r1 + 44 + (i // 2) * (ah + 12)
		s.box(ax, ay, aw, ah, t, [sub], "new", tsize=10.5, rsize=9)

	# P4 <-> P3
	s.arrow([(cx(A), r1), (cx(A), y4 + h4)])
	s.label(cx(A), (r1 + y4 + h4) / 2 + 4, "utterance audio")
	s.arrow([(cx(B), y4 + h4), (cx(B), r1)], tail=True)
	s.label(cx(B), (r1 + y4 + h4) / 2 + 4, "prompt ↑  JSON reply ↓")
	s.arrow([(cx(C), y4 + h4), (cx(C), r1)])
	s.label(cx(C), (r1 + y4 + h4) / 2 + 4, "speech audio")
	# inside P3
	s.arrow([(A + W, r1 + 26), (B, r1 + 26)])
	s.arrow([(B + W, r1 + 26), (C, r1 + 26)])
	s.arrow([(A + W - 22, r1 + hr), (A + W - 22, r1 + hr + 13), (B + 26, r1 + hr + 13), (B + 26, r2)])
	s.label((A + W + B) / 2 + 2, r1 + hr + 10, "wake word, touch noise", size=7.8, bg=True)
	s.arrow([(cx(B) + 30, r1 + hr), (cx(B) + 30, r2)], tail=True)
	s.arrow([(A + W, r2 + hr / 2), (B, r2 + hr / 2)])
	s.arrow([(C + W, r2 + hr / 2), (D, r2 + hr / 2)], color=NEW_LINE)
	s.label((C + W + D) / 2, r2 + hr / 2 - 6, "task", size=7.8, bg=False)

	# --- P2: brainstem ---
	y2, h2 = 380, 66
	s.box(A, y2, W, h2, "Spatial attention", ["salience: vision + sound", "orient · approach · flee"])
	s.box(B, y2, W, h2, "Affect", ["valence · arousal", "sleep oscillator"])
	s.box(C, y2, W, h2, "Expression", ["lights · ears · tail", "eyelids (SPG)"])
	s.box(D, y2, W, h2, "Basal ganglia", ["action selection,", "one winner at a time"])
	s.box(E, y2, W, h2, "Motor patterns", ["kinematics, gaze,", "wheels (MPG)"])
	s.arrow([(B + W, y2 + h2 / 2), (C, y2 + h2 / 2)])
	s.arrow([(D + W, y2 + h2 / 2), (E, y2 + h2 / 2)])
	sb = y2 + h2 + 13
	s.arrow([(A + W - 25, y2 + h2), (A + W - 25, sb), (D + 30, sb), (D + 30, y2 + h2)])
	s.label((B + C + W) / 2 + 40, sb + 3, "orient / approach bids", size=7.8)
	# P3 -> P2
	s.arrow([(cx(B), r2 + hr), (cx(B), y2)], color=NEW_LINE, dash="4 3")
	s.label(cx(B), (r2 + hr + y2) / 2 + 3, "affect target")
	s.arrow([(cx(C), r2 + hr), (cx(C), y2)], color=NEW_LINE, dash="4 3")
	s.label(cx(C), (r2 + hr + y2) / 2 + 3, "override")
	s.arrow([(cx(D), r2 + hr), (cx(D), y2)], color=NEW_LINE, sw=1.6)
	s.label(cx(D), (r2 + hr + y2) / 2 + 3, "bid 0.84")

	# --- P1 ---
	y1, h1 = 498, 42
	s.box(A, y1, B + W - A, h1, "Signal conditioning", ["touch · light · sound level · body state"], tsize=10.5)
	s.box(D, y1, W, h1, "Reflexes", ["cliff · freeze (halt)"], tsize=10.5)
	s.arrow([(cx(A) - 20, y1), (cx(A) - 20, y2 + h2)])
	s.arrow([(cx(B), y1), (cx(B), y2 + h2)])
	s.arrow([(D + W - 30, y1), (D + W - 30, y2 + h2)])

	# --- body ---
	yb, hb = 574, 58
	s.box(A, yb, B + W - A, hb, "Sensors",
		["4 microphones · 2 cameras · touch · light · cliff · IMU"], "plain", tsize=10.5)
	s.box(C, yb, E + W - C, hb, "Actuators",
		["wheels · neck · ears · tail · eyelids · LEDs · speaker"], "plain", tsize=10.5)
	s.text((A + E + W) / 2 + 60, 645, "MiRo-e robot, or the Gazebo simulator with the PC's microphone and speaker",
		size=8.4, color=MUTED, italic=True)
	s.arrow([(A + W + 7, yb), (A + W + 7, y1 + h1)])
	s.arrow([(cx(C), y2 + h2), (cx(C), yb)])
	s.arrow([(cx(E), y2 + h2), (cx(E), yb)])
	# speech output channel
	s.arrow([(C + W, r1 + hr - 14), (CH, r1 + hr - 14), (CH, yb)])
	s.label(CH, yb - 42, "speech", size=7.8)
	# audio and video buses on the left
	s.arrow([(A, yb + 44), (116, yb + 44), (116, r1 + 40), (A, r1 + 40)])
	s.label(116, 470, "audio", size=7.8, bg=True)
	s.arrow([(A, yb + 16), (127, yb + 16), (127, r2 + 44), (A, r2 + 44)])
	s.label(127, 400, "video", size=7.8, bg=True)
	s.save("fig1_architecture")


# ---------------------------------------------------------------------------
# Figure 2: the conversation pipeline, offline and online
# ---------------------------------------------------------------------------

def fig_pipeline():

	s = Svg(1000, 372)
	Svg._markers = {}
	X0, W, G = 92, 88, 11.5
	xs = [X0 + i * (W + G) for i in range(9)]
	stages = ["Micro-\nphones", "Front\nend", "Voice\nactivity", "Wake\nword", "End of\nturn",
		"Speech-\nto-text", "Language\nmodel", "Text-to-\nspeech", "Speaker"]
	off = [None, None, "Silero VAD v5", "Whisper +\nfuzzy match", "smart-turn\nv3.2",
		"Whisper\nv3-turbo", "Qwen3.5-4B\nllama.cpp", "Kokoro-82M", None]
	on = [None, None, "Picovoice\nCobra", "Picovoice\nPorcupine", "3 s of\nsilence",
		"OpenAI\nwhisper-1", "GPT-4o", "ElevenLabs", None]
	both = {0: "4 × MiRo mics\nor PC mic", 1: "20→16 kHz\n80 Hz HPF", 8: "MiRo or\nPC speaker"}
	YS, HS = 96, 50
	YO, YN, HC = 166, 222, 46

	# branch: camera -> YOLO -> scene sentence -> LLM
	s.box(xs[3], 22, W * 2 + G, 40, "Camera → YOLOv10m", ["\"I can see a cup and a person\""], "new", tsize=10, rsize=8.6)
	s.arrow([(xs[3] + 2 * W + G, 42), (xs[6] + 26, 42), (xs[6] + 26, YS)])
	s.label(xs[5] + W / 2 + 4, 38, "scene context", bg=False)
	# branch: LLM -> body
	s.box(xs[7], 22, W * 2 + G, 40, "Affect · basal ganglia", ["emotion, valence, arousal, actions"], "stock", tsize=10, rsize=8.6)
	s.arrow([(xs[6] + W - 22, YS), (xs[6] + W - 22, 42), (xs[7], 42)])

	for i, x in enumerate(xs):
		parts = stages[i].split("\n")
		s.rect(x, YS, W, HS, fill=NEW_FILL if 2 <= i <= 7 else "#ffffff", stroke=NEW_LINE if 2 <= i <= 7 else INK, sw=1.1)
		s.lines(x + W / 2, YS + HS / 2 + 4 - (len(parts) - 1) * 7.5, parts, size=10.5, gap=15, weight="bold")
		if i < 8:
			s.arrow([(x + W, YS + HS / 2), (x + W + G, YS + HS / 2)])
		if i in both:
			s.rect(x, YO, W, YN + HC - YO, fill="#fafafa", stroke=BAND_LINE, sw=1, dash="3 3")
			s.lines(x + W / 2, YO + 44, both[i].split("\n"), size=9)
			continue
		for (y, txt, kind) in ((YO, off[i], "off"), (YN, on[i], "on")):
			fill, line = (OFF_FILL, OFF_LINE) if kind == "off" else (ON_FILL, ON_LINE)
			s.rect(x, y, W, HC, fill=fill, stroke=line, sw=1)
			rows = txt.split("\n")
			s.lines(x + W / 2, y + HC / 2 - (len(rows) - 1) * 6 + 3.5, rows, size=9.2, gap=12)
	# row labels
	s.text(14, YS + 30, "Stage", size=11, weight="bold", anchor="start")
	s.tag(14, YO + 27, "offline", "off")
	s.text(14, YO + 42, "local, no keys", size=8, color=MUTED, anchor="start")
	s.tag(14, YN + 27, "online", "on")
	s.text(14, YN + 42, "HRI '25 cloud", size=8, color=MUTED, anchor="start")
	# brace: where each stage runs
	yb = YN + HC + 14
	for i0, i1, txt in ((2, 4, "always listening: every 32 ms frame"), (5, 7, "once per turn, streamed sentence by sentence")):
		x0, x1 = xs[i0], xs[i1] + W
		s.add('<path d="M %.1f %.1f v 6 h %.1f v -6" fill="none" stroke="%s" stroke-width="1"/>'
			% (x0, yb, x1 - x0, MUTED))
		s.text((x0 + x1) / 2, yb + 20, txt, size=8.8, color=MUTED, italic=True)
	# latency strip
	yl = 326
	s.rect(14, yl, 972, 36, fill="#ffffff", stroke=BAND_LINE, sw=1)
	s.text(26, yl + 15, "From the end of the user's speech to MiRo answering:", size=9.2, weight="bold", anchor="start")
	s.tag(420, yl + 15, "offline", "off")
	s.text(474, yl + 15, "end of turn 0.32 s + STT 0.09 s + LLM 1st sentence 0.44 s + TTS 0.05 s ≈ 0.9–1.2 s (RTX 4070 Ti)",
		size=9, anchor="start")
	s.tag(420, yl + 30, "online", "on")
	s.text(474, yl + 30, "3 s of silence + 0.5–5 s for the cloud services (HRI '25)", size=9, anchor="start")
	s.save("fig2_pipeline")


# ---------------------------------------------------------------------------
# Figure 3: the conversation state machine
# ---------------------------------------------------------------------------

def fig_dialogue():

	s = Svg(1000, 290)
	Svg._markers = {}
	names = ["IDLE", "FACING", "GREETING", "LISTENING", "THINKING", "RESPONDING"]
	does = ["explores as\nstock MiRo", "turns to find\nthe user", "greeting clip,\nears perk up",
		"nods at speech\nonset", "\"Hmm\" clip,\nwaiting tune", "speaks; emotion\npose + lights"]
	W, H, Y = 118, 44, 92
	xs = [22 + i * 166 for i in range(6)]
	for i, x in enumerate(xs):
		s.rect(x, Y, W, H, fill=STOCK_FILL if i == 0 else NEW_FILL,
			stroke=STOCK_LINE if i == 0 else NEW_LINE, sw=1.3, rx=22)
		s.text(x + W / 2, Y + 27, names[i], size=11, weight="bold")
		s.lines(x + W / 2, Y + H + 17, does[i].split("\n"), size=8.6, color=MUTED, gap=11)
	trans = ["\"Hey MiRo\"", "user found", "clip ends", "end of turn", "first sentence"]
	for i in range(5):
		x0, x1 = xs[i] + W, xs[i + 1]
		s.arrow([(x0, Y + H / 2), (x1, Y + H / 2)], sw=1.3)
		s.label((x0 + x1) / 2, Y + H / 2 - 7, trans[i], size=8.2, bg=False, color=INK)
	# reply spoken: RESPONDING -> LISTENING (above)
	xl, xr = xs[3] + W / 2 + 18, xs[5] + W / 2
	s.arrow([(xr, Y), (xr, Y - 40), (xl, Y - 40), (xl, Y)], sw=1.3)
	s.label((xl + xr) / 2, Y - 46, "reply spoken: listen for the next turn", size=8.4, bg=False, color=INK)
	# GOODBYE below LISTENING -> IDLE
	gx, gy = xs[2] + 20, 226
	s.rect(gx, gy, W, H, fill=NEW_FILL, stroke=NEW_LINE, sw=1.3, rx=22)
	s.text(gx + W / 2, gy + 27, "GOODBYE", size=11, weight="bold")
	s.arrow([(xs[3] + 30, Y + H + 32), (xs[3] + 30, gy + H / 2), (gx + W, gy + H / 2)], sw=1.3)
	s.label(xs[3] + 24, gy - 8, "no speech for 3 s", size=8.2, bg=False, color=INK, anchor="end")
	s.arrow([(gx, gy + H / 2), (xs[0] + W / 2, gy + H / 2), (xs[0] + W / 2, Y + H + 32)], sw=1.3)
	s.label((xs[0] + W / 2 + gx) / 2, gy + H / 2 - 6, "clip ends", size=8.2, bg=False, color=INK)
	# SUSPENDED below RESPONDING
	sx, sy = xs[5] - 20, 226
	s.rect(sx, sy, W + 30, H, fill=NEW_FILL, stroke=NEW_LINE, sw=1.3, rx=22)
	s.text(sx + (W + 30) / 2, sy + 27, "SUSPENDED", size=11, weight="bold")
	s.arrow([(xs[5] + W - 14, Y + H + 32), (xs[5] + W - 14, sy)], sw=1.3)
	s.label(xs[5] + W - 20, sy - 22, "dance / tricks /", size=8.2, bg=False, color=INK, anchor="end")
	s.label(xs[5] + W - 20, sy - 11, "identify requested", size=8.2, bg=False, color=INK, anchor="end")
	s.arrow([(sx, sy + H / 2), (xs[3] + W - 20, sy + H / 2), (xs[3] + W - 20, Y + H + 32)], sw=1.3)
	s.label((xs[3] + W + sx) / 2 - 10, sy + H / 2 - 6, "task ends: \"want to keep talking?\"", size=8.2, bg=False, color=INK)
	# stop
	s.text(14, 22, "\"Stop MiRo\" in any state → GOODBYE, then IDLE", size=8.8, anchor="start", color=MUTED, italic=True)
	s.save("fig3_dialogue")


# ---------------------------------------------------------------------------
# Figure 4: deployment - simulator / robot and offline / online
# ---------------------------------------------------------------------------

def fig_deployment():

	s = Svg(1000, 400)
	Svg._markers = {}
	# host PC
	s.rect(190, 20, 620, 360, fill=BAND, stroke=INK, sw=1.2, rx=8)
	s.text(204, 40, "Host PC (Linux, Docker)", size=11, weight="bold", anchor="start")
	# ROS container
	s.rect(206, 54, 330, 312, fill="#ffffff", stroke=NEW_LINE, sw=1.2, rx=6)
	s.text(220, 74, "ROS container", size=10.5, weight="bold", anchor="start")
	s.text(220, 88, "ROS Noetic · MiRo MDK · repository bind-mounted", size=8.4, anchor="start", color=MUTED)
	procs = [("main client", "50 Hz brain: P1–P3, hearing, dialogue, voice"),
		("camera clients", "faces, motion, AprilTags"),
		("mics client", "sound localisation"),
		("scene client", "YOLOv10m + MediaPipe (GPU if present)")]
	for i, (t, sub) in enumerate(procs):
		y = 100 + i * 40
		s.rect(220, y, 302, 33, fill=NEW_FILL if i in (0, 3) else STOCK_FILL,
			stroke=NEW_LINE if i in (0, 3) else STOCK_LINE, sw=1)
		s.text(230, y + 14, t, size=9.6, weight="bold", anchor="start")
		s.text(230, y + 27, sub, size=8.6, anchor="start", color="#333")
	# sim-only parts inside the container
	s.rect(220, 266, 302, 88, fill="#ffffff", stroke=OFF_LINE, sw=1, dash="4 3")
	s.text(230, 282, "--sim only", size=8.6, weight="bold", anchor="start", color=OFF_LINE)
	s.text(230, 300, "roscore + Gazebo 11 (MiRo model)", size=9, anchor="start")
	s.text(230, 316, "host audio bridge: PC mic → /miro/host/mics", size=9, anchor="start")
	s.text(230, 332, "PulseAudio: MiRo speaks through the PC speaker", size=9, anchor="start")
	s.text(230, 346, "same code; config/platform/sim.yaml", size=8.4, anchor="start", color=MUTED, italic=True)
	# local AI
	s.rect(556, 54, 240, 200, fill="#ffffff", stroke=OFF_LINE, sw=1.2, rx=6)
	s.tag(570, 74, "--offline", "off")
	s.text(632, 74, "docker/local-ai", size=10.5, weight="bold", anchor="start")
	s.rect(570, 88, 212, 60, fill=OFF_FILL, stroke=OFF_LINE, sw=1)
	s.text(580, 104, "llm :8080", size=9.6, weight="bold", anchor="start")
	s.text(580, 119, "llama.cpp, Qwen3.5-4B / 9B", size=8.8, anchor="start")
	s.text(580, 133, "JSON-schema replies, streamed", size=8.8, anchor="start")
	s.rect(570, 156, 212, 60, fill=OFF_FILL, stroke=OFF_LINE, sw=1)
	s.text(580, 172, "speech :8090", size=9.6, weight="bold", anchor="start")
	s.text(580, 187, "faster-whisper, Kokoro-82M,", size=8.8, anchor="start")
	s.text(580, 201, "smart-turn v3.2", size=8.8, anchor="start")
	s.text(676, 240, "GPU (or CPU) · no keys, no internet", size=8.4, color=MUTED, italic=True)
	s.rect(556, 266, 240, 100, fill="#ffffff", stroke=BAND_LINE, sw=1)
	s.text(566, 284, "Volumes", size=9.6, weight="bold", anchor="start")
	s.text(566, 301, "heymiro-models: pinned model files", size=8.6, anchor="start")
	s.text(566, 316, "heymiro-cache: YOLO weights, state", size=8.6, anchor="start")
	s.text(566, 331, "secrets.env (online only), read-only", size=8.6, anchor="start")
	s.text(566, 352, "one container, many terminals", size=8.4, anchor="start", color=MUTED, italic=True)
	# robot (left)
	s.rect(14, 100, 156, 120, fill="#ffffff", stroke=ON_LINE, sw=1.2, rx=6)
	s.tag(26, 120, "--robot-ip IP", "on")
	s.text(26, 142, "MiRo-e robot", size=10.5, weight="bold", anchor="start")
	s.lines(26, 160, ["ROS master on the robot", "sensors, motors, speaker", "(or --ext-mic: PC mic)"],
		size=8.8, anchor="start", gap=13)
	s.arrow([(170, 116), (220, 116)], tail=True, sw=1.3)
	s.label(195, 109, "Wi-Fi", size=8, bg=False)
	# cloud (right)
	s.rect(830, 100, 156, 120, fill="#ffffff", stroke=ON_LINE, sw=1.2, rx=6)
	s.tag(842, 120, "--online", "on")
	s.text(842, 142, "Cloud APIs", size=10.5, weight="bold", anchor="start")
	s.lines(842, 160, ["OpenAI (STT, GPT-4o)", "ElevenLabs (TTS)", "Picovoice keys (local", "Porcupine, Cobra)"],
		size=8.8, anchor="start", gap=13)
	s.arrow([(522, 112), (546, 112), (546, 34), (908, 34), (908, 100)], color=ON_LINE, sw=1.3)
	s.label(728, 30, "HTTPS + keys", size=8, bg=False, color=ON_LINE)
	s.arrow([(522, 120), (570, 120)], color=OFF_LINE, sw=1.3)
	s.arrow([(522, 124), (552, 124), (552, 186), (570, 186)], color=OFF_LINE, sw=1.3)
	s.save("fig4_deployment")


# ---------------------------------------------------------------------------
# Figure 5: the evaluation harness - what each level covers
# ---------------------------------------------------------------------------

def fig_harness():

	s = Svg(1000, 330)
	Svg._markers = {}
	cols = ["Front\nend", "Voice\nactivity", "Wake\nword", "End of\nturn", "Speech-\nto-text", "Language\nmodel",
		"Text-to-\nspeech", "Dialogue,\nbehaviours", "Scene\n(YOLO, gest.)"]
	X0, CW = 180, 58
	xs = [X0 + i * CW for i in range(len(cols))]
	YH = 22
	for i, c in enumerate(cols):
		a, b = c.split("\n")
		s.text(xs[i] + CW / 2, YH + 12, a, size=9, weight="bold")
		s.text(xs[i] + CW / 2, YH + 24, b, size=9, weight="bold")
		s.add('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="%s" stroke-width="0.7"/>'
			% (xs[i], YH + 34, xs[i], 300, BAND_LINE))
	s.add('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="%s" stroke-width="0.7"/>'
		% (xs[-1] + CW, YH + 34, xs[-1] + CW, 300, BAND_LINE))
	XR = xs[-1] + CW + 14
	s.text(XR, YH + 18, "Measures", size=9.5, weight="bold", anchor="start")
	s.text(XR + 146, YH + 18, "Last result (offline)", size=9.5, weight="bold", anchor="start")
	rows = [
		("Unit tests", "tests/, in CI on every push", [(0, 9)], "stub",
			["the logic of every module,", "no GPU or network"], ["302 / 302 pass"]),
		("Stage benchmark", "tools/bench_pipeline.py", [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (8, 9)], "bar",
			["each stage alone: F1, word", "error rate in noise, latency"], ["tables in", "TECHNICAL.md §17"]),
		("LLM evaluation", "tools/eval_dialogue.py", [(5, 6)], "bar",
			["40 scripted messages: valid", "JSON, action F1, latency"], ["40/40 valid, F1 0.98,", "0.46 s to 1st sentence"]),
		("Fake user", "tools/fake_user.py, in Gazebo", [(0, 8)], "e2e",
			["the whole robot: reaction", "time per spoken line"], ["MiRo reacts in", "0.5–1.8 s"]),
	]
	RH, Y0 = 60, 66
	for r, (name, tool, spans, kind, meas, res) in enumerate(rows):
		y = Y0 + r * RH
		if r % 2 == 0:
			s.rect(10, y, 980, RH, fill="#fafafa", stroke="none", rx=0)
		s.text(20, y + 26, name, size=10.5, weight="bold", anchor="start")
		s.text(20, y + 41, tool, size=8.6, anchor="start", color=MUTED)
		for (a, b) in spans:
			bx, bw = xs[a] + 4, (b - a) * CW - 8
			if kind == "stub":
				s.rect(bx, y + 18, bw, 24, fill="#ffffff", stroke=STOCK_LINE, sw=1, dash="4 3")
				s.text(bx + bw / 2, y + 34, "every module; models, ROS and keys replaced by stubs", size=8.6, color=MUTED)
			elif kind == "e2e":
				s.rect(bx, y + 18, bw, 24, fill=NEW_FILL, stroke=NEW_LINE, sw=1.1)
				s.text(bx + bw / 2, y + 34, "synthetic voice on /miro/host/mics → real pipeline → MiRo's reaction",
					size=8.6, color=NEW_LINE)
			else:
				s.rect(bx, y + 18, bw, 24, fill=OFF_FILL, stroke=OFF_LINE, sw=1.1)
		s.lines(XR, y + 26, meas, size=8.6, anchor="start", gap=12)
		s.lines(XR + 146, y + 26, res, size=8.6, anchor="start", gap=12, weight="bold")
	s.text(20, 318, "Test speech is synthesised by Kokoro in a voice different from MiRo's, so the right transcript is "
		"always known; bench_pipeline.py adds noise at 20, 10, 5 and 0 dB SNR.", size=8.6, anchor="start", color=MUTED, italic=True)
	s.save("fig5_harness")


if __name__ == "__main__":
	fig_architecture()
	fig_pipeline()
	fig_dialogue()
	fig_deployment()
	fig_harness()
