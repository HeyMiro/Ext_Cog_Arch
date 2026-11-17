#
#	Hey MiRo - LED colours and patterns.
#
#	MiRo has six body LEDs; output.illum takes six uint32 values packed
#	as 0xAARRGGBB (A = brightness). Patterns here return such 6-lists
#	and are pure functions of time, so they can be tested and replayed.
#
#	Ported from HRI'25 dance_illum.py / action_llm.set_colour_info, with
#	fixes:
#	  * one shared pack_rgb() (HRI formatted a hex string and re-parsed it)
#	  * transition() interpolates properly (HRI's second half used t/2
#	    and jumped; its "processing" pattern overwrote itself twice)
#	  * patterns are driven by the beat period 60 / bpm
#

import math


# HRI colour table (get_rgbs / set_colour_info)
COLOURS = {
	"red": (255, 0, 0),
	"orange": (255, 165, 0),
	"yellow": (255, 255, 0),
	"green": (0, 255, 64),
	"blue": (0, 153, 255),
	"purple": (255, 102, 255),
	"pink": (255, 105, 180),
	"white": (255, 255, 255),
	"black": (0, 0, 0),
}

# genre -> colours to dance with (HRI set_colours_by_genre; "dance"
# took HRI's "electronic" set, "disco" is new)
PALETTES = {
	"pop": [COLOURS[c] for c in ("red", "pink", "purple", "yellow", "orange", "white")],
	"disco": [COLOURS[c] for c in ("pink", "purple", "blue", "yellow", "white")],
	"soul": [COLOURS[c] for c in ("purple", "blue", "green")],
	"rock": [COLOURS[c] for c in ("black", "blue", "white")],
	"dance": [COLOURS[c] for c in ("blue", "red", "pink")],
	"default": [COLOURS[c] for c in ("red", "blue", "orange", "yellow", "purple", "green")],
}

# emotion of an LLM reply -> colour (HRI: happy green, sad blue, angry red)
EMOTION_COLOURS = {
	"happy": COLOURS["green"],
	"fine": COLOURS["white"],
	"angry": COLOURS["red"],
	"worried": COLOURS["orange"],
	"sad": COLOURS["blue"],
	"scared": COLOURS["purple"],
}

N_LEDS = 6



def _byte(x):

	x = float(x)
	if not math.isfinite(x):
		return 0
	return int(min(max(round(x), 0), 255))


def pack_rgb(r, g, b, a=255):

	# -> uint32 0xAARRGGBB as a plain int (fits UInt32MultiArray)
	return (_byte(a) << 24) | (_byte(r) << 16) | (_byte(g) << 8) | _byte(b)


def unpack_rgb(value):

	# 0xAARRGGBB -> (r, g, b, a)
	value = int(value)
	return ((value >> 16) & 255, (value >> 8) & 255, value & 255, (value >> 24) & 255)


def palette(genre):

	return PALETTES.get(str(genre).lower(), PALETTES["default"])


def colour(c):

	# a name ("red") or an (r, g, b) triple -> (r, g, b)
	if isinstance(c, str):
		return COLOURS.get(c.lower(), COLOURS["white"])
	return (c[0], c[1], c[2])


def solid(rgb, brightness=1.0):

	# all six LEDs the same colour; brightness scales the colour (the
	# alpha byte stays 255 so the stock LED driver behaves as usual)
	r, g, b = colour(rgb)
	k = min(max(float(brightness), 0.0), 1.0)
	value = pack_rgb(r * k, g * k, b * k)
	return [value] * N_LEDS


def transition(c0, c1, x):

	# linear blend between two colours, x = 0 -> c0, x = 1 -> c1;
	# returns [r, g, b] ints, continuous in x
	x = float(x)
	x = min(max(x if math.isfinite(x) else 0.0, 0.0), 1.0)
	a = colour(c0)
	b = colour(c1)
	return [_byte(a[i] + x * (b[i] - a[i])) for i in range(3)]


def _beat_period(bpm):

	try:
		bpm = float(bpm)
	except (TypeError, ValueError):
		bpm = 120.0
	if not math.isfinite(bpm) or bpm <= 0.0:
		bpm = 120.0
	return 60.0 / min(max(bpm, 40.0), 240.0)


def flash(colours, t, bpm):

	# hard switch to the next colour on every beat (HRI flashing_lights
	# with flash_length = one beat)
	colours = list(colours) or [COLOURS["white"]]
	n = int(math.floor(max(0.0, float(t)) / _beat_period(bpm)))
	return solid(colours[n % len(colours)])


def fade(colours, t, bpm, beats=2.0):

	# smooth ping-pong through the colours, one colour every `beats`
	# beats (HRI transition_lights, fixed); cosine easing so the LEDs
	# rest briefly on each colour
	colours = list(colours) or [COLOURS["white"]]
	if len(colours) == 1:
		return solid(colours[0])
	u = max(0.0, float(t)) / (float(beats) * _beat_period(bpm))
	n = int(math.floor(u))
	x = 0.5 - 0.5 * math.cos(math.pi * (u - n))
	rgb = transition(colours[n % len(colours)], colours[(n + 1) % len(colours)], x)
	return solid(rgb)


def thinking_pulse(t, period_s=3.0):

	# cloud call in progress (HRI "processing": blue -> yellow -> red),
	# as a continuous loop with a gentle breathing brightness
	cycle = [COLOURS["blue"], COLOURS["yellow"], COLOURS["red"]]
	u = (max(0.0, float(t)) / float(period_s)) * len(cycle)
	n = int(math.floor(u))
	rgb = transition(cycle[n % 3], cycle[(n + 1) % 3], u - n)
	breathe = 0.65 + 0.35 * math.cos(2.0 * math.pi * float(t) / float(period_s))
	return solid(rgb, breathe)


def scale(illum, k):

	# dim an illum 6-list by k (0..1), e.g. for a fade-out
	k = min(max(float(k), 0.0), 1.0)
	out = []
	for v in illum:
		r, g, b, a = unpack_rgb(v)
		out.append(pack_rgb(r * k, g * k, b * k, a))
	return out
