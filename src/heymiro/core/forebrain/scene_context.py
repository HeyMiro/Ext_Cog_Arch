#
#	Hey MiRo - scene context for the conversation.
#
#	While MiRo talks, the scene client runs YOLO ("context" mode) on the
#	left eye camera and publishes the objects it sees. This module turns
#	the latest message into ONE short sentence that is added to the
#	turn's context, e.g.
#
#	  "Your camera sees: a person in front of you, a cup on your left."
#
#	Why text and not the image: a camera frame costs a vision-language
#	model hundreds of image tokens per turn (measured in docs/TECHNICAL.md)
#	and adds prompt-processing time to every answer; YOLO already runs
#	for the object game and gives class, confidence and position in
#	~4 ms on the GPU, which become ~19 text tokens. The trade-off is that
#	MiRo only knows the 80 COCO classes, not colours, text or what a
#	drawing shows (the LLM is a vision-language model, so sending a frame
#	for such a question would be a possible extension; not done here).
#
#	Pure functions (no ROS), so they are unit tested.
#

import time

# only objects confirmed in k of the last n frames are described (the
# scene client applies identify.confirm), at most this many
MAX_OBJECTS = 4

# a message older than this is not "what MiRo sees now"
MAX_AGE_S = 3.0

NUMBERS = {2: "two", 3: "three", 4: "four", 5: "five"}

PLURALS = {"person": "people", "mouse": "mice", "knife": "knives", "sheep": "sheep", "skis": "skis",
	"scissors": "scissors", "bus": "buses", "wine glass": "wine glasses", "couch": "couches",
	"toothbrush": "toothbrushes", "sandwich": "sandwiches", "bench": "benches"}


def _article(label, count=1):

	label = str(label).replace("_", " ")
	if count > 1:
		plural = PLURALS.get(label, label + "s")
		return NUMBERS.get(count, "several") + " " + plural
	return ("an " if label[:1] in "aeiou" else "a ") + label


def _where(box):

	# box = [x0, y0, x1, y1] normalised; MiRo's left eye camera looks
	# forward, so the image x position is left/right from MiRo's view
	if not box or len(box) != 4:
		return ""
	cx = 0.5 * (box[0] + box[2])
	area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
	if cx < 0.33:
		side = "on your left"
	elif cx > 0.67:
		side = "on your right"
	else:
		side = "in front of you"
	if area > 0.25:
		side = "right " + side if side != "in front of you" else "right in front of you"
	return side


def describe_scene(state_objects, now=None, max_age_s=MAX_AGE_S, max_objects=MAX_OBJECTS):

	"""
	state_objects: (receive time, parsed core/scene/objects dict) as kept
	in the main client's state, or None. Returns one sentence, or None if
	there is nothing (fresh) to say.
	"""

	if not state_objects:
		return None
	t, msg = state_objects
	now = time.time() if now is None else now
	if msg is None or now - float(t) > max_age_s:
		return None
	confirmed = list(msg.get("confirmed") or [])
	if not confirmed:
		return None
	# position from the best detection of each confirmed label
	best = {}
	count = {}
	for o in msg.get("objects") or []:
		label = o.get("label")
		if label not in confirmed:
			continue
		count[label] = count.get(label, 0) + 1
		if label not in best or o.get("score", 0) > best[label].get("score", 0):
			best[label] = o
	ordered = sorted(confirmed, key=lambda l: -float(best.get(l, {}).get("score", 0.0)))[:max_objects]
	parts = []
	for label in ordered:
		where = _where(best.get(label, {}).get("box"))
		n = count.get(label, 1)
		# the position of one object only makes sense for a single one
		where = where if n == 1 else ""
		parts.append(_article(label, n) + (" " + where if where else ""))
	return "Your camera sees: " + ", ".join(parts) + "."
