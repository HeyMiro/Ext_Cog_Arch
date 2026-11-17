#
#	Wiring checks that the unit tests of single modules cannot catch:
#	every node the main client creates must also be ticked (a node that is
#	never ticked silently loses its per-tick behaviour, e.g. the hearing
#	node's self-hearing guard), and the TTS content-type rules.
#

import ast
import os
import unittest

import _testpath


def demo_nodes_methods():

	path = os.path.join(_testpath.CORE, "client_demo.py")
	tree = ast.parse(open(path).read())
	for node in ast.walk(tree):
		if isinstance(node, ast.ClassDef) and node.name == "DemoNodes":
			return dict((f.name, f) for f in node.body if isinstance(f, ast.FunctionDef))
	raise AssertionError("DemoNodes not found")


def self_attrs(func, pattern):

	# names X in statements like self.X = Node...(...) / self.X.tick()
	found = []
	for node in ast.walk(func):
		if pattern == "assign" and isinstance(node, ast.Assign):
			for t in node.targets:
				if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self":
					if isinstance(node.value, ast.Call) and getattr(node.value.func, "id", "").startswith("Node"):
						found.append(t.attr)
		if pattern == "tick" and isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
			if node.func.attr == "tick" and isinstance(node.func.value, ast.Attribute):
				found.append(node.func.value.attr)
	return found


class TestMainClientWiring(unittest.TestCase):

	def test_every_forebrain_node_is_ticked(self):

		methods = demo_nodes_methods()
		created = self_attrs(methods["instantiate"], "assign")
		ticked = self_attrs(methods["tick"], "tick")
		# stock nodes ticked elsewhere (spatial from camera callbacks,
		# detectors in the camera/mics clients, scene by the client loop)
		elsewhere = {"spatial", "decode", "detect_april", "detect_motion", "detect_face",
			"detect_ball", "detect_audio", "scene"}
		missing = [n for n in created if n not in ticked and n not in elsewhere]
		self.assertEqual(missing, [], "nodes created but never ticked: " + ", ".join(missing))

	def test_hearing_ticks_after_voice(self):

		ticked = self_attrs(demo_nodes_methods()["tick"], "tick")
		self.assertLess(ticked.index("voice"), ticked.index("hearing"))


class TestTtsContentType(unittest.TestCase):

	def test_declared_pcm_is_never_sniffed_as_mp3(self):

		from cloud.tts import is_mp3
		# raw s16le whose first sample (-1) looks like an MPEG sync word
		self.assertFalse(is_mp3("audio/pcm", b"\xff\xff\x00\x00"))
		self.assertFalse(is_mp3("application/octet-stream", b"\xff\xe0\x00\x00"))
		self.assertTrue(is_mp3("audio/mpeg", b"\x00\x00"))
		self.assertTrue(is_mp3("", b"ID3\x04"))
		# no content type: a real MPEG-1 layer III header is accepted, junk is not
		self.assertTrue(is_mp3(None, b"\xff\xfb\x90\x64"))
		self.assertFalse(is_mp3(None, b"\xff\xff\x00\x00"))


if __name__ == "__main__":
	unittest.main()
