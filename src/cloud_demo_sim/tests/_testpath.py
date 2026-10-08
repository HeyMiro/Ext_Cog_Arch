#
#	Test helper: put core/ (and tools/) on sys.path, and optionally install
#	light-weight stand-ins for rospy, std_msgs and miro2 so that nodes and
#	actions can be imported without a ROS installation.
#
#	usage (first lines of a test module):
#		import _testpath
#		_testpath.install_stubs()      # only if you import ROS-dependent modules
#

import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CORE = os.path.join(ROOT, "core")
TOOLS = os.path.join(ROOT, "tools")

for p in (CORE, TOOLS):
	if p not in sys.path:
		sys.path.insert(0, p)


def _module(name):

	mod = types.ModuleType(name)
	sys.modules[name] = mod
	return mod


class _Msg(object):

	# generic ROS message stand-in: any attribute, data list by default
	def __init__(self, *args, **kwargs):
		self.data = []
		for k, v in kwargs.items():
			setattr(self, k, v)


class _Pub(object):

	def __init__(self, *args, **kwargs):
		self.sent = []

	def publish(self, msg):
		self.sent.append(msg)

	def unregister(self):
		pass


def install_stubs():

	# only install once, and never shadow the real packages
	if "rospy" not in sys.modules:
		try:
			import rospy  # noqa: F401
		except ImportError:
			rospy = _module("rospy")
			rospy.Publisher = _Pub
			rospy.Subscriber = _Pub
			rospy.get_time = lambda: 0.0
			rospy.loginfo = rospy.logwarn = rospy.logerr = lambda *a, **k: None
			rospy.core = types.SimpleNamespace(is_shutdown=lambda: False)

	for pkg, names in (
			("std_msgs", ["String", "Int16MultiArray", "UInt16MultiArray", "UInt32MultiArray",
				"Float32MultiArray", "UInt32", "MultiArrayDimension", "Bool"]),
			("sensor_msgs", ["Image", "CompressedImage", "JointState"]),
			("geometry_msgs", ["TwistStamped", "Vector3"])):
		if pkg in sys.modules:
			continue
		try:
			__import__(pkg + ".msg")
		except ImportError:
			top = _module(pkg)
			msg = _module(pkg + ".msg")
			top.msg = msg
			for n in names:
				setattr(msg, n, type(n, (_Msg,), {}))

	if "miro2" not in sys.modules:
		try:
			import miro2  # noqa: F401
		except ImportError:
			_install_miro2()


def _install_miro2():

	import numpy as np

	miro = _module("miro2")
	c = _module("miro2.constants")
	miro.constants = c
	deg = np.pi / 180.0
	c.MIC_SAMPLE_RATE = 20000
	c.SPKR_SAMPLE_RATE = 8000
	c.TILT_RAD_MIN = c.TILT_RAD_MAX = c.TILT_RAD_CALIB = -6.0 * deg
	c.LIFT_RAD_MIN, c.LIFT_RAD_MAX, c.LIFT_RAD_CALIB = 8.0 * deg, 60.0 * deg, 34.0 * deg
	c.YAW_RAD_MIN, c.YAW_RAD_MAX, c.YAW_RAD_CALIB = -55.0 * deg, 55.0 * deg, 0.0
	c.PITCH_RAD_MIN, c.PITCH_RAD_MAX, c.PITCH_RAD_CALIB = -22.0 * deg, 8.0 * deg, 0.0
	c.PUSH_FLAG_IMPULSE = 1
	c.PUSH_FLAG_VELOCITY = 2
	c.PUSH_FLAG_NO_TRANSLATION = 4
	c.PUSH_FLAG_NO_ROTATION = 8
	c.PUSH_FLAG_NO_NECK_MOVEMENT = 16
	c.LINK_HEAD = 3
	c.LINK_BODY = 1
	c.PLATFORM_U_FLAG_SIMULATOR = 1 << 15

	lib = _module("miro2.lib")
	miro.lib = lib

	class PerformanceTimer(object):
		def __init__(self, *a, **k):
			pass

	class KinematicPush(object):
		def __init__(self):
			self.link = 0
			self.flags = 0
			self.pos = np.zeros(3)
			self.vec = np.zeros(3)

	kc = _module("miro2.lib.kc")
	kc.KinematicPush = KinematicPush
	lib.kc = kc
	lib.PerformanceTimer = PerformanceTimer
	lib.get = lambda key: np.array([0.1, 0.0, 0.0])
	miro.msg = _module("miro2.msg")
