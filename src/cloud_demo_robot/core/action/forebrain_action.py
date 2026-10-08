#
#	Hey MiRo - ForebrainAction: base class of the forebrain actions
#	(converse, dance, tricks, identify).
#
#	These actions compete in the stock basal ganglia like any other, in
#	their own priority band (pars.action.priority_forebrain = 0.84):
#	above orient / approach / flee (<= 0.8), below halt / avert /
#	touch-mull (>= 0.9). HRI'25 used priorities of 2, 1.5 and 1.2, which
#	the basal ganglia clip to 1.0, so they tied with the protective
#	actions and cliff aversion could lose. Here the priority is also
#	scaled by conf_surf (confidence that there is floor ahead), so avert
#	wins as soon as a cliff appears.
#
#	An action is "wanted" while the forebrain wants it (wanted() is
#	defined by each subclass). While it runs, its clock is used as a
#	hold-open button (reset every tick), and it ends itself (finish())
#	when it is no longer wanted or its episode is over. Pre-emption by a
#	higher-priority action calls stop() like any stock action.
#
#	Subclasses define NAME (and TASK if they run a forebrain task) and
#	the hooks on_start() (return False to not start), on_service() and
#	on_stop(). Helpers: set_head(), body_velocity(), now().
#

import time

import numpy as np

import miro2 as miro

from . import action_types
from choreography import moves


# ticks the hold-open clock runs before it must be reset again
HOLD_STEPS = 50



class ForebrainAction(action_types.ActionTemplate):

	NAME = "forebrain"
	TASK = None       # forebrain task this action runs, if any

	def finalize(self):

		self.name = self.NAME

		# these behaviours answer people; they should not fade away
		# because MiRo is drowsy (a "go to sleep" request ends them instead)
		self.modulate_by_wakefulness = False
		self.priority_forebrain = float(getattr(self.pars.action, "priority_forebrain", 0.84))
		self.ongoing_priority = self.priority_forebrain

		self.clock_fn = time.monotonic
		self.t_start = 0.0
		self.stop_reason = None

		self.init()

	def init(self):

		# subclass set-up (called once from finalize)
		pass

	# ------------------------------------------------------------- accessors

	def node(self, name):

		return getattr(self.parent.nodes, name, None)

	@property
	def forebrain(self):

		return self.node("forebrain")

	@property
	def voice(self):

		return self.node("voice")

	@property
	def express(self):

		return self.node("express")

	def now(self):

		return self.clock_fn()

	def elapsed(self):

		return self.now() - self.t_start

	# -------------------------------------------------------------- priority

	def wanted(self):

		raise NotImplementedError("ForebrainAction subclasses define wanted()")

	def task_wanted(self):

		# for task actions: wanted while our task is pending or running
		fb = self.forebrain
		if fb is None or self.TASK is None:
			return False
		return fb.pending_task() == self.TASK or fb.active_task == self.TASK

	def compute_priority(self):

		if not self.wanted():
			return 0.0
		return self.priority_forebrain * float(self.input.conf_surf)

	def ascending(self):

		# while selected and running, hold our band (the stock template
		# would decay to ongoing_priority); otherwise ask wanted(). Both
		# are scaled by conf_surf so that cliff aversion can take over.
		if self.interface.inhibition == 0 and self.clock.isActive():
			self.interface.priority = self.priority_forebrain * float(self.input.conf_surf)
		else:
			self.interface.priority = self.compute_priority()

	# ------------------------------------------------------------- lifecycle

	def start(self):

		# selected: begin an episode (unless the reason went away meanwhile)
		if not self.wanted():
			return
		self.t_start = self.now()
		self.stop_reason = None
		if self.on_start() is False:
			return
		self.clock.start(HOLD_STEPS)

	def service(self):

		if not self.wanted():
			self.finish("done")
			return
		self.hold()
		self.on_service()

	def hold(self):

		# use the clock as a hold-open button rather than a clock
		self.clock.reset()

	def finish(self, reason="done"):

		# end the episode ourselves (as opposed to being pre-empted)
		self.stop_reason = reason
		self.stop()

	def stop(self):

		# called by finish() and, on pre-emption, by the stock action loop
		reason = self.stop_reason or "preempted"
		self.stop_reason = None
		if self.clock.isActive():
			self.on_stop(reason)
		express = self.express
		if express is not None:
			express.release(self.name)
		fb = self.forebrain
		if self.TASK is not None and fb is not None and fb.active_task == self.TASK:
			fb.end_task(self.TASK, reason)
		self.clock.stop()

	def on_start(self):

		pass

	def on_service(self):

		pass

	def on_stop(self, reason):

		pass

	# ----------------------------------------------------------------- motor

	def set_head(self, cfg):

		# clipped to the joint limits (and NaN-free) before it reaches kc
		self.kc.setConfig(moves.clip_config(cfg))

	def body_velocity(self, vx, wz):

		"""
		Drive the body with a velocity push at fovea_BODY (0.1 m ahead of
		the body centre, in BODY): vx forward (m/s), wz the sideways push
		there, which turns the body (+ = to the left). The values follow
		HRI'25 ([0, 0.4, 0] to spin). Forward motion is scaled by conf_surf
		so we never drive towards a cliff; the stock action loop adds the
		BODY_ENABLE_* flags.
		"""

		vx = float(vx) * float(self.input.conf_surf)
		self.apply_push_body(np.array([vx, float(wz), 0.0]), flags=miro.constants.PUSH_FLAG_VELOCITY)

	def override(self, channel, value, ttl_ticks=5):

		express = self.express
		if express is not None:
			express.override(channel, value, ttl_ticks=ttl_ticks, owner=self.name)
