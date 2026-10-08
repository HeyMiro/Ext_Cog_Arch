#
#	Hey MiRo - choreography (pure Python + numpy).
#
#	moves.py   joint limits, min-jerk Trajectory, conversation poses
#	           (look_up, lean_in, light_up, ...) and beat-driven head
#	           moves (head_bop, head_bang, party_nod, ...)
#	lights.py  LED colours, genre palettes, flash / fade / pulse patterns
#	dance.py   DanceController: beat clock, 8-16 beat sections, wheels,
#	           cosmetics and lights for one song
#
#	These replace the HRI'25 dance_body / dance_illum / dance_wheels /
#	miro_movebank modules. Nothing here imports rospy; moves.py reads
#	the joint limits from miro2.constants when it is installed.
#

"""Hey MiRo choreography helpers (no ROS dependency)."""
