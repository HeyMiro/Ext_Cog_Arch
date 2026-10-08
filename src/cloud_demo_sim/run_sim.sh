#!/bin/bash
#
# Hey MiRo cloud demo in the MiRo Gazebo simulator.
#
#   ./run_sim.sh                  # roscore + Gazebo + host audio bridge + demo clients
#   ./run_sim.sh --no-gazebo      # attach to a simulator you started yourself
#   ./run_sim.sh -- <args>        # extra arguments for ~/mdk/sim/launch_sim.sh
#
# Options:
#   --no-gazebo        do not start roscore/Gazebo (use the running ones)
#   --no-bridge        do not start tools/host_audio_bridge.py
#   --device DEV       microphone for the bridge (index or name substring)
#   --no-speaker       bridge publishes the mic only (no control/stream relay)
#   --no-cameras       do not start the caml/camr clients
#   --no-scene         do not start the scene (YOLO/gesture) client
#
# The simulator has no microphones or speaker: the bridge publishes this
# computer's microphone on /<robot>/host/mics (MiRo's mic format) and plays
# MiRo's speaker stream; speech and music play on this computer's speaker.
# Headphones stop MiRo hearing itself.

set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source-path=SCRIPTDIR source=tools/demo_launch.bash
source "$HERE/tools/demo_launch.bash"

GAZEBO=1
BRIDGE=1
SPEAKER=1
DEVICE=""
CAMERAS=1
SCENE=1
SIM_ARGS=()
while [ $# -gt 0 ]; do
	case "$1" in
		--no-gazebo) GAZEBO=0; shift ;;
		--no-bridge) BRIDGE=0; shift ;;
		--no-speaker) SPEAKER=0; shift ;;
		--device) DEVICE="$2"; shift 2 ;;
		--no-cameras) CAMERAS=0; shift ;;
		--no-scene) SCENE=0; shift ;;
		--) shift; SIM_ARGS=("$@"); break ;;
		-h|--help) sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
		*) echo "unknown option: $1 (see --help)"; exit 2 ;;
	esac
done

heymiro_env
if [ "$MIRO_MODE" = "robot" ] && [ -n "$MIRO_ROBOT_IP" ]; then
	echo "this container is in robot mode; use src/cloud_demo_robot/run_robot.sh (or restart with --sim)"
	exit 1
fi
export ROS_MASTER_URI="${ROS_MASTER_URI:-http://localhost:11311}"

trap heymiro_cleanup EXIT INT TERM

if [ "$GAZEBO" = 1 ]; then
	if ! heymiro_master_up; then
		heymiro_start_bg roscore roscore
		for _ in $(seq 1 30); do heymiro_master_up && break; sleep 1; done
	fi
	SIM_DIR="${MIRO_DIR_MDK:-$HOME/mdk}/sim"
	heymiro_start_bg gazebo "$SIM_DIR/launch_sim.sh" "${SIM_ARGS[@]}"
fi

if ! heymiro_master_up; then
	echo "no ROS master at $ROS_MASTER_URI"
	exit 1
fi

echo "waiting for the simulated MiRo (/$MIRO_ROBOT_NAME/sensors/package)..."
if ! timeout 120 rostopic echo -n 1 "/$MIRO_ROBOT_NAME/sensors/package/flags" >/dev/null 2>&1; then
	echo "no sensors/package after 120 s - is the simulator running (and not paused)?"
	exit 1
fi

if [ "$BRIDGE" = 1 ]; then
	bridge=(python3 -u "$HERE/tools/host_audio_bridge.py")
	[ "$SPEAKER" = 1 ] && bridge+=(--speaker)
	[ -n "$DEVICE" ] && bridge+=(--device "$DEVICE")
	heymiro_start_bg audio_bridge "${bridge[@]}"
fi

heymiro_start_clients "$CAMERAS" "$SCENE"
heymiro_run_main
