#!/bin/bash
#
# Hey MiRo cloud demo on a physical MiRo-e.
#
#   ./run_robot.sh                # MiRo's own microphones (quiet rooms)
#   ./run_robot.sh --ext-mic      # host / external microphone (noisy rooms, HRI paper)
#
# Options:
#   --ext-mic          capture speech from this computer's microphone via
#                      tools/host_audio_bridge.py (topic /<robot>/host/mics)
#   --device DEV       input device for --ext-mic (index or name substring)
#   --no-cameras       do not start the caml/camr clients
#   --no-scene         do not start the scene (YOLO/gesture) client
#
# Before running: find the robot's IP in the MiRo app, start the container
# with docker/run_docker.bash --robot-ip <IP>, and turn OFF demo mode in the
# MiRo app (otherwise the on-board demo and this one fight over the robot).

set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source-path=SCRIPTDIR source=tools/demo_launch.bash
source "$HERE/tools/demo_launch.bash"

EXT_MIC=0
DEVICE=""
CAMERAS=1
SCENE=1
while [ $# -gt 0 ]; do
	case "$1" in
		--ext-mic) EXT_MIC=1; shift ;;
		--device) DEVICE="$2"; shift 2 ;;
		--no-cameras) CAMERAS=0; shift ;;
		--no-scene) SCENE=0; shift ;;
		-h|--help) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
		*) echo "unknown option: $1 (see --help)"; exit 2 ;;
	esac
done

heymiro_env
if [ "$MIRO_MODE" = "sim" ]; then
	echo "this container is in sim mode; use src/cloud_demo_sim/run_sim.sh (or restart with --robot-ip)"
	exit 1
fi
if ! heymiro_master_up; then
	echo "cannot reach the ROS master at ${ROS_MASTER_URI:-?}"
	echo "  is the MiRo on, on the same network, and was the container started with --robot-ip?"
	exit 1
fi
heymiro_warn_foreign_demo

trap heymiro_cleanup EXIT
trap 'exit 130' INT TERM

if [ "$EXT_MIC" = 1 ]; then
	bridge=(python3 -u "$HERE/tools/host_audio_bridge.py")
	[ -n "$DEVICE" ] && bridge+=(--device "$DEVICE")
	heymiro_start_bg audio_bridge "${bridge[@]}"
	# speech from the host mic; sound localisation stays on MiRo's ears
	export HEYMIRO_SET="hearing.input_topic=host/mics${HEYMIRO_SET:+;$HEYMIRO_SET}"
fi

heymiro_start_clients "$CAMERAS" "$SCENE"
heymiro_run_main
