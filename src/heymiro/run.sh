#!/bin/bash
#
# Hey MiRo demo - one launcher for the physical MiRo-e and the Gazebo simulator.
#
#   ./run.sh --sim                    # Gazebo + this computer's mic/speaker
#   ./run.sh --robot                  # physical MiRo (its own microphones)
#   ./run.sh --robot --ext-mic        # physical MiRo, speech from this computer's mic
#   ./run.sh --sim --online           # the cloud (HRI'25) pipeline instead of the local one
#
# Platform (default: the mode the shell was opened in, $MIRO_MODE):
#   --sim              MiRo Gazebo simulator (shell from docker/run_docker.bash --sim)
#   --robot            physical MiRo-e       (shell from docker/run_docker.bash --robot-ip IP)
#
# AI pipeline (default: ai.provider in config/heymiro.yaml = local):
#   --offline          everything on this computer (docker/local-ai): no keys, no internet
#   --online           cloud services (OpenAI, ElevenLabs, Picovoice): keys in the secrets file
#
# Robot options:
#   --ext-mic          capture speech from this computer's microphone via
#                      tools/host_audio_bridge.py (topic /<robot>/host/mics)
# Simulator options:
#   --no-gazebo        do not start roscore/Gazebo (use the running ones)
#   --no-bridge        do not start tools/host_audio_bridge.py (e.g. for tools/fake_user.py)
#   --no-speaker       bridge publishes the mic only (no control/stream relay)
#   -- ARGS            extra arguments for ~/mdk/sim/launch_sim.sh
# Both:
#   --device DEV       microphone for the bridge (index or name substring)
#   --no-cameras       do not start the caml/camr clients
#   --no-scene         do not start the scene (YOLO/gesture) client
#   --no-check         skip the pre-flight check of keys / local AI services
#
# Robot: turn OFF demo mode in the MiRo app first (the on-board demo and this
# one would fight over the robot). Simulator: use headphones so MiRo does not
# hear itself.

set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source-path=SCRIPTDIR source=tools/demo_launch.bash
source "$HERE/tools/demo_launch.bash"

PLATFORM=""
PROVIDER=""
EXT_MIC=0
GAZEBO=1
BRIDGE=1
SPEAKER=1
DEVICE=""
CAMERAS=1
SCENE=1
CHECK=1
SIM_ARGS=()
while [ $# -gt 0 ]; do
	case "$1" in
		--sim) PLATFORM=sim; shift ;;
		--robot) PLATFORM=robot; shift ;;
		--offline|--local) PROVIDER=local; shift ;;
		--online|--cloud) PROVIDER=cloud; shift ;;
		--ext-mic) EXT_MIC=1; shift ;;
		--no-gazebo) GAZEBO=0; shift ;;
		--no-bridge) BRIDGE=0; shift ;;
		--no-speaker) SPEAKER=0; shift ;;
		--device) DEVICE="$2"; shift 2 ;;
		--no-cameras) CAMERAS=0; shift ;;
		--no-scene) SCENE=0; shift ;;
		--no-check) CHECK=0; shift ;;
		--) shift; SIM_ARGS=("$@"); break ;;
		-h|--help) sed -n '2,34p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
		*) echo "unknown option: $1 (see --help)"; exit 2 ;;
	esac
done

heymiro_env

# --- platform: must match how this shell's ROS network was set up -------------------
PLATFORM="${PLATFORM:-${MIRO_MODE:-}}"
if [ "$PLATFORM" != "sim" ] && [ "$PLATFORM" != "robot" ]; then
	echo "choose a platform: --sim or --robot (see --help)"
	exit 2
fi
if [ -n "$MIRO_MODE" ] && [ "$MIRO_MODE" != "$PLATFORM" ]; then
	echo "this shell is set up for the $MIRO_MODE (ROS_MASTER_URI=${ROS_MASTER_URI:-unset})."
	if [ "$PLATFORM" = "sim" ]; then
		echo "  open a simulator shell:  docker/run_docker.bash --sim"
	else
		echo "  open a robot shell:      docker/run_docker.bash --robot-ip <MiRo IP>"
	fi
	exit 1
fi
export HEYMIRO_PLATFORM="$PLATFORM"

# --- AI pipeline: appended last, so it wins over earlier HEYMIRO_SET entries ------------
PROVIDER="${PROVIDER:-${HEYMIRO_PROVIDER:-}}"
if [ -n "$PROVIDER" ]; then
	export HEYMIRO_SET="${HEYMIRO_SET:+$HEYMIRO_SET;}ai.provider=$PROVIDER"
fi

if [ "$CHECK" = 1 ]; then
	python3 "$HERE/tools/preflight.py" || exit 1
fi

trap heymiro_cleanup EXIT
trap 'exit 130' INT TERM

# --- ROS master, simulator / robot ----------------------------------------------------------
if [ "$PLATFORM" = "sim" ]; then
	export ROS_MASTER_URI="${ROS_MASTER_URI:-http://localhost:11311}"
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
	if ! timeout --foreground 120 rostopic echo -n 1 "/$MIRO_ROBOT_NAME/sensors/package/flags" >/dev/null 2>&1; then
		echo "no sensors/package after 120 s - is the simulator running (and not paused)?"
		exit 1
	fi
else
	if ! heymiro_master_up; then
		echo "cannot reach the ROS master at ${ROS_MASTER_URI:-?}"
		echo "  is the MiRo on, on the same network, and was the shell opened with --robot-ip?"
		exit 1
	fi
	heymiro_warn_foreign_demo
fi

# --- host microphone bridge -------------------------------------------------------------------
if [ "$PLATFORM" = "sim" ] && [ "$BRIDGE" = 1 ]; then
	# the simulator has no mics/speaker: this computer's mic -> /<robot>/host/mics,
	# MiRo's control/stream -> this computer's speaker
	bridge=(python3 -u "$HERE/tools/host_audio_bridge.py")
	[ "$SPEAKER" = 1 ] && bridge+=(--speaker)
	[ -n "$DEVICE" ] && bridge+=(--device "$DEVICE")
	heymiro_start_bg audio_bridge "${bridge[@]}"
elif [ "$PLATFORM" = "robot" ] && [ "$EXT_MIC" = 1 ]; then
	bridge=(python3 -u "$HERE/tools/host_audio_bridge.py")
	[ -n "$DEVICE" ] && bridge+=(--device "$DEVICE")
	heymiro_start_bg audio_bridge "${bridge[@]}"
	# speech from the host mic; sound localisation stays on MiRo's ears
	export HEYMIRO_SET="hearing.input_topic=host/mics${HEYMIRO_SET:+;$HEYMIRO_SET}"
fi

heymiro_start_clients "$CAMERAS" "$SCENE"
heymiro_run_main
