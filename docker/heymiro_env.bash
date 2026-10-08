# shellcheck shell=bash
#
# Hey MiRo container environment - sourced by the entrypoint and by every
# interactive shell (~/.bashrc), so `docker exec` terminals match.
#
# Network setup is driven by environment variables passed by run_docker.bash:
#   MIRO_MODE=robot|sim     robot: ROS master on the MiRo; sim: local roscore + Gazebo
#   MIRO_ROBOT_IP=a.b.c.d   (robot mode) MiRo's IP, from the MiRo app
#   MIRO_HOST_IP=auto|IP    this computer's IP as seen by the robot (default auto)
#   MIRO_ROBOT_NAME=miro    robot name used in topic names
#
# This replaces editing ROS_MASTER_IP in ~/mdk/share/config/setup.bash.
# Never use `set -x` here: it would echo environment values.

# ROS + MDK (sources the catkin workspace and sets MIRO_DIR_*)
# shellcheck disable=SC1091
source /opt/ros/noetic/setup.bash
# shellcheck disable=SC1091
[ -f /root/mdk/setup.bash ] && source /root/mdk/setup.bash
# shellcheck disable=SC1091
[ -f /root/mdk/catkin_ws/devel/setup.bash ] && source /root/mdk/catkin_ws/devel/setup.bash

export MIRO_MODE="${MIRO_MODE:-robot}"
export MIRO_ROBOT_NAME="${MIRO_ROBOT_NAME:-miro}"
unset ROS_HOSTNAME

if [ "$MIRO_MODE" = "sim" ]; then
	export ROS_MASTER_URI="http://localhost:11311"
	export ROS_IP="127.0.0.1"
else
	if [ -n "$MIRO_ROBOT_IP" ]; then
		export ROS_MASTER_URI="http://${MIRO_ROBOT_IP}:11311"
		if [ -z "$MIRO_HOST_IP" ] || [ "$MIRO_HOST_IP" = "auto" ]; then
			# the source address the kernel would use to reach the robot
			_heymiro_ip="$(ip -4 route get "$MIRO_ROBOT_IP" 2>/dev/null | sed -n 's/.* src \([0-9.]*\).*/\1/p')"
			[ -n "$_heymiro_ip" ] && export ROS_IP="$_heymiro_ip"
			unset _heymiro_ip
		else
			export ROS_IP="$MIRO_HOST_IP"
		fi
	fi
fi

# state/config folders used by client_demo.py
[ -n "$MIRO_DIR_STATE" ] && mkdir -p "$MIRO_DIR_STATE"
[ -n "$MIRO_DIR_CONFIG" ] && mkdir -p "$MIRO_DIR_CONFIG"

export HEYMIRO_CACHE_DIR="${HEYMIRO_CACHE_DIR:-/root/.cache/heymiro}"
mkdir -p "$HEYMIRO_CACHE_DIR"

# the projects (bind-mounted)
export HEYMIRO_ROBOT=/root/mdk/catkin_ws/src/cloud_demo_robot
export HEYMIRO_SIM=/root/mdk/catkin_ws/src/cloud_demo_sim

heymiro_status() {
	# report configuration WITHOUT printing any secret value
	echo "Hey MiRo environment"
	echo "  mode:            $MIRO_MODE"
	echo "  robot name:      $MIRO_ROBOT_NAME"
	echo "  ROS_MASTER_URI:  ${ROS_MASTER_URI:-unset}"
	echo "  ROS_IP:          ${ROS_IP:-unset}"
	if [ "$MIRO_MODE" = "robot" ] && [ -z "$MIRO_ROBOT_IP" ]; then
		echo "  WARNING: robot mode but MIRO_ROBOT_IP is not set (run_docker.bash --robot-ip IP)"
	fi
	if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
		echo "  GPU:             $(nvidia-smi -L | head -n1 | cut -c1-60)"
	else
		echo "  GPU:             none (YOLO runs on CPU)"
	fi
	if pactl info >/dev/null 2>&1; then
		echo "  audio:           PulseAudio OK"
	else
		echo "  audio:           no PulseAudio server (host mic/speaker unavailable)"
	fi
	if [ -n "$HEYMIRO_SECRETS_FILE" ] && [ -r "$HEYMIRO_SECRETS_FILE" ]; then
		echo "  secrets file:    $HEYMIRO_SECRETS_FILE"
		# names only: keys that have a non-empty value
		_heymiro_set="$(sed -n 's/^\(export \)\{0,1\}\([A-Z0-9_]*\)=..*$/\2/p' "$HEYMIRO_SECRETS_FILE" | tr '\n' ' ')"
		echo "  secrets set:     ${_heymiro_set:-none}"
		unset _heymiro_set
	else
		echo "  secrets file:    none (cloud features off; see README 'Secrets and API routes')"
	fi
	echo
	echo "  robot: cd \$HEYMIRO_ROBOT && ./run_robot.sh [--ext-mic]"
	echo "  sim:   cd \$HEYMIRO_SIM && ./run_sim.sh"
	echo "  more terminals: run docker/run_docker.bash again (attaches with docker exec)"
}
