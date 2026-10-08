# shellcheck shell=bash
#
# Shared launcher functions for run_robot.sh / run_sim.sh (sourced, not run).
#
# Starts the MDK-style demo clients the same way the MiRo itself does:
#   client_demo.py - caml / - camr   camera clients (face, motion, AprilTag)
#   client_demo.py - mics            sound localisation client
#   client_demo.py - scene           Hey MiRo scene awareness (YOLO, gestures)
#   client_demo.py -                 main client (affect, action selection, forebrain)
# Background processes log to $LOGDIR; Ctrl-C stops everything this script started.

HEYMIRO_PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HEYMIRO_PIDS=()
HEYMIRO_PGIDS=()

heymiro_env() {
	# outside the container's interactive shell, set up ROS + MDK first
	if [ -z "$ROS_DISTRO" ] || ! command -v rostopic >/dev/null 2>&1; then
		if [ -f /opt/heymiro/heymiro_env.bash ]; then
			# shellcheck disable=SC1091
			source /opt/heymiro/heymiro_env.bash
		else
			echo "ROS is not set up (source /opt/ros/noetic/setup.bash and ~/mdk/setup.bash)"
			exit 1
		fi
	fi
	export MIRO_ROBOT_NAME="${MIRO_ROBOT_NAME:-miro}"
	LOGDIR="${LOGDIR:-$HOME/.ros/heymiro/$(date +%F-%H%M%S)}"
	mkdir -p "$LOGDIR"
	echo "logs: $LOGDIR"
}

heymiro_master_up() {
	timeout --foreground 5 rostopic list >/dev/null 2>&1
}

heymiro_start_bg() {
	# heymiro_start_bg <name> <command...>  (runs in its own process group)
	local name="$1"
	shift
	setsid "$@" >"$LOGDIR/$name.log" 2>&1 &
	HEYMIRO_PIDS+=("$!")
	HEYMIRO_PGIDS+=("$!")
	echo "started $name (pid $!, log $LOGDIR/$name.log)"
}

heymiro_cleanup() {
	trap - EXIT INT TERM
	echo
	echo "stopping..."
	local i
	for (( i=${#HEYMIRO_PGIDS[@]}-1; i>=0; i-- )); do
		kill -INT -- "-${HEYMIRO_PGIDS[$i]}" 2>/dev/null || true
	done
	sleep 3
	for (( i=${#HEYMIRO_PGIDS[@]}-1; i>=0; i-- )); do
		kill -TERM -- "-${HEYMIRO_PGIDS[$i]}" 2>/dev/null || true
	done
	wait 2>/dev/null || true
}

heymiro_warn_foreign_demo() {
	# a demo already running on the robot (MiRo app "demo mode") fights ours
	local others
	others="$(timeout --foreground 5 rosnode list 2>/dev/null | grep "/${MIRO_ROBOT_NAME}_client_demo" || true)"
	if [ -n "$others" ]; then
		echo "WARNING: other demo clients are running:"
		while read -r node; do echo "    $node"; done <<< "$others"
		echo "  Turn OFF demo mode in the MiRo app before running the cloud demo."
	fi
}

heymiro_start_clients() {
	# heymiro_start_clients <cameras 0|1> <scene 0|1>
	local cameras="$1" scene="$2"
	cd "$HEYMIRO_PROJECT/core" || exit 1
	if [ "$cameras" = 1 ]; then
		heymiro_start_bg caml python3 -u client_demo.py - caml
		heymiro_start_bg camr python3 -u client_demo.py - camr
	fi
	heymiro_start_bg mics python3 -u client_demo.py - mics
	if [ "$scene" = 1 ]; then
		heymiro_start_bg scene python3 -u client_demo.py - scene
	fi
}

heymiro_run_main() {
	cd "$HEYMIRO_PROJECT/core" || exit 1
	echo "starting main client (Ctrl-C to stop everything)"
	python3 -u client_demo.py - 2>&1 | tee "$LOGDIR/main.log"
}
