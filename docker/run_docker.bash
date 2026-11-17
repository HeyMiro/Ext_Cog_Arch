#!/bin/bash
#
# Open a terminal in the Hey MiRo environment container (created on first use).
#
#   docker/run_docker.bash --sim                         # simulator terminal (local roscore + Gazebo)
#   docker/run_docker.bash --robot-ip 192.168.0.42       # robot terminal (ROS master on the MiRo)
#   docker/run_docker.bash --sim --offline               # ... and start the local AI services first
#   docker/run_docker.bash --robot-ip IP --online        # ... using the cloud (HRI'25) services
#   docker/run_docker.bash --stop                        # remove the container
#
# One long-lived container ('heymiro') is shared by every terminal. The mode
# (--sim / --robot-ip) and the AI pipeline (--offline / --online) are set per
# terminal, so you can switch between the simulator and the robot without
# restarting anything: just open a new terminal with the other flag.
#
# The image holds the environment only (ROS Noetic, MiRo MDK, Gazebo, Python
# packages). The code is bind-mounted, so edits on the host (or a `git pull`)
# are visible inside immediately:
#
#   src/heymiro             -> /root/mdk/catkin_ws/src/heymiro   ($HEYMIRO_DIR)
#   docker/heymiro_env.bash -> /opt/heymiro/heymiro_env.bash     (read-only)
#
# Secrets (API keys, online pipeline only) are read from
# ~/.config/heymiro/secrets.env (create it from config/secrets.env.example)
# and mounted read-only; they are never baked into the image or passed on
# the command line.
#
# Per terminal:
#   --sim                simulator: local roscore + Gazebo
#   --robot-ip IP        physical MiRo at IP (from the MiRo app)
#   --robot              robot mode with the IP this container was created with
#   --host-ip IP|auto    this computer's IP as seen by the robot (default auto)
#   --robot-name NAME    MiRo robot name (default miro)
#   --offline [9b]       local models only: start docker/local-ai/up.bash first
#                        (9b = the larger Qwen3.5-9B); run.sh then uses them
#   --online             cloud services (needs the secrets file)
#   -- CMD ...           command to run instead of an interactive bash
# When the container is created:
#   --gpu auto|on|off    pass the NVIDIA GPU through (default auto)
#   --image IMAGE        environment image (default: see "which image" below)
#   --name NAME          container name (default heymiro)
#   --secrets FILE       secrets file (default ~/.config/heymiro/secrets.env)
#   --no-audio           do not pass the host sound server / devices
#   --no-x11             do not pass the X display (no Gazebo GUI)
#   --privileged         run privileged (normally not needed)
# Other:
#   --pull               download the environment image, then exit
#   --stop               stop and remove the container, then exit
#   --local-ai [9b]      old name of --offline

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# under sudo, use the invoking user's home and sound server, not root's
# (better: add yourself to the docker group, see the README)
USER_HOME="$HOME"
USER_RUNTIME="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
if [ -n "$SUDO_USER" ] && [ "$SUDO_USER" != "root" ]; then
	USER_HOME="$(getent passwd "$SUDO_USER" | cut -d: -f6)"
	USER_RUNTIME="/run/user/$(id -u "$SUDO_USER")"
	echo "NOTE: running under sudo; using $SUDO_USER's secrets file and sound server"
fi

MODE=""
ROBOT_IP=""
HOST_IP=""
ROBOT_NAME=""
PROVIDER=""
LOCAL_AI=""
GPU="auto"
IMAGE=""
NAME="heymiro"
SECRETS="$USER_HOME/.config/heymiro/secrets.env"
AUDIO=1
X11=1
PRIVILEGED=0
PULL=0
STOP=0
CMD=(bash)

usage() {
	sed -n '2,/^set -e/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//; /^set -e/d'
}

while [ $# -gt 0 ]; do
	case "$1" in
		--sim) MODE="sim"; shift ;;
		--robot-ip) MODE="robot"; ROBOT_IP="$2"; shift 2 ;;
		--robot) MODE="robot"; shift ;;
		--host-ip) HOST_IP="$2"; shift 2 ;;
		--robot-name) ROBOT_NAME="$2"; shift 2 ;;
		--offline|--local-ai)
			PROVIDER="local"; LOCAL_AI="default"; shift
			if [ "$1" = "9b" ]; then LOCAL_AI="9b"; shift; fi ;;
		--online) PROVIDER="cloud"; shift ;;
		--gpu) GPU="$2"; shift 2 ;;
		--image) IMAGE="$2"; shift 2 ;;
		--name) NAME="$2"; shift 2 ;;
		--secrets) SECRETS="$2"; shift 2 ;;
		--no-audio) AUDIO=0; shift ;;
		--no-x11) X11=0; shift ;;
		--privileged) PRIVILEGED=1; shift ;;
		--pull) PULL=1; shift ;;
		--stop) STOP=1; shift ;;
		--) shift; CMD=("$@"); break ;;
		-h|--help) usage; exit 0 ;;
		*) echo "unknown option: $1 (see --help)"; exit 2 ;;
	esac
done

if [ "$STOP" = 1 ]; then
	if docker rm -f "$NAME" >/dev/null 2>&1; then echo "removed container '$NAME'"; else echo "no container '$NAME'"; fi
	# take back the X display permission given when the container was created
	command -v xhost >/dev/null && xhost -SI:localuser:root >/dev/null 2>&1
	exit 0
fi

# --- which environment image ----------------------------------------------------------------
# CI publishes one per branch (slashes become dashes). Prefer, in order: the
# image for the branch checked out here, a local build (docker/build_image.bash),
# :latest, and the last published environment image.
REPO_IMAGE="ghcr.io/heymiro/ext_cog_arch"
FALLBACK_IMAGE="$REPO_IMAGE:claude-local-ai-pipeline"
image_exists() {
	docker image inspect "$1" >/dev/null 2>&1 || docker manifest inspect "$1" >/dev/null 2>&1
}
if [ -z "$IMAGE" ]; then
	branch="$(git -C "$REPO_ROOT" branch --show-current 2>/dev/null | tr '/' '-')"
	candidates=()
	[ -n "$branch" ] && [ "$branch" != "main" ] && candidates+=("$REPO_IMAGE:$branch")
	candidates+=("heymiro-env:local" "$REPO_IMAGE:latest" "$FALLBACK_IMAGE")
	for candidate in "${candidates[@]}"; do
		if image_exists "$candidate"; then IMAGE="$candidate"; break; fi
	done
	IMAGE="${IMAGE:-$FALLBACK_IMAGE}"
fi

if [ "$PULL" = 1 ]; then
	echo "downloading $IMAGE ..."
	docker pull "$IMAGE"
	exit $?
fi

# --- local AI services (LLM, speech) on this computer ----------------------------------------
if [ -n "$LOCAL_AI" ]; then
	if [ "$LOCAL_AI" = "9b" ]; then
		"$REPO_ROOT/docker/local-ai/up.bash" --9b
	else
		"$REPO_ROOT/docker/local-ai/up.bash"
	fi
fi

if [ "$PROVIDER" = "cloud" ] && [ ! -f "$SECRETS" ]; then
	echo "NOTE: --online needs API keys in $SECRETS"
	echo "      create it with: install -D -m 600 $REPO_ROOT/config/secrets.env.example $SECRETS"
fi

# --- create the container once ---------------------------------------------------------------
if [ -z "$(docker ps -q -f "name=^${NAME}$")" ]; then
	if [ -z "$MODE" ]; then
		echo "choose a mode: --sim or --robot-ip <MiRo IP> (see --help)"
		exit 2
	fi
	if [ "$MODE" = "robot" ] && [ -z "$ROBOT_IP" ]; then
		echo "the first robot terminal needs the robot's address: --robot-ip <MiRo IP>"
		exit 2
	fi
	docker rm -f "$NAME" >/dev/null 2>&1 || true

	ARGS=(run -d --rm --init --name "$NAME" --net=host --ipc=host)
	# defaults for terminals that do not pass a mode
	ARGS+=(-e "MIRO_MODE=$MODE" -e "MIRO_ROBOT_NAME=${ROBOT_NAME:-miro}" -e "MIRO_HOST_IP=${HOST_IP:-auto}")
	[ -n "$ROBOT_IP" ] && ARGS+=(-e "MIRO_ROBOT_IP=$ROBOT_IP")

	# code (read-write) and the shell environment script (read-only)
	ARGS+=(-v "$REPO_ROOT/src/heymiro:/root/mdk/catkin_ws/src/heymiro")
	ARGS+=(-v "$REPO_ROOT/docker/heymiro_env.bash:/opt/heymiro/heymiro_env.bash:ro")

	# YOLO weights, ultralytics settings etc. persist across runs (never in git)
	ARGS+=(-v "heymiro-cache:/root/.cache")

	# secrets: mounted read-only, never in the image or on the command line
	if [ -f "$SECRETS" ]; then
		SECRETS="$(cd "$(dirname "$SECRETS")" && pwd)/$(basename "$SECRETS")"
		perms="$(stat -c %a "$SECRETS" 2>/dev/null || echo 600)"
		if [ "$perms" != "600" ] && [ "$perms" != "400" ]; then
			echo "WARNING: $SECRETS is readable by others (mode $perms); run: chmod 600 $SECRETS"
		fi
		ARGS+=(-v "$SECRETS:/run/secrets/heymiro.env:ro" -e "HEYMIRO_SECRETS_FILE=/run/secrets/heymiro.env")
	fi

	# display (Gazebo, rqt)
	if [ "$X11" = 1 ] && [ -n "$DISPLAY" ]; then
		XAUTH=/tmp/.docker.xauth
		touch "$XAUTH"
		xauth nlist "$DISPLAY" 2>/dev/null | sed -e 's/^..../ffff/' | xauth -f "$XAUTH" nmerge - 2>/dev/null || true
		chmod 644 "$XAUTH"
		command -v xhost >/dev/null && xhost +SI:localuser:root >/dev/null
		ARGS+=(-e "DISPLAY=$DISPLAY" -e "QT_X11_NO_MITSHM=1" -e "XAUTHORITY=$XAUTH")
		ARGS+=(-v /tmp/.X11-unix:/tmp/.X11-unix:rw -v "$XAUTH:$XAUTH")
		[ -d /dev/dri ] && ARGS+=(--device /dev/dri)
	fi

	# sound: this computer's mic (simulator / --ext-mic) and speaker (MiRo's voice)
	if [ "$AUDIO" = 1 ]; then
		PULSE_DIR="$USER_RUNTIME/pulse"
		if [ -S "$PULSE_DIR/native" ]; then
			ARGS+=(-v "$PULSE_DIR/native:/run/pulse/native" -e "PULSE_SERVER=unix:/run/pulse/native")
			[ -f "$USER_HOME/.config/pulse/cookie" ] && ARGS+=(-v "$USER_HOME/.config/pulse/cookie:/root/.config/pulse/cookie:ro")
		else
			echo "NOTE: no PulseAudio socket at $PULSE_DIR/native - host audio may not work"
		fi
		[ -d /dev/snd ] && ARGS+=(--device /dev/snd --group-add audio)
	fi

	# GPU (YOLO; the local LLM/speech run in their own containers)
	if [ "$GPU" = "on" ]; then
		ARGS+=(--gpus all)
	elif [ "$GPU" = "auto" ]; then
		if docker info 2>/dev/null | grep -qiE 'runtimes:.*nvidia|nvidia\.com\/gpu' || command -v nvidia-container-cli >/dev/null 2>&1; then
			ARGS+=(--gpus all)
		fi
	fi

	[ "$PRIVILEGED" = 1 ] && ARGS+=(--privileged)

	echo "creating container '$NAME' from $IMAGE"
	docker "${ARGS[@]}" -e HEYMIRO_QUIET=1 "$IMAGE" sleep infinity >/dev/null
else
	echo "using running container '$NAME'"
fi

# --- open a terminal: mode and AI pipeline are per terminal -------------------------------------
EXEC=(exec -i)
[ -t 0 ] && [ -t 1 ] && EXEC+=(-t)
[ -n "$MODE" ] && EXEC+=(-e "MIRO_MODE=$MODE")
[ -n "$ROBOT_IP" ] && EXEC+=(-e "MIRO_ROBOT_IP=$ROBOT_IP")
[ -n "$HOST_IP" ] && EXEC+=(-e "MIRO_HOST_IP=$HOST_IP")
[ -n "$ROBOT_NAME" ] && EXEC+=(-e "MIRO_ROBOT_NAME=$ROBOT_NAME")
[ -n "$PROVIDER" ] && EXEC+=(-e "HEYMIRO_PROVIDER=$PROVIDER")
[ "$X11" = 1 ] && [ -n "$DISPLAY" ] && EXEC+=(-e "DISPLAY=$DISPLAY")

if [ "${CMD[*]}" = "bash" ]; then
	# interactive terminal: show the status banner first
	exec docker "${EXEC[@]}" "$NAME" bash -c 'source /opt/heymiro/heymiro_env.bash; heymiro_status; exec bash'
fi
exec docker "${EXEC[@]}" "$NAME" /opt/heymiro/entrypoint.sh "${CMD[@]}"
