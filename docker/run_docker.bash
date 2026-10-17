#!/bin/bash
#
# Start (or attach to) the Hey MiRo environment container.
#
#   docker/run_docker.bash --robot-ip 192.168.0.42     # physical MiRo
#   docker/run_docker.bash --sim                       # Gazebo simulator
#   docker/run_docker.bash                             # again: opens another terminal
#
# The image holds the environment only. The two project folders of this
# repository are bind-mounted into the container, so edits on the host (or
# a `git pull`) are visible inside immediately:
#
#   src/cloud_demo_robot -> /root/mdk/catkin_ws/src/cloud_demo_robot
#   src/cloud_demo_sim   -> /root/mdk/catkin_ws/src/cloud_demo_sim
#
# Secrets (API keys) are read from ~/.config/heymiro/secrets.env (create it
# from config/secrets.env.example) and mounted read-only; they are never
# baked into the image or passed on the command line.
#
# Options:
#   --robot-ip IP        robot mode: ROS master on the MiRo at IP
#   --sim                sim mode: local roscore + Gazebo
#   --host-ip IP|auto    this computer's IP as seen by the robot (default auto)
#   --robot-name NAME    MiRo robot name (default miro)
#   --gpu auto|on|off    pass the NVIDIA GPU through (default auto)
#   --image IMAGE        default: the image CI built for this git branch
#                        (ghcr.io/heymiro/ext_cog_arch:<branch>), else :latest
#   --name NAME          container name (default heymiro)
#   --secrets FILE       secrets file (default ~/.config/heymiro/secrets.env)
#   --no-audio           do not pass the host sound server / devices
#   --no-x11             do not pass the X display (no Gazebo GUI)
#   --privileged         run privileged (normally not needed)
#   --local-ai [9b]      start the local AI services first (docker/local-ai/up.bash)
#   --pull               download the newest environment image, then exit
#   -- CMD ...           command to run instead of an interactive bash

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
HOST_IP="auto"
ROBOT_NAME="miro"
GPU="auto"
IMAGE=""
NAME="heymiro"
SECRETS="$USER_HOME/.config/heymiro/secrets.env"
AUDIO=1
X11=1
PRIVILEGED=0
LOCAL_AI=""
PULL=0
CMD=(bash)

usage() {
	sed -n '2,/^set -e/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//; /^set -e/d'
}

while [ $# -gt 0 ]; do
	case "$1" in
		--robot-ip) MODE="robot"; ROBOT_IP="$2"; shift 2 ;;
		--sim) MODE="sim"; shift ;;
		--host-ip) HOST_IP="$2"; shift 2 ;;
		--robot-name) ROBOT_NAME="$2"; shift 2 ;;
		--gpu) GPU="$2"; shift 2 ;;
		--image) IMAGE="$2"; shift 2 ;;
		--name) NAME="$2"; shift 2 ;;
		--secrets) SECRETS="$2"; shift 2 ;;
		--no-audio) AUDIO=0; shift ;;
		--no-x11) X11=0; shift ;;
		--privileged) PRIVILEGED=1; shift ;;
		--pull) PULL=1; shift ;;
		--local-ai)
			LOCAL_AI="default"; shift
			if [ "$1" = "9b" ]; then LOCAL_AI="9b"; shift; fi ;;
		--) shift; CMD=("$@"); break ;;
		-h|--help) usage; exit 0 ;;
		*) echo "unknown option: $1 (see --help)"; exit 2 ;;
	esac
done

# --- local AI services (LLM, speech) on this computer's GPU ------------------------------
if [ -n "$LOCAL_AI" ]; then
	if [ "$LOCAL_AI" = "9b" ]; then
		"$REPO_ROOT/docker/local-ai/up.bash" --9b
	else
		"$REPO_ROOT/docker/local-ai/up.bash"
	fi
fi

# --- which environment image ----------------------------------------------------------------
# CI publishes one per branch (slashes become dashes); prefer the one for
# the branch checked out here, so the code and its environment match
if [ -z "$IMAGE" ]; then
	REPO_IMAGE="ghcr.io/heymiro/ext_cog_arch"
	IMAGE="$REPO_IMAGE:latest"
	branch="$(git -C "$REPO_ROOT" branch --show-current 2>/dev/null | tr '/' '-')"
	if [ -n "$branch" ] && [ "$branch" != "main" ]; then
		if docker image inspect "$REPO_IMAGE:$branch" >/dev/null 2>&1 || \
			docker manifest inspect "$REPO_IMAGE:$branch" >/dev/null 2>&1; then
			IMAGE="$REPO_IMAGE:$branch"
		fi
	fi
fi

if [ "$PULL" = 1 ]; then
	echo "downloading the newest $IMAGE ..."
	docker pull "$IMAGE"
	exit $?
fi

# docker -v needs an absolute path (a relative one names a volume instead)
if [ -e "$SECRETS" ]; then
	SECRETS="$(cd "$(dirname "$SECRETS")" && pwd)/$(basename "$SECRETS")"
fi

# --- already running? open another terminal in it --------------------------------
if [ -n "$(docker ps -q -f "name=^${NAME}$")" ]; then
	running_env="$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$NAME")"
	running_mode="$(sed -n 's/^MIRO_MODE=//p' <<< "$running_env")"
	running_ip="$(sed -n 's/^MIRO_ROBOT_IP=//p' <<< "$running_env")"
	running_host="$(sed -n 's/^MIRO_HOST_IP=//p' <<< "$running_env")"
	if [ -n "$MODE" ] && [ "$MODE" != "$running_mode" ]; then
		echo "container '$NAME' is already running in $running_mode mode; stop it first (docker stop $NAME) to switch to $MODE"
		exit 1
	fi
	if { [ -n "$ROBOT_IP" ] && [ "$ROBOT_IP" != "$running_ip" ]; } || { [ "$HOST_IP" != "auto" ] && [ "$HOST_IP" != "$running_host" ]; }; then
		echo "container '$NAME' is running with robot IP '${running_ip:-none}' / host IP '${running_host:-auto}';"
		echo "stop it first (docker stop $NAME) to use the new addresses"
		exit 1
	fi
	echo "attaching a new terminal to running container '$NAME' ($running_mode mode)"
	if [ -t 0 ]; then
		exec docker exec -it "$NAME" "${CMD[@]}"
	fi
	exec docker exec -i "$NAME" "${CMD[@]}"
fi

if [ -z "$MODE" ]; then
	echo "choose a mode: --robot-ip <MiRo IP> or --sim (see --help)"
	exit 2
fi

# a TTY only when there is one (scripts and CI run without)
ARGS=(run -i --rm --name "$NAME" --net=host --ipc=host)
[ -t 0 ] && ARGS+=(-t)
ARGS+=(-e "MIRO_MODE=$MODE" -e "MIRO_ROBOT_NAME=$ROBOT_NAME" -e "MIRO_HOST_IP=$HOST_IP")
[ "$MODE" = "robot" ] && ARGS+=(-e "MIRO_ROBOT_IP=$ROBOT_IP")

# --- code: the two project folders, read-write --------------------------------------
for project in cloud_demo_robot cloud_demo_sim; do
	ARGS+=(-v "$REPO_ROOT/src/$project:/root/mdk/catkin_ws/src/$project")
done

# YOLO weights, ultralytics settings etc. persist across runs (never in git)
ARGS+=(-v "heymiro-cache:/root/.cache")

# --- secrets: mounted read-only, never in the image or on the command line ------------
if [ -f "$SECRETS" ]; then
	perms="$(stat -c %a "$SECRETS" 2>/dev/null || echo 600)"
	if [ "$perms" != "600" ] && [ "$perms" != "400" ]; then
		echo "WARNING: $SECRETS is readable by others (mode $perms); run: chmod 600 $SECRETS"
	fi
	ARGS+=(-v "$SECRETS:/run/secrets/heymiro.env:ro" -e "HEYMIRO_SECRETS_FILE=/run/secrets/heymiro.env")
else
	echo "NOTE: no secrets file at $SECRETS - cloud features (conversation, TTS, wake words) will be off."
	echo "      create it with: install -D -m 600 $REPO_ROOT/config/secrets.env.example $SECRETS"
fi

# --- display (Gazebo, rqt) ---------------------------------------------------------------
if [ "$X11" = 1 ] && [ -n "$DISPLAY" ]; then
	XAUTH=/tmp/.docker.xauth
	touch "$XAUTH"
	xauth nlist "$DISPLAY" 2>/dev/null | sed -e 's/^..../ffff/' | xauth -f "$XAUTH" nmerge - 2>/dev/null || true
	chmod 644 "$XAUTH"
	if command -v xhost >/dev/null; then
		xhost +SI:localuser:root >/dev/null
		trap 'xhost -SI:localuser:root >/dev/null 2>&1 || true' EXIT
	fi
	ARGS+=(-e "DISPLAY=$DISPLAY" -e "QT_X11_NO_MITSHM=1" -e "XAUTHORITY=$XAUTH")
	ARGS+=(-v /tmp/.X11-unix:/tmp/.X11-unix:rw -v "$XAUTH:$XAUTH")
	[ -d /dev/dri ] && ARGS+=(--device /dev/dri)
fi

# --- sound (host mic for the bridge, speaker for MiRo's voice) -------------------------------
if [ "$AUDIO" = 1 ]; then
	PULSE_DIR="$USER_RUNTIME/pulse"
	if [ -S "$PULSE_DIR/native" ]; then
		ARGS+=(-v "$PULSE_DIR/native:/run/pulse/native" -e "PULSE_SERVER=unix:/run/pulse/native")
		[ -f "$USER_HOME/.config/pulse/cookie" ] && ARGS+=(-v "$USER_HOME/.config/pulse/cookie:/root/.config/pulse/cookie:ro")
	else
		echo "NOTE: no PulseAudio socket at $PULSE_DIR/native - host audio may not work"
	fi
	if [ -d /dev/snd ]; then
		ARGS+=(--device /dev/snd --group-add audio)
	fi
fi

# --- GPU ---------------------------------------------------------------------------------------
if [ "$GPU" = "on" ]; then
	ARGS+=(--gpus all)
elif [ "$GPU" = "auto" ]; then
	if docker info 2>/dev/null | grep -qiE 'runtimes:.*nvidia|nvidia\.com\/gpu' || command -v nvidia-container-cli >/dev/null 2>&1; then
		ARGS+=(--gpus all)
	fi
fi

[ "$PRIVILEGED" = 1 ] && ARGS+=(--privileged)

ARGS+=("$IMAGE" "${CMD[@]}")

echo "starting $IMAGE ($MODE mode) as '$NAME'"
docker "${ARGS[@]}"
