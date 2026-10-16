#!/bin/bash
#
# Start the Hey MiRo local AI services and wait until they are ready.
#
#   docker/local-ai/up.bash            # Qwen3.5-4B (default; leaves room for Gazebo)
#   docker/local-ai/up.bash --9b       # Qwen3.5-9B (physical robot: more GPU memory free)
#   docker/local-ai/up.bash --down     # stop them
#
# First time: docker/local-ai/download_models.bash [--9b]
# Then run the demo with ai.provider: local (the default in config/heymiro.yaml).

set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE=(docker compose -f "$HERE/compose.yaml")

case "$1" in
	--down) "${COMPOSE[@]}" down; exit 0 ;;
	--9b)
		export HEYMIRO_LLM_GGUF=/models/unsloth/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf
		export HEYMIRO_LLM_MMPROJ=/models/unsloth/Qwen3.5-9B-GGUF/mmproj-F16.gguf ;;
	"") ;;
	-h|--help) sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
	*) echo "unknown option: $1"; exit 2 ;;
esac

if ! docker volume inspect heymiro-models >/dev/null 2>&1; then
	echo "no models yet: run $HERE/download_models.bash first"
	exit 1
fi
if ! docker info 2>/dev/null | grep -qi 'runtimes:.*nvidia'; then
	echo "Docker cannot use the GPU. Run once:"
	echo "  sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"
	exit 1
fi

"${COMPOSE[@]}" up -d --build

echo -n "waiting for the models to load"
for _ in $(seq 1 120); do
	if curl -fsS http://127.0.0.1:8080/health >/dev/null 2>&1 && curl -fsS http://127.0.0.1:8090/health >/dev/null 2>&1; then
		echo " ready"
		curl -fsS http://127.0.0.1:8090/health; echo
		exit 0
	fi
	echo -n "."
	sleep 2
done
echo " timed out; see: docker compose -f $HERE/compose.yaml logs"
exit 1
