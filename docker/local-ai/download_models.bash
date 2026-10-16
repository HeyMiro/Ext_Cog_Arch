#!/bin/bash
#
# Download the pinned local-AI models (models.env) into the Docker volume
# heymiro-models, once. Needs internet; everything after this runs offline.
#
#   docker/local-ai/download_models.bash          # default (4B LLM)
#   docker/local-ai/download_models.bash --9b     # also the 9B LLM
#
# Files land in the volume as /models/<repo>/<file>. Re-running only
# fetches what is missing.

set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WANT=(LLM_4B STT TTS TURN)
[ "$1" = "--9b" ] && WANT+=(LLM_9B)

specs=()
for name in "${WANT[@]}"; do
	line="$(sed -n "s/^${name}=//p" "$HERE/models.env")"
	[ -n "$line" ] || { echo "no $name in models.env"; exit 1; }
	specs+=("$line")
done

docker volume create heymiro-models >/dev/null
docker run --rm -v heymiro-models:/models -e HF_HUB_DISABLE_TELEMETRY=1 python:3.12-slim bash -c '
	set -e
	pip install --quiet --root-user-action=ignore "huggingface_hub>=0.25,<1.0"
	for spec in "$@"; do
		repo="${spec%%@*}"; rest="${spec#*@}"; rev="${rest%%:*}"; files="${rest#*:}"
		echo "== $repo @ ${rev:0:8}"
		if [ "$files" = "*" ]; then
			huggingface-cli download "$repo" --revision "$rev" --local-dir "/models/$repo" --quiet
		else
			huggingface-cli download "$repo" ${files//,/ } --revision "$rev" --local-dir "/models/$repo" --quiet
		fi
	done
	rm -rf /models/*/*/.cache
	du -sh /models/*/*
' _ "${specs[@]}"
