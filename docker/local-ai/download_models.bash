#!/bin/bash
#
# Download the pinned local-AI models (models.env) once. Needs internet;
# everything after this runs offline.
#
#   docker/local-ai/download_models.bash          # default (4B LLM)
#   docker/local-ai/download_models.bash --9b     # also the 9B LLM
#
# LLM, speech-to-text, voice and end-of-turn models land in the volume
# heymiro-models as /models/<repo>/<file>; the YOLO weights for the scene
# client land in the volume heymiro-cache (mounted at /root/.cache in the
# ROS container). Re-running only fetches what is missing.

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

# YOLO weights for the scene client (heymiro-cache -> /root/.cache/heymiro/weights)
yolo_url="$(sed -n "s/^YOLO=//p" "$HERE/models.env")"
[ -n "$yolo_url" ] || { echo "no YOLO in models.env"; exit 1; }
docker volume create heymiro-cache >/dev/null
docker run --rm -v heymiro-cache:/cache python:3.12-slim python3 -c '
import os, sys, urllib.request
url = sys.argv[1]
path = os.path.join("/cache/heymiro/weights", os.path.basename(url))
os.makedirs(os.path.dirname(path), exist_ok=True)
if os.path.isfile(path) and os.path.getsize(path) > 100000:
	print("== YOLO weights already there: " + os.path.basename(path))
else:
	print("== YOLO weights " + os.path.basename(path))
	urllib.request.urlretrieve(url, path + ".part")
	os.replace(path + ".part", path)
print("%.0f MB  heymiro-cache:/heymiro/weights/%s" % (os.path.getsize(path) / 1e6, os.path.basename(path)))
' "$yolo_url"
