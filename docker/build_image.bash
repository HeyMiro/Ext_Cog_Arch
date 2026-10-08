#!/bin/bash
#
# Build the Hey MiRo environment image locally.
#
#   docker/build_image.bash                         # CUDA torch, MDK from Google Drive
#   docker/build_image.bash --mdk ~/mdk_2-230105.tgz
#   docker/build_image.bash --cpu --tag heymiro-env:cpu
#
# then: docker/run_docker.bash --image heymiro-env:local --sim
#
# Normally you do not need this: CI builds and publishes
# ghcr.io/heymiro/ext_cog_arch on every change to docker/.

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAG="heymiro-env:local"
TORCH_VARIANT="cu121"
TARGET="env"

while [ $# -gt 0 ]; do
	case "$1" in
		--tag) TAG="$2"; shift 2 ;;
		--cpu) TORCH_VARIANT="cpu"; shift ;;
		--mdk) cp "$2" "$REPO_ROOT/docker/mdk/mdk_2-230105.tgz"; shift 2 ;;
		--test) TARGET="test"; shift ;;
		-h|--help) sed -n '2,13p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
		*) echo "unknown option: $1"; exit 2 ;;
	esac
done

cd "$REPO_ROOT"
DOCKER_BUILDKIT=1 docker build \
	-f docker/Dockerfile \
	--target "$TARGET" \
	--build-arg TORCH_VARIANT="$TORCH_VARIANT" \
	-t "$TAG" \
	.

echo "built $TAG"
