#!/bin/bash
#
# Container entrypoint: set up ROS/MDK and the network for MIRO_MODE, print
# a short status banner (no secret values), then run the given command.

# shellcheck disable=SC1091
source /opt/heymiro/heymiro_env.bash

if [ -z "$HEYMIRO_QUIET" ]; then
	heymiro_status
fi

exec "$@"
