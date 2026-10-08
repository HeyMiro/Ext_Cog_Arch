#!/bin/bash
#
# Environment smoke test (run by the Dockerfile "test" stage in CI).
# Checks the environment only: no robot, no audio device, no API keys.

# shellcheck disable=SC1091
source /opt/heymiro/heymiro_env.bash
set -e

echo "== python"
python3 - <<'PY'
import sys
assert sys.version_info[:2] == (3, 8), sys.version
import importlib
mods = ["rospy", "miro2", "miro2_msg.msg", "cv2", "cv_bridge", "numpy", "scipy", "yaml",
	"requests", "openai", "pvporcupine", "pvcobra", "ultralytics", "mediapipe",
	"mediapipe.tasks.python.vision", "torch", "torchvision", "soundfile", "apriltag", "tf"]
for m in mods:
	importlib.import_module(m)
	print("  import", m, "OK")
from importlib import metadata
v = metadata.version("pvporcupine")
assert v.startswith("3.0."), "pvporcupine must be 3.0.x to load *_v3_0_0.ppn, got " + v
import numpy, torch, cv2
print("  numpy", numpy.__version__, "| torch", torch.__version__, "| cv2", cv2.__version__, "| pvporcupine", v)
try:
	import sounddevice
	print("  sounddevice", sounddevice.__version__)
except OSError as e:
	raise SystemExit("PortAudio missing: " + str(e))
PY

echo "== MDK"
test -f /root/mdk/setup.bash
test -d /root/mdk/sim
find /root/mdk/sim -maxdepth 1 | head -n 20 || true
test -n "$MIRO_DIR_STATE" || { echo "MIRO_DIR_STATE not set by the MDK"; exit 1; }
test -n "$MIRO_DIR_CONFIG" || { echo "MIRO_DIR_CONFIG not set by the MDK"; exit 1; }

echo "== tools"
for t in ffmpeg catkin roscore gzserver pactl; do
	command -v "$t" >/dev/null || { echo "missing $t"; exit 1; }
	echo "  $t OK"
done

echo "== roscore"
roscore >/tmp/roscore.log 2>&1 &
pid=$!
for _ in $(seq 1 30); do
	rostopic list >/dev/null 2>&1 && break
	sleep 1
done
rostopic list
kill "$pid"
wait "$pid" 2>/dev/null || true

echo "== no project code, assets or keys baked in"
if [ -e /root/mdk/catkin_ws/src/cloud_demo_robot ] || [ -e /root/mdk/catkin_ws/src/cloud_demo_sim ]; then
	echo "project folders must be bind-mounted, not baked in"; exit 1
fi
if find /root /opt -xdev \( -name '*.ppn' -o -name 'api_key*.txt' -o -name 'secrets.env' \) 2>/dev/null | grep -q .; then
	echo "found key/asset files in the image"; exit 1
fi
if env | grep -E '^(OPENAI|ELEVENLABS|PICOVOICE|SPOTIFY)_' >/dev/null; then
	echo "secret variables present in the image environment"; exit 1
fi

echo "smoke test passed"
