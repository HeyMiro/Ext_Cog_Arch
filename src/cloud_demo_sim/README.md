# cloud_demo_sim — Hey MiRo in the MiRo Gazebo simulator

This is the same HRI '25 "Hey MiRo!" demo as [`../cloud_demo_robot`](../cloud_demo_robot), set up for the MDK's
Gazebo simulator. The code is identical. Only `config/heymiro.yaml`, the launcher (`run_sim.sh`) and this README
differ.

The simulator has **no microphones and no speaker**. To fill that gap:

- `tools/host_audio_bridge.py` publishes this computer's microphone on a new topic, **`/miro/host/mics`**, in
  exactly the format of MiRo's own `/miro/sensors/mics` (4 channels × 500 samples at 20 kHz). The hearing pipeline
  (wake words, voice activity detection, utterances) and the stock sound localiser read it unchanged.
- MiRo's voice, clips and music play on this computer's speaker. With `--speaker`, the bridge also plays
  `/miro/control/stream` (MiRo's speaker topic) and emulates its buffer feedback on `/miro/sensors/stream`.

```bash
# inside the container: ./docker/run_docker.bash --sim
cd ~/mdk/catkin_ws/src/cloud_demo_sim
./run_sim.sh                   # roscore + Gazebo + audio bridge + all demo clients
./run_sim.sh --no-gazebo       # attach to a simulator that is already running
./run_sim.sh --device "USB"    # pick a microphone (python3 tools/mic_check.py --list)
```

Then say **"Hey MiRo"** into your microphone. Wear headphones so MiRo doesn't hear itself, or rely on the
self-hearing guard. Without wake-word keys, drive it from another terminal:

```bash
rostopic pub -1 /miro/core/forebrain/command std_msgs/String "data: 'dance dancing queen'"
rostopic pub -1 /miro/core/forebrain/command std_msgs/String "data: 'stop'"
```

## Checks

```bash
rostopic hz /miro/host/mics                     # ~40 Hz while the bridge runs
python3 tools/mic_check.py --topic host/mics    # level meter of what MiRo hears
python3 tools/mic_check.py --wakeword           # offline wake-word + VAD test with your keys
rostopic echo /miro/core/forebrain/state        # what the forebrain is doing
```

## Settings that differ from the robot

| Setting | sim (here) | robot |
|---|---|---|
| `platform` | sim | robot |
| `mdk.demo_flags` | `vxS`: no orienting to sound, because a mono host mic has no direction | `vx` |
| `hearing.input_topic` / `salience_topic` | `host/mics` | `sensors/mics` |
| `hearing.self_hearing_guard_s` | 0.6 | 0.3 |
| `hearing.touch_from_mics` | false | true |
| `converse.face_user` | none | apriltag |
| `scene.rate_hz` | 1 | 5 |

If the simulated light makes MiRo sleepy, add `s` to `mdk.demo_flags`. If MiRo backs away from cliffs that
aren't there, add `cu`.

Everything else (the code structure, how to add actions and nodes, configuration and secrets) is described in
[`../cloud_demo_robot/README.md`](../cloud_demo_robot/README.md) and [`docs/simulation.md`](../../docs/simulation.md).
After changing shared code, run `python3 tools/check_copies.py --fix-from sim` from the repository root to copy it
to the robot version.
