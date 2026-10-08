# Running Hey MiRo in the Gazebo simulator

The MDK's Gazebo simulator models MiRo's body, cameras, sonar and cliff sensors, but **not sound**. It does not
publish `/miro/sensors/mics`, and nothing plays `/miro/control/stream`. As a result:

- the old demo never heard anything in simulation;
- it also never updated MiRo's emotions, because the affect node ticked from the microphone callback. That is
  fixed now: affect ticks in the 50 Hz main loop again.

`src/cloud_demo_sim` runs the same code as `src/cloud_demo_robot`, with settings for the simulator, and adds a
**host audio bridge**.

## The host audio bridge: a new mic topic

`tools/host_audio_bridge.py` is a small ROS node. It captures your computer's microphone and publishes it as
**`/miro/host/mics`** (`std_msgs/Int16MultiArray`):

| Property | Value |
|---|---|
| layout | exactly MiRo's `sensors/mics`: 4 channels × 500 samples, channel-major (`data[0:500]` = left ear, `[500:1000]` = right, `[1000:1500]` = centre, `[1500:2000]` = tail) |
| rate | 20 kHz, one message every 25 ms (40 Hz) |
| channels | the mono microphone is copied to all four channels; `--stereo` maps a real stereo pair to left/right |

Because the format is identical, both the hearing pipeline (wake word, voice activity detection, utterance
capture) and the stock MiRo sound localiser read it unchanged. Which topic they read is set in
`config/heymiro.yaml`:

```yaml
hearing:
  input_topic: host/mics      # speech (robot copy: sensors/mics)
  salience_topic: host/mics   # the "- mics" sound-localisation client
```

A mono microphone has no left/right difference, so every sound seems to come from straight ahead. That is why the
sim's demo flags include `S` ("disable attend sound"): MiRo does not orient to every word. Loudness still drives
arousal.

With **`--speaker`** (`run_sim.sh` uses it), the bridge also:

- plays `/miro/control/stream` (MiRo's 8 kHz speaker topic) through your speaker;
- publishes `/miro/sensors/stream` `[space, total]` at 50 Hz, emulating MiRo's speaker buffer.

Together these let the `voice.backend: miro` mode and the stock MDK sounds be tested in simulation. The bridge
refuses `--speaker` on a real robot.

Speech, clips and music normally play straight on your speaker (`voice.backend: host`), in sim and on the robot.

Bridge options:

```
host_audio_bridge.py [--device D] [--gain-db G] [--topic host/mics] [--stereo] [--speaker]
host_audio_bridge.py --wav FILE [--loop]       # replay a recording instead of the mic
host_audio_bridge.py --list-devices
```

## Running

```bash
./docker/run_docker.bash --sim
cd ~/mdk/catkin_ws/src/cloud_demo_sim
./run_sim.sh                      # roscore + Gazebo + bridge + demo clients
./run_sim.sh --no-gazebo          # attach to a simulator you started yourself
```

`run_sim.sh` runs these steps:

1. Starts `roscore` if none is running.
2. Starts `~/mdk/sim/launch_sim.sh` (any extra arguments are passed on), then waits for `/miro/sensors/package`.
3. Starts the bridge, then the `- caml`, `- camr`, `- mics` and `- scene` clients, logging to
   `~/.ros/heymiro/<date>/`.
4. Runs the main client in the foreground. Ctrl-C stops everything the script started.

To start each piece in its own terminal, run `docker/run_docker.bash` again for each new shell:

```bash
roscore
~/mdk/sim/launch_sim.sh
python3 tools/host_audio_bridge.py --speaker
cd core && ./client_demo.py - caml      # and - camr, - mics, - scene
cd core && ./client_demo.py
```

## Checks

```bash
rostopic hz /miro/host/mics                 # ~40 Hz
python3 tools/mic_check.py --topic host/mics    # level meter of what MiRo hears
python3 tools/mic_check.py --wakeword       # offline wake-word + VAD test with your keys
rostopic echo /miro/core/forebrain/state    # what the forebrain is doing
```

## Differences from the robot (all set in `config/heymiro.yaml`)

| Setting | Robot | Sim | Why |
|---|---|---|---|
| `platform` | robot | sim | checked against the simulator flag in `sensors/package` |
| `mdk.demo_flags` | `vx` | `vxS` | no sound orienting from a mono host microphone |
| `hearing.input_topic`, `salience_topic` | `sensors/mics` | `host/mics` | the simulator has no microphones |
| `hearing.self_hearing_guard_s` | 0.3 | 0.6 | the laptop speaker is close to the laptop mic |
| `hearing.touch_from_mics` | true | false | the "scratching the ears" detector only works on MiRo's own microphones |
| `converse.face_user` | apriltag | none | no AprilTag in the simulated world |
| `scene.rate_hz` | 5 | 1 | leave CPU for Gazebo |

## Caveats

- **Simulated sensor content.** Which Gazebo sensors are populated (touch, light, cliff) depends on the MDK
  simulator. The main client prints `platform: simulator` when it starts.
  - If the simulated light is dark, MiRo gets sleepy quickly. Add `s` to `mdk.demo_flags`.
  - If MiRo keeps backing away from cliffs that aren't there, add `cu`.
- **No echo cancellation.** MiRo stops listening while it speaks (and for `self_hearing_guard_s` afterwards), but
  headphones work best. Alternatively, use PulseAudio's `module-echo-cancel` on the host and pass its source with
  `--device`.
- **Real-time factor.** If Gazebo runs slower than real time, the 50 Hz loop slows with it. Audio timing (the 3 s
  of silence that ends an utterance) counts audio samples, so it is unaffected.
