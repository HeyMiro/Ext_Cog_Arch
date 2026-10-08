# cloud_demo_robot — Hey MiRo on a physical MiRo-e

This is the HRI '25 "Hey MiRo!" demo, built in the shape of the MiRo MDK demo (`client_demo.py`, `node_*.py`,
`action/action_*.py`). It runs off-board, on your computer (P4), against a MiRo-e on the same network. For the
simulator, use [`../cloud_demo_sim`](../cloud_demo_sim). The two folders hold the same code; only
`config/heymiro.yaml`, the launcher and this README differ.

```bash
# inside the container: ./docker/run_docker.bash --robot-ip <MiRo IP>
cd ~/mdk/catkin_ws/src/cloud_demo_robot
./run_robot.sh              # speech from MiRo's own microphones
./run_robot.sh --ext-mic    # speech from this computer's microphone (noisy rooms)
```

**Turn off demo mode in the MiRo app first.** Otherwise the robot's on-board demo and this one both drive the
motors.

## Processes (clients)

`run_robot.sh` starts the same clients the MiRo runs on board, plus a new one:

| Client | Command | What it does |
|---|---|---|
| main | `core/client_demo.py -` | 50 Hz loop: affect, expression, action selection, and all forebrain nodes |
| caml / camr | `core/client_demo.py - caml` / `- camr` | stock camera processing: faces, motion, AprilTags |
| mics | `core/client_demo.py - mics` | stock sound localisation |
| scene | `core/client_demo.py - scene` | YOLO objects and mediapipe gestures; runs only while the main client asks for it |
| bridge | `tools/host_audio_bridge.py` | with `--ext-mic`: this computer's mic → `/miro/host/mics` |

You can also start each client by hand in its own terminal, exactly as with the stock MDK demo.

## Layout

```
core/
  client_demo.py            entry point (stock structure; forebrain wiring added)
  pars.py                   stock parameters + pars.forebrain (heymiro_config) + forebrain priority band
  heymiro_config.py         settings, secrets and API routes (no ROS)
  audio_util.py             resampling, MiRo mic/speaker formats (no ROS)
  node_*.py                 stock nodes (lower, affect, express, spatial, detect_*, loop, action)
  node_hearing.py           forebrain: wake words, voice activity, utterances, ear/tail scratch noise
  node_voice.py             forebrain: clips, speech and music (host speaker or MiRo's speaker)
  node_forebrain.py         forebrain: events → intentions; task slot; top-down modulation
  node_dialogue.py          forebrain: runs the conversation (forebrain/dialogue.py)
  node_cues.py              forebrain: eyelid/ear attention cues
  node_scene.py             forebrain: scene awareness (in the "- scene" client)
  action/
    action_*.py             stock actions (mull, orient, approach, flee, avert, halt, retreat, special)
    forebrain_action.py     base class of the forebrain actions
    action_converse.py      the conversation's body language
    action_dance.py         dancing to music
    action_tricks.py        tricks for hand signs
    action_identify.py      "what is this?" object game
  forebrain/                pure logic: dialogue state machine, segmenter, wake words, touch noise, clips
  cloud/                    off-board services: OpenAI (STT + chat), ElevenLabs, songs/Spotify, worker
  choreography/             moves, lights and the beat-locked DanceController
config/                     heymiro.yaml, persona.md, clips.yaml, songs.yaml
assets/                     clips/, songs/, wakewords/, models/
tools/                      host_audio_bridge.py, mic_check.py, make_clip.py, estimate_bpm.py, demo_launch.bash
tests/                      unit tests: python3 -m unittest discover -s tests
```

## Nodes and actions

A **node** does processing every tick, whatever MiRo is doing: sensing, affect, expression, hearing. Nodes are
created in `DemoNodes.instantiate()` in `client_demo.py` and ticked in `DemoNodes.tick()`. They share
`self.state`, `self.input`, `self.output` and each other (`self.nodes.<name>`).

An **action** is a behaviour that competes for MiRo's body in the basal ganglia. Every tick, each action proposes
a priority (`compute_priority()`), and the highest wins (`start()`, then `service()` each tick). A higher
priority can pre-empt it (`stop()`). Only one action runs at a time. The forebrain actions sit in their own band
(0.84): above orienting, below halting and cliff aversion.

### Adding a forebrain action

1. Create `core/action/action_wave.py`:

   ```python
   from . import forebrain_action
   from choreography import moves

   class ActionWave(forebrain_action.ForebrainAction):

   	NAME = "wave"
   	TASK = "wave"                 # run when the forebrain task "wave" is requested

   	def wanted(self):
   		return self.task_wanted()

   	def on_start(self):
   		self.forebrain.say("Hello!")

   	def on_service(self):
   		self.set_head(moves.nod(moves.NEUTRAL, self.elapsed()))
   		if self.elapsed() > 3.0:
   			self.finish()

   	def on_stop(self, reason):
   		pass
   ```

2. Register it in `core/node_action.py`: import it, then add `ActionWave(self)` to `self.actions`.
3. Trigger it:
   - from code, with `self.nodes.forebrain.request_task("wave")`;
   - from the command topic, after adding a word in `node_forebrain.parse_command`;
   - from the LLM, after adding a command flag in `cloud/llm.py` and `config/persona.md`.

### Adding a node

1. Create `core/node_thing.py`, with a class `NodeThing(node.Node)` that has a `tick()` (and a `shutdown()` if it
   starts threads or opens devices).
2. In `client_demo.py`: `from node_thing import *`, add `self.thing = NodeThing(sys)` in
   `DemoNodes.instantiate()` for the right client type, and call `self.thing.tick()` in `DemoNodes.tick()`.
   Never block in `tick()`. Use the forebrain's worker (`self.nodes.forebrain.worker.submit(...)`) for anything
   slow, and poll the returned future.
3. Reach it from actions with `self.parent.nodes.thing`.

### Kinematics in one paragraph

`self.kc` (for actions) / `self.kc_m` (for nodes) is MiRo's kinematic chain. The configuration
`[tilt, lift, yaw, pitch]` is in radians; `kc.getConfig()` and `kc.setConfig(cfg)` read and set it.
`choreography.moves.clip_config()` keeps a configuration inside the joint limits. Wheels are driven by pushes:
`apply_push_body(np.array([vx, wz, 0]), flags=miro.constants.PUSH_FLAG_VELOCITY)`, or
`ForebrainAction.body_velocity(vx, wz)`. The main client publishes the result on `control/kinematic_joints` and
`control/cmd_vel` every tick. Cosmetic joints (ears, tail, eyelids) and the LEDs go through `node_express`; a
forebrain action borrows them with `self.override(channel, value)`.

## Configuration and secrets

- Settings: `config/heymiro.yaml`. See [`docs/configuration.md`](../../docs/configuration.md).
- Keys and API routes: `~/.config/heymiro/secrets.env`. See the top-level README.
- MiRo's personality for the LLM: `config/persona.md`.
- New voice clips: `tools/make_clip.py "Hello there!" --category greeting`.
- New songs: copy the mp3 into `assets/songs/`, measure its tempo with `tools/estimate_bpm.py`, and add an entry
  to `config/songs.yaml`.

## Differences from the sim copy

| Setting | here (robot) | cloud_demo_sim |
|---|---|---|
| speech input | `sensors/mics` (MiRo's ears), or `host/mics` with `--ext-mic` | `host/mics` (host microphone bridge) |
| sound orienting | on | off (`S`); a mono host mic has no direction |
| ear/tail scratch detection | on | off |
| face the user | AprilTag | none |
| scene rate | 5 Hz | 1 Hz |

After changing shared code, keep both copies in sync with `python3 tools/check_copies.py --fix-from robot`, run
from the repository root.
