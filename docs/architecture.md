# Architecture

## Background: MiRo's layered, brain-based control

MiRo-e's control system follows Mitchinson & Prescott (2016), *MIRO: A robot "mammal" with a biomimetic
brain-based control system*. Its control loops are **layered** like the vertebrate nervous system. Lower layers
keep working on their own, and higher layers **modulate** them (suppress, potentiate, prime) rather than replace
them.

- **Spinal cord (P1)**
  - Signal conditioning.
  - Reflexes: the *cliff reflex* stops the wheels at an edge; the *freeze reflex* stops all movement when MiRo is
    picked up.
- **Brainstem (P2)**
  - **Affect** as a circumplex of *valence* and *arousal*, driven by touch, sound, light and the time of day,
    with a sleep oscillator.
  - **Social pattern generators** (SPG): lights, ears, tail and eyelids express that affect.
  - A **spatial behaviour system** (superior colliculus) builds a salience map from vision and from sound
    localised with a Jeffress model. It generates plans: *orient*, *approach*, *flee*, *avert*.
  - The **basal ganglia** (BG) pick one plan at a time, with persistence and pre-emption, for the **motor
    pattern generator** (MPG).
- **Forebrain (P3)**: higher systems that *modulate* the lower ones.
- **P4**: off-board processing. It is slower but more capable.

The stock MDK demo (`client_demo.py`) implements P1–P2 as a set of nodes (`node_lower`, `node_affect`,
`node_express`, `node_spatial`, `node_detect_*`, `node_action` + `action/basal_ganglia.py`, `node_loop`) spread
over several processes: the main client plus the `caml`, `camr` and `mics` clients.

## What Hey MiRo adds (HRI '25)

The HRI '25 paper (Fig. 2) adds **forebrain** components. In this code base they are new nodes and actions that
plug into the stock architecture without replacing it, plus **P4** clients for cloud services.

| Layer | Stock modules (unchanged unless noted) | Hey MiRo additions |
|---|---|---|
| Spinal cord (P1) | `node_lower` (touch, jerk, sound level); cliff and freeze reflex flags | `ActionAvert` (cliff aversion) restored |
| Brainstem: affect | `node_affect` | `set_forebrain_target()` lets LLM replies, dancing and touch pull valence/arousal; `wake()` and `request_sleep()`; fixed `mood.arousal` clipping |
| Brainstem: SPG | `node_express` | `override()` / `release()`: forebrain behaviours borrow lights, ears, tail or eyelids for a few ticks |
| Brainstem: spatial | `node_spatial`, `node_detect_*`, Jeffress engine | the sound-localisation input topic is configurable (`host/mics` in sim) |
| Brainstem: BG | `node_action`, `basal_ganglia.py` | four forebrain actions in a priority band of their own (below) |
| Brainstem: MPG / loop | `node_loop`, kc | speech counts as vocalising for reafference gating |
| **Forebrain: linguistic** | — | `node_hearing` (wake words, VAD, utterance capture, touch-noise), `node_dialogue` + `forebrain/dialogue.py` (conversation state machine), `node_voice` (clips, speech, music) |
| **Forebrain: scene awareness** | — | `node_scene` in its own `- scene` client (YOLO objects, mediapipe gestures) |
| **Forebrain: integration** | — | `node_forebrain` (events → intentions, top-down modulation), `node_cues` (attentional eyelid cues) |
| **Forebrain: behaviours** | — | `action_converse`, `action_dance` (+ `choreography/`), `action_tricks`, `action_identify` |
| **Off-board (P4)** | — | `cloud/llm.py` (Whisper and GPT through any OpenAI-compatible route), `cloud/tts.py` (ElevenLabs), `cloud/music.py` (song library, optional Spotify), `cloud/worker.py` |

`core/heymiro_config.py` holds the settings, secrets and API routes.

### Processes

| Client | Command | Does |
|---|---|---|
| main | `client_demo.py -` | affect, express, action selection, forebrain nodes, kinematics; 50 Hz, driven by `sensors/package` |
| caml / camr | `client_demo.py - caml` / `- camr` | stock face, motion and AprilTag detection per camera |
| mics | `client_demo.py - mics` | stock sound localisation (from `hearing.salience_topic`) |
| scene | `client_demo.py - scene` | YOLO + mediapipe; subscribes to the camera only while main asks for it on `core/scene/mode` |
| bridge (optional) | `tools/host_audio_bridge.py` | host microphone → `host/mics` (sim, or `--ext-mic` on the robot) |

### Threads in the main client

The 50 Hz tick runs in the `sensors/package` callback. It must stay fast: no network calls, no model inference,
no waiting.

| Thread | Work |
|---|---|
| 50 Hz tick | All nodes' `tick()` and the actions. It only polls futures and queues. |
| Mic callback | Puts frames on NodeHearing's queue. |
| NodeHearing worker | Resampling 20 → 16 kHz, Porcupine, Cobra, the utterance segmenter, the touch-noise detector. |
| CloudWorker (2 threads) | STT → LLM → parse → TTS for one turn; song lookup and mp3 decoding. |
| Audio output stream (sounddevice) | Mixes the speech and music channels. |

Results carry an *episode id*, so an answer that arrives after the conversation has moved on is discarded.

## The conversation (HRI paper §II-B-1)

```
"Hey MiRo" (Porcupine)
   │  node_forebrain → affect.wake(), dialogue.on_wake()
   ▼
FACING     turn until an AprilTag is in view (or to the last sound, or skip)
GREETING   greeting clip ("What's up?") + lean in, look up, ears perk
LISTENING  segmenter armed: speech onset (Cobra VAD) → nod; ends after ~3 s of silence
THINKING   "Hmm" clip + loading tune; worker: wav → Whisper → GPT-4o (persona + current
           valence/arousal + time, JSON reply) → ElevenLabs
RESPONDING speak; emotion pose + LED colour; flourishes (spin, ear flapping, impression voice);
           affect pulled to the reply's valence/arousal
   └──► LISTENING …  no speech for ~3 s → GOODBYE clip → IDLE (MiRo resumes exploring)
```

The LLM is asked for a JSON object:

```json
{"valence": 0.8, "arousal": 0.7, "emotion": "happy", "reply": "...",
 "commands": {"spin": false, "sleep": false, "move_ears": false, "impression": false,
              "dance": false, "song": null, "tricks": false, "identify_object": false}}
```

A lenient parser also accepts the older HRI "Key: value" line format and plain text, so a model that ignores the
format still works.

- **Behaviour commands.** `dance`, `tricks` and `identify_object` hand over to a separate behaviour. The dialogue
  is suspended and then resumes ("that was fun — want to keep talking?").
- **`sleep`.** The affect node's sleep pressure is raised.
- **Stop.** "Stop MiRo" ends whatever is happening.

## Action selection

`basal_ganglia.py` is unchanged: winner-take-all, +0.05 hysteresis for the running action, a little noise, and
priorities clipped to [0, 1]. The forebrain actions use their own band:

| Behaviour | Priority |
|---|---|
| mull (idle) | 0.1 |
| orient / approach / flee | up to 0.8 (`priority_attend`) |
| **converse / dance / tricks / identify** | **0.84 × surface confidence** (0.89 while running) |
| halt (stall), avert (cliff), mull while touched | 0.9 – 1.0 |

What this ordering gives:

- A forebrain request pre-empts orienting and approaching.
- A running forebrain behaviour is never interrupted by them.
- Cliff aversion and halting still win at any time.
- Nothing reaches the 1.0 clip, where actions would tie.

The HRI version used 2 / 1.5 / 1.2. Those were all clipped to 1.0 and flapped against each other.

Only one forebrain behaviour is wanted at a time. `node_forebrain` holds a single task slot, and the dialogue
suspends while a task (dance, tricks, identify) runs.

## Dancing (HRI paper §II-B-3)

1. "Dance MiRo", or a dance request in conversation ("dance to Dancing Queen"), resolves a song from
   `config/songs.yaml` (fuzzy match on title, artist and aliases; a random song otherwise).
2. The song is decoded on the worker thread.
3. `action_dance` plays an intro clip and the music, and steps a `choreography.DanceController` with the
   **music playback position**. Head moves, wheel spins, ears and lights follow the beat (period 60/bpm).
4. The dance stops when the song ends, after `dance.max_duration_s`, on "Stop MiRo", or when a protective
   behaviour pre-empts it.
5. While it runs, valence rises and arousal follows the tempo.

Spotify's audio-analysis and preview endpoints are no longer available to new apps. The tempo of each bundled
song is therefore measured offline with `tools/estimate_bpm.py` and stored in `songs.yaml`. A Spotify lookup can
be switched on with `features.spotify`.

## Scene awareness (HRI paper §II-B-2)

The `- scene` client runs only while main requests it on `core/scene/mode`:

- **Gestures** (mediapipe), during *tricks* (or always, if `features.spontaneous_gestures`):
  - Open_Palm → greeting and nod
  - Pointing_Up → spin
  - Victory → ear wiggle
  - ILoveYou → party lights
- **Objects** (YOLO, 13 COCO classes), during the *identify* game: an object confirmed in 4 of 10 frames gets an
  LLM one-liner.

Touch on MiRo's body and head drives affect through the stock pathways. Scratching the ears or tail is detected
from the noise it makes in MiRo's own microphones; it plays an "Ouch!" clip and pushes valence down. Stroking
plays pleased noises.

## Latency

Typical turns take 0.5–5 s, depending on the network (HRI paper §III). The "Hmm" clip and loading tune fill the
gap. To move the LLM on-premises, point `HEYMIRO_LLM_BASE_URL` at a local OpenAI-compatible server; this was the
paper's first future-work item.
