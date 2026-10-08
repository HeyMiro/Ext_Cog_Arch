# Configuration reference

Settings live in each project's `config/` folder. Secrets and API routes do not; they are covered in
[Secrets and API routes](#secrets-and-api-routes) below.

| File | What |
|---|---|
| `config/heymiro.yaml` | All settings, with comments. **This is the only file that differs between the robot and sim copies.** |
| `config/heymiro.local.yaml` | Your personal overrides. Gitignored; it only needs the keys you change. |
| `config/persona.md` | MiRo's personality for the LLM. `$event_name` is replaced by `llm.event_name`. |
| `config/clips.yaml` | Voice clip categories and the phrase folders under `assets/clips/` that belong to each. |
| `config/songs.yaml` | Songs MiRo can dance to: file, title, artist, aliases, bpm, genre. |

## Override order

Later sources win:

1. `config/heymiro.yaml`
2. `config/heymiro.local.yaml`
3. the YAML file named by `$HEYMIRO_CONFIG`
4. `$HEYMIRO_SET`, a list of `key=value` pairs separated by `;`, e.g.
   `HEYMIRO_SET="voice.volume=0.5;hearing.vad=energy"`. `run_robot.sh --ext-mic` uses this to set
   `hearing.input_topic=host/mics`.

Relative paths are resolved from the project folder.

## `heymiro.yaml` keys

### platform, mdk

| Key | Default (robot / sim) | Meaning |
|---|---|---|
| `platform` | `robot` / `sim` | Compared with the simulator flag in `sensors/package`; a mismatch prints a warning. |
| `mdk.demo_flags` | `vx` / `vxS` | Stock MDK demo flags forced on top of the robot's `platform_parameters`: `v` = MiRo's own vocaliser off (it would hum over speech), `x` = special AprilTag responses off, `S` = don't orient to sound. The full list is printed by `pars.py` at start-up. |

### features

Each switch is also turned off automatically when the key it needs is missing.

| Key | Default | Needs |
|---|---|---|
| `conversation` | true | `OPENAI_API_KEY` (or a custom LLM/STT route) |
| `wake_word` | true | `PICOVOICE_ACCESS_KEY` |
| `dance` | true | — |
| `tricks` | true | the scene client (mediapipe) |
| `identify` | true | the scene client (YOLO); uses the LLM for the comment if available |
| `touch_reactions` | true | MiRo's own microphones (`hearing.input_topic: sensors/mics` and `hearing.touch_from_mics`) |
| `purr` | true | — |
| `sleep_command` | true | — |
| `spontaneous_gestures` | false | React to hand signs without being asked. |
| `spotify` | false | `SPOTIFY_CLIENT_ID` / `SPOTIFY_CLIENT_SECRET` |
| `log_transcripts` | false | Print what people said (off for privacy). |

### hearing

| Key | Default | Meaning |
|---|---|---|
| `input_topic` | `sensors/mics` (sim: `host/mics`) | Topic (under `/<robot>/`) the speech pipeline listens to. |
| `salience_topic` | same | Topic the stock `- mics` sound-localisation client listens to. |
| `miro_channels` | `[0, 1]` | Mic channels averaged for speech (0 = left ear, 1 = right ear). |
| `gain` | 1.0 | Speech input gain. |
| `wake_words` | hey_miro, dance_miro, stop_miro | Each entry has `id`, `file` (`.ppn`, Porcupine v3), `sensitivity`, and `access_key_env` (a per-keyword key variable that falls back to `PICOVOICE_ACCESS_KEY`). |
| `vad` | `cobra` | `cobra` (Picovoice) or `energy` (no key needed). |
| `vad_threshold` | 0.6 | Voice probability counted as speech. |
| `energy_threshold_dbfs` | -38 | Threshold for the energy VAD. |
| `onset_frames` | 3 | Voiced 32 ms frames in a row needed to start an utterance. |
| `preroll_s` | 0.5 | Audio kept from before the onset. |
| `end_silence_s` | 3.0 | Silence that ends an utterance (HRI paper: about 3 s). |
| `no_speech_timeout_s` | 3.0 | No speech for this long while listening → goodbye. |
| `max_utterance_s` | 20 | Cap on one utterance. |
| `self_hearing_guard_s` | 0.3 (sim 0.6) | Keep ignoring the mic this long after MiRo stops speaking. |
| `touch_from_mics` | true (sim false) | Detect ears or tail being scratched from mic noise. |
| `touch_noise.*` | | Detector thresholds: `zcr_min`, `peak_ear`, `peak_tail`; `hits_to_react` within `window_s`; `cooldown_s`. |

### voice, tts, stt, llm

| Key | Default | Meaning |
|---|---|---|
| `voice.backend` | `host` | `host` plays through this computer's speaker (laptop or Bluetooth speaker); `miro` uses MiRo's speaker (8 kHz, paced by `sensors/stream`). |
| `voice.device` | null | sounddevice output device (index or name substring). |
| `voice.volume`, `voice.music_volume` | 1.0, 0.6 | |
| `voice.thinking_music` | true | Play "Hmm" plus a short tune while the cloud is busy. |
| `tts.provider` | `elevenlabs` | `elevenlabs` or `none`. |
| `tts.model_id`, `voice_id`, `impression_voice_id` | | ElevenLabs model and voices; the impression voice is used for the Mickey Mouse impression. |
| `tts.stability`, `similarity_boost`, `timeout_s` | 0.5, 0.75, 20 | |
| `stt.model`, `language`, `timeout_s` | `whisper-1`, `en`, 20 | |
| `llm.model` | `gpt-4o` | Any chat model available on the LLM route. |
| `llm.temperature`, `max_tokens` | 0.8, 400 | |
| `llm.json_mode` | true | Ask for a JSON reply. Servers that don't support it are retried without it, and a lenient parser is used. |
| `llm.timeout_s`, `max_history_turns` | 25, 10 | |
| `llm.persona_file`, `event_name` | `config/persona.md`, "the Festival of Mind" | |
| `llm.object_comment_model` | `gpt-4o-mini` | Model used for the object game's one-liners. |

### converse, dance, tricks, identify, scene

| Key | Default | Meaning |
|---|---|---|
| `converse.face_user` | `apriltag` (sim `none`) | How MiRo turns to face the speaker before greeting them: turn until an AprilTag is seen, turn to the loudest sound, or don't turn. |
| `converse.face_rotate_speed`, `face_timeout_s` | 0.08, 8 | |
| `converse.spin_speed`, `spin_s` | 0.4, 3 | The "spin around" command. |
| `converse.preempt_abort_s` | 10 | End the conversation if a higher-priority behaviour (e.g. cliff avoidance) holds the body for this long. |
| `dance.songs_manifest` | `config/songs.yaml` | |
| `dance.max_duration_s` | 60 | Dances stop after this long. |
| `dance.default_bpm` | 120 | Used when a song's bpm is unknown. |
| `dance.wheels`, `allow_translation` | true, true | |
| `tricks.duration_s`, `max_tricks`, `min_score` | 20, 5, 0.6 | |
| `tricks.map` | Open_Palm: greet, Pointing_Up: spin, Victory: ears, ILoveYou: party | Gesture → trick. |
| `identify.duration_s` | 15 | |
| `identify.confirm` | `[4, 10]` | An object counts once it is seen in 4 of the last 10 detections. |
| `identify.classes` | 13 COCO classes | |
| `scene.rate_hz`, `camera` | 5 (sim 1), `caml` | |
| `scene.yolo_weights` | `yolov10m.pt` | Downloaded on first use into `$HEYMIRO_CACHE_DIR/weights`. |
| `scene.yolo_conf`, `device` | 0.4, `auto` | |
| `scene.gesture_model` | `assets/models/gesture_recognizer.task` | |
| `scene.undistort`, `intrinsics` | false, null | Optional lens undistortion; always off in the simulator. |
| `clips.manifest` | `config/clips.yaml` | |

## Secrets and API routes

These come from the environment or the secrets file, never from YAML. See the README section
"Secrets and API routes" and [security.md](security.md).

| Variable | Default |
|---|---|
| `OPENAI_API_KEY` | — |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` |
| `HEYMIRO_LLM_BASE_URL` / `HEYMIRO_LLM_API_KEY` | `OPENAI_*` |
| `HEYMIRO_STT_BASE_URL` / `HEYMIRO_STT_API_KEY` | `OPENAI_*` |
| `ELEVENLABS_API_KEY` | — |
| `ELEVENLABS_BASE_URL` | `https://api.elevenlabs.io` |
| `PICOVOICE_ACCESS_KEY`, `PICOVOICE_ACCESS_KEY_<ID>` | — |
| `SPOTIFY_CLIENT_ID` / `SPOTIFY_CLIENT_SECRET` | — |
| `SPOTIFY_API_BASE_URL` | `https://api.spotify.com/v1` |
| `SPOTIFY_TOKEN_URL` | `https://accounts.spotify.com/api/token` |

Other environment variables:

| Variable | Meaning |
|---|---|
| `HEYMIRO_SECRETS_FILE` | Path of the secrets file; set by `run_docker.bash`. |
| `HEYMIRO_CACHE_DIR` | Where downloaded weights go (default `~/.cache/heymiro`). |
| `HEYMIRO_CONFIG` | Extra YAML overlay. |
| `HEYMIRO_SET` | Inline overrides. |
| `MIRO_ROBOT_NAME` | Robot name in topic names (default `miro`). |

## Topics added by the forebrain

All topics are under `/<robot>/`.

| Topic | Type | Direction |
|---|---|---|
| `host/mics` | `std_msgs/Int16MultiArray` | host audio bridge → hearing and `- mics` (MiRo mic format) |
| `core/forebrain/command` | `std_msgs/String` | you → main: `converse`, `dance [song]`, `tricks`, `identify`, `stop`, `say <text>` |
| `core/forebrain/state` | `std_msgs/String` (JSON) | main → monitoring |
| `core/scene/mode` | `std_msgs/String` | main → scene client: `off`, `gestures`, `objects`, `both` |
| `core/scene/objects`, `core/scene/gesture` | `std_msgs/String` (JSON) | scene client → main |
