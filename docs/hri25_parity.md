# Parity with the HRI '25 demo

This compares three things for every behaviour:

- **the paper**: Htet et al., *Hey Miro!*, HRI '25, §II-B and §III;
- **the original demo code**: the `New_Miro_Demo` folder inside the Docker image `aung9htet/ros_miro:public`
  (2025-03-31, last commit `523170c`);
- **this repository**: `src/cloud_demo_robot` / `src/cloud_demo_sim`, with `ai.provider: cloud` (the HRI services)
  and `ai.provider: local` (docs/local_ai.md).

It was checked in October 2026 by:

- reading the original code;
- running the unit tests (291 pass in each copy);
- running the environment smoke test;
- running both providers end to end in the Gazebo simulator.

"Intended change" means the refactor does something on purpose that differs from the original. **Regression**
means something the original did that the refactor broke. One regression was found, and it is fixed.

## Regression found and fixed

| What | Effect | Fix |
|---|---|---|
| `client_demo.py` read `sensors/package.flags` as an int. It is a `std_msgs/UInt32` message. | `TypeError` on **every** 50 Hz tick, robot and sim: the whole brain (stock behaviours *and* the forebrain) never ran. The unit tests did not catch it because they do not build a real ROS message. | `msg.flags.data` (commit `6863112`). With the fix, orient/approach/flee/mull and every forebrain command were checked in Gazebo. |

## Behaviour by behaviour

| Behaviour (paper) | Original code | This repository | Status |
|---|---|---|---|
| **"Hey MiRo" wake word** (§II-B-1, Porcupine) | `node_detect_word`: Porcupine with `Hey-Miro_en_linux_v3_0_0.ppn`. Only "Hey MiRo" was active. | cloud: Porcupine with the same `.ppn` files, plus "Dance MiRo" and "Stop MiRo". local: the phrase is recognised by local Whisper (no key needed). | Same; extended |
| **Attention on wake**: lean in, head tilt, nod (§II-B-1) | `action_llm`: `look_up` → greeting clip → `lean_forward`; random nods while listening | `action_converse`: FACING (AprilTag) → GREETING (same clips) → lean in, look up, ears perk; a nod on each speech onset | Same |
| **Record until ~3 s of silence** (Cobra VAD) | `silence_count` of 32 ms blocks; Cobra re-run on the whole recording; ends at 80 blocks ≈ 2.6 s | cloud: Cobra per 32 ms frame, `end_silence_s: 3.0`. local: Silero VAD, then smart-turn ends the turn once the speaker has finished (0.3–1.2 s) | Same (cloud); faster (local) |
| **Whisper transcription** | `openai.Audio.transcribe("whisper-1")`, blocking inside the 50 Hz tick | cloud: whisper-1 on a worker thread. local: Whisper large-v3-turbo | Same; no longer blocks the robot |
| **GPT-4o with persona, emotion state, time** | One long prompt: valence/arousal plus time as extra system messages each turn. Fixed "Key: value" reply lines, parsed by line number. | cloud: GPT-4o with the same persona (`config/persona.md`), the same emotion table and the same flags, in JSON. The old line format is still parsed. local: Qwen3.5 with a JSON-schema-constrained reply | Same content; robust parser |
| **ElevenLabs voice** (higher pitched) | voice `vy5FHcwTzRxpcxreOntL`, impression `dfZGXKiIzjizWtJ0NgPy` | cloud: `tts.voice_id: 0m2tDjDewtOfXrhxqgrJ`, `impression_voice_id: dfZGXKiIzjizWtJ0NgPy`. local: Kokoro `af_heart` with pitch ×1.15 | **Check:** MiRo's voice id differs from the March 2025 image. If the HRI demo used `vy5F…`, set it in `config/heymiro.yaml`. |
| **Pre-recorded clips** (hums, cries, greetings, goodbyes) | `miro_clips_HD/*` | the same files in `assets/clips/`, listed in `config/clips.yaml` | Same |
| **"Hmm" + music while waiting** (§III) | `hmm_thinking`, then the `loading` tune | the same clips. local: only played when the reply is not ready yet | Same |
| **Repeat until no voice for 3 s**, then explore | `silence_count >= 80` with no voice → goodbye clip, history reset | `no_speech_timeout_s: 3.0` → GOODBYE clip → IDLE; memory cleared | Same |
| **Spin** | spins at 0.4 rad/s while speaking | spins at 0.4 rad/s for `converse.spin_s` (3 s) | Same speed; time-bounded |
| **Move ears** | 10 flaps at 2 Hz | 10 flaps at 2 Hz | Same |
| **Mickey Mouse impression** | prefixes "Hot Diggity Dog H-Hey everybody! It's me, Mickey Mouse! …" and switches voice | The prompt asks the model to write the reply as Mickey would say it, plus the impression voice. The fixed prefix is gone. | **Intended change.** Put the prefix back in `cloud/llm.py` if you want it word for word. |
| **Emotion → lights and pose** | happy: green + light_up; sad: blue + look down; angry: red + look down | the same colours and poses; worried and scared added (orange, purple) | Same; extended |
| **Emotion → affect** | `drive_by_response()` raised valence for happy **and** for sad and angry (same gain) | the reply's valence/arousal becomes the affect target (`set_forebrain_target`) | **Intended fix** (sad/angry no longer cheer MiRo up) |
| **Go to sleep** | flag parsed, never used | `affect.request_sleep()` after the goodbye | **Intended fix** |
| **Dance to music on request** (§II-B-3, Spotify tempo) | Spotify search, preview and audio analysis (tempo, sections), else one of 8 bundled songs; moves change at Spotify section boundaries; `drive_by_dance(genre)` | the same 8 songs with offline-measured bpm/genre (`config/songs.yaml`), because Spotify closed audio-analysis to new apps; the choreography follows the beat; genre drives affect | **Intended change** (forced by Spotify). Section-based move changes became beat-based phrases. |
| **Song requested by name** | Spotify lookup, else "Hmm, I don't know that one, how about this one instead" | fuzzy match on title, artist and aliases. local: the model can only pick a bundled song or "any" (schema enum) | Same idea; cannot pick a song that does not exist |
| **"Stop MiRo"** | `stop_miro` flag checked in dance (no wake word for it was active in this snapshot) | a wake word; stops dance, tricks, identify and conversation | Works now |
| **Tricks / hand gestures** (mediapipe) | `NodeVision` gestures. `ActionGestures` was **commented out** in `node_action.py` in this snapshot. | `action_tricks` + the `- scene` client; Open_Palm / Pointing_Up / Victory / ILoveYou | Works now (check the gesture → trick mapping is the one you showed) |
| **Identify object** (YOLO) | YOLOv10m, the same 13 COCO classes, confirm, GPT one-liner | YOLOv10m, the same 13 classes, confirm in 4 of 10 frames, LLM one-liner | Same |
| **Touch** (stroke → pleased noises, ear/tail scratch → "Ouch") | `node_affect`: mic-noise thresholds, pleased noises on petting | `forebrain/touch_noise.py` (same idea, thresholds rescaled for the averaged channels); purr on petting | Same |
| **Basal-ganglia selection of the new behaviours** | priorities 2 / 1.5 / 1.2, all clipped to 1.0 by `basal_ganglia.py`, so they flapped against each other | their own band, 0.84 (0.89 while running), below halt/avert (0.9–1.0) | **Intended fix** |
| **Cliff aversion** (`ActionAvert`) | commented out | restored | **Intended fix** |
| **Stock MiRo brain** (affect, SPG, spatial, BG, kc) | MDK 230105 demo, edited | MDK 230105: 21 of its 30 demo files byte-identical; small, documented edits in 8 (`client_demo`, `node_affect`, `node_express`, `node_action`, `pars`, `node_loop`, `node_spatial`, `action_special`); `node_detect_audio_perftest` (a benchmark) dropped | Same |

## Things to confirm

1. **MiRo's ElevenLabs voice id.** It was `vy5FHcwTzRxpcxreOntL` in the March 2025 image and is now `0m2tDjDewtOfXrhxqgrJ`.
2. **The Mickey Mouse prefix.** It was spoken word for word in the original. It is now in the style of the generated
   reply.
3. **Old keys.** The old image `aung9htet/ros_miro:public` still contains API keys in its source files
   (`action_llm.py`, `action_dance.py`, `node_detect_word.py`). As `docs/security.md` says, rotate them, and make the
   image private or delete it.
