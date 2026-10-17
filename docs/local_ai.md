# Local AI pipeline (`ai.provider: local`)

> Stage-by-stage internals and benchmarks (VAD, wake words, end of turn, speech-to-text, LLM, TTS, YOLO, scene
> context) are in [TECHNICAL.md](TECHNICAL.md).

Everything the HRI '25 demo did in the cloud now runs on this computer's NVIDIA GPU:

- wake words;
- voice activity detection;
- speech-to-text;
- the LLM;
- text-to-speech;
- deciding when the speaker has finished.

No API keys are needed, there is no per-use cost, and no internet is needed once the models are downloaded. This
was future-work item (i) of the paper: *"a local CPU/GPU cluster in place of cloud processing to improve response
latency … and enhance privacy"*.

`ai.provider: cloud` still runs the paper's setup (Picovoice, OpenAI, ElevenLabs) unchanged.

## Quick start

```bash
# once: download the models (~5 GB; add --9b for the larger LLM)
docker/local-ai/download_models.bash
# check that containers see the GPU (if not: RUN_SIMULATOR.md, 'If something goes wrong')
docker run --rm --gpus all ubuntu nvidia-smi

# every time: start the services, then the demo as usual
docker/run_docker.bash --sim --local-ai          # or --robot-ip <IP> --local-ai
cd ~/mdk/catkin_ws/src/cloud_demo_sim && ./run_sim.sh
```

At start-up the main client prints whether it can reach the local services:

```
[hearing] wake words: hey_miro, dance_miro, stop_miro (local ASR); VAD: silero; end of turn: smart-turn
[forebrain] local llm service ready
[forebrain] local speech service ready
```

To test without talking yourself, a synthetic user can speak to MiRo (see [Testing](#testing)):

```bash
./run_sim.sh --no-bridge            # terminal 1
python3 tools/fake_user.py          # terminal 2 (docker/run_docker.bash again)
```

## What runs where

```
 ROS container (Python 3.8, Noetic)                      docker/local-ai (Python 3.12, CUDA)
 ───────────────────────────────────                     ────────────────────────────────────
 mics ─► NodeHearing                                      speech  :8090
          Silero VAD (CPU, 0.7 ms/frame)                    POST /v1/audio/transcriptions  faster-whisper
          AsrKeywordSpotter ── short voiced segments ──►         (Whisper large-v3-turbo, int8)
          UtteranceSegmenter ── "finished?" ──────────►    POST /v1/turn   smart-turn v3 (CPU)
                │ utterance                                 POST /v1/audio/speech  Kokoro-82M
                ▼                                                (streams 24 kHz PCM per sentence)
          TurnRunner (CloudWorker thread)
            transcribe ───────────────────────────────►  speech
            chat_stream (json_schema) ────────────────►  llm  :8080  llama.cpp, Qwen3.5 GGUF
            ReplyStream: header ─► body reacts now            (vision, prompt cache, no thinking)
                         sentence ─► synthesize_stream ─► speech
                │ ("header", "speech" progress)
                ▼
          DialogueMachine ─► NodeVoice (speaks sentence 1 while 2 is synthesized)
```

The ROS side must stay on Python 3.8 (ROS Noetic). The models need modern Python, CUDA and PyTorch, so they run
in their own containers behind **OpenAI-compatible HTTP endpoints**. The robot code reaches them with the same
clients it uses for the cloud (`cloud/llm.py`, `cloud/tts.py`); only the base URL changes. That also means:

- **Another computer's GPU.** Set `HEYMIRO_LLM_BASE_URL` / `HEYMIRO_STT_BASE_URL`, or `ai.local.llm_url` /
  `ai.local.speech_url`.
- **Another model.** Edit the command in `compose.yaml`.

The services listen on 127.0.0.1 only.

## Models

All the models are ungated on Hugging Face (no login, no licence click-through) and pinned to a revision in
`docker/local-ai/models.env`.

| Job | Model | Licence | Size | Runs on |
|---|---|---|---|---|
| LLM (default) | `unsloth/Qwen3.5-4B-GGUF` Q4_K_M + mmproj (vision) | Apache-2.0 | 3.4 GB | GPU |
| LLM (quality profile, `--9b`) | `unsloth/Qwen3.5-9B-GGUF` Q4_K_M + mmproj | Apache-2.0 | 6.6 GB | GPU |
| Speech-to-text | `deepdml/faster-whisper-large-v3-turbo-ct2` (int8_float16) | MIT | 1.6 GB | GPU |
| Text-to-speech | `hexgrad/Kokoro-82M` | Apache-2.0 | 0.3 GB | GPU |
| End of turn | `pipecat-ai/smart-turn-v3` v3.2 (int8 ONNX) | BSD-2-Clause | 9 MB | CPU |
| Voice activity | Silero VAD v5 (pip `silero-vad`, model file only) | MIT | 2 MB | CPU (robot container) |

**Why 4B by default:** measured on the RTX 4070 Ti with `tools/eval_dialogue.py`, 9B picked the right action no more
often than 4B (38/40 each, F1 0.96 against 0.95). It was slower, though: 0.69 s against 0.44 s to the first sentence.
It also needs 9.5 GB instead of 7 GB with the speech models loaded, which is too much alongside Gazebo. It fits on the
robot (no Gazebo on the GPU): `--local-ai 9b`.

**MiRo's voice:** the paper chose a higher-pitched voice to suit MiRo's size. Kokoro's `af_heart` is synthesized
slower and resampled faster (`ai.local.pitch: 1.15`), which raises the pitch without changing the speed. The Mickey
Mouse impression uses `af_sky` at pitch 1.45.

## What is faster, and why

From the moment the user stops talking until MiRo starts to answer:

| Step | HRI '25 (cloud) | Local |
|---|---|---|
| Decide the user has finished | fixed ~3 s of silence | smart-turn: from 0.3 s of silence (at most `ai.local.end_silence_s` = 1.2 s) |
| Speech-to-text | upload + Whisper API | Whisper turbo on the GPU |
| LLM | the whole reply, then parse | streamed; the body reacts after ~25 tokens (emotion and actions come first) |
| Text-to-speech | the whole reply, then play | the first sentence plays while the rest is generated |

Measured numbers are in [Results](#results).

## Prompt and tool-call design

The LLM's job each turn is to say something *and* to decide what MiRo's body should do. Small local models get this
wrong more easily than GPT-4o, so the design takes the guesswork away:

1. **One registry of actions** (`core/cloud/tools.py`): spin, move_ears, impression, sleep, dance(song), tricks
   and identify_object. Each has a description, *when to use it* and *when not to*. The registry generates:
   - the reply's JSON Schema;
   - the "What you can do" section of the system prompt;
   - the mapping onto the command flags the rest of the code already understands.

   The prompt, the schema and the code cannot drift apart.
2. **Grammar-constrained output.** The schema is sent as `response_format: json_schema`, and llama.cpp compiles it
   into a grammar, so the model *cannot* produce anything else:
   - one of the six emotions;
   - valence and arousal in [0, 1];
   - at most two known actions;
   - for dance, a song from `config/songs.yaml` or `"any"`, so it cannot pick a song that does not exist;
   - a reply of at most 240 characters.
3. **Field order chosen for latency.** The order is `emotion, valence, arousal, actions, reply`, so the pose, the
   lights and the behaviour are decided before the first spoken word is generated.
4. **Few-shot examples, including negative ones.** Six fixed example exchanges show the format. They include
   "I love dancing too" and "Your ears are so cute!", which must **not** trigger an action. Small models over-trigger
   tools, and these examples fixed most of that (see Results).
5. **Prompt-cache-friendly layout.** The order is: system prompt → examples → history → `[context]` + user message.
   Everything up to the history is identical every turn, so llama.cpp reuses its cache and only processes the new
   message. The per-turn context (valence, arousal, time) goes *in front of the user's words* rather than in a second
   system message. Qwen's chat template rejects a second system message, and putting the context there would also
   break the cache.
6. **Spoken-language rules.** The reply must be one or two short sentences, with no emoji, markdown or stage
   directions. The model is also told that the words come from speech recognition and can be wrong, so it asks
   again instead of guessing.
7. **Sampling and server settings.** Qwen's non-thinking settings: temperature 0.7, top_p 0.8, top_k 20.
   "Thinking" is off on the server (`--reasoning-budget 0`), since a pet robot should answer immediately.

The same schema and prompt also work with GPT-4o, and the reply parser still accepts the HRI "Key: value" format and
free text, so the cloud provider benefits too.

## Settings (`config/heymiro.yaml`, `ai:`)

| Key | Default | Meaning |
|---|---|---|
| `ai.provider` | `local` | `local` or `cloud` |
| `ai.local.llm_url`, `speech_url` | `http://127.0.0.1:8080/v1`, `:8090/v1` | where the services are (`HEYMIRO_LLM_BASE_URL` / `HEYMIRO_STT_BASE_URL` override them) |
| `ai.local.stream` | true | speak the first sentence while the rest is generated |
| `ai.local.structured_output` | true | send the JSON Schema (grammar-constrained replies) |
| `ai.local.wake_word` | `asr` | `asr` (local Whisper) or `porcupine` (needs a Picovoice key) |
| `ai.local.vad` | `silero` | `silero`, `energy` or `cobra` |
| `ai.local.end_of_turn` | `smart_turn` | `smart_turn` or `silence` (fixed `hearing.end_silence_s`, the paper's behaviour) |
| `ai.local.turn_threshold`, `turn_min_silence_s`, `end_silence_s` | 0.9, 0.3, 1.2 | smart-turn decision threshold, first check, longest wait |
| `ai.local.voice`, `pitch`, `speed` | `af_heart`, 1.15, 1.05 | MiRo's Kokoro voice |
| `ai.local.impression_voice`, `impression_pitch` | `af_sky`, 1.45 | the Mickey Mouse impression |
| `ai.local.settings` | `llm.model: heymiro`, … | replaces the cloud `llm` / `stt` settings when local |

## Testing

- **Unit tests** (no GPU, no network): `python3 -m unittest discover -s tests`. `tests/test_local.py` covers the
  registry, the schema, the streaming parser, streamed turns, the wake phrases, smart-turn and the clients.
- **LLM evaluation:** `python3 tools/eval_dialogue.py` runs 40 scripted messages through the real prompt and schema.
  It reports JSON validity, action precision, recall and F1, reply length, and latency (time to the header, the
  first sentence and the whole reply).
- **End to end:** `python3 tools/fake_user.py` lets a Kokoro voice (`am_michael`) talk to MiRo over `host/mics`.
  The robot runs its real pipeline on it: VAD, wake word, smart-turn, Whisper, LLM and TTS. The script waits for
  MiRo's dialogue state before each line and prints a latency table.

## GPU memory

Measured on an RTX 4070 Ti (12 GB) with everything running: see [Results](#results). If memory runs short:

- use a lower `HEYMIRO_LLM_CTX` (default 16384, two slots);
- set `scene.device: cpu` for YOLO;
- use `HEYMIRO_STT_COMPUTE=int8`.

## Results

**Run 1: CPU only** (October 2026, 24-thread desktop CPU; Docker's GPU runtime not yet configured on the test
machine). Everything ran locally, but without the GPU.

`tools/eval_dialogue.py`, Qwen3.5-4B Q4_K_M, 40 cases:

| Metric | Result |
|---|---|
| valid JSON | 40/40 |
| keys in schema order | 40/40 |
| actions exactly right (incl. the song) | 39/40 |
| precision / recall / F1 | 0.96 / 1.00 / 0.98 |
| reply length | mean 11.7 words, max 18; no emoji or stage directions |
| latency (CPU) | header p50 2.7 s, first sentence p50 3.3 s |

The one miss was "I'm so tired, I want to sleep": MiRo decided to nap too. Before the negative few-shot example and
the "talking about it is not a request" rule, the score was 36/40 (F1 0.91), with three false triggers.

`tools/fake_user.py` in Gazebo (wake → question → spin → "dance to Dancing Queen" → "Stop MiRo"):

- Every step was understood and carried out, in three runs.
- End of the user's words → MiRo's answer took about 18 s on the CPU at first, and about 14 s after two fixes:
  - the busy wake-word check no longer transcribes whole sentences a second time;
  - the LLM prompt cache is warmed at start-up.

  The rest is Whisper (about 3.7 s) and LLM processing on a CPU that Gazebo also uses.
- Wake word: about 5 s on the CPU.
- The speech service alone (CPU): Kokoro synthesizes a sentence in 0.35–0.45 s, and smart-turn decides in about
  17 ms.

**Run 2: GPU (RTX 4070 Ti, 12 GB).** Same machine and models.

| Step | CPU | GPU |
|---|---|---|
| Whisper large-v3-turbo, 3 s utterance | 3.7 s | 0.10 s |
| Kokoro, one sentence | 0.4 s | 0.06 s |
| LLM: time to header (emotion + actions) | 2.7 s | 0.37 s (p50) |
| LLM: time to the first sentence | 3.3 s | 0.44 s (p50) |
| LLM prompt-cache warm-up at start | 8–12 s | 0.5 s |

`tools/eval_dialogue.py` on the GPU:

- **Format:** 40/40 valid JSON, all in schema order.
- **Actions:** 38/40 exact, F1 0.95.
- **The two misses:** "I'm so tired, I want to sleep" → MiRo naps too; "What is this object?" → no object game.
  Results vary by one or two cases between runs at temperature 0.7.

`tools/fake_user.py` in Gazebo, measured from the moment the user's voice stops (this includes smart-turn deciding
that they have finished):

| User says | MiRo reacts | after |
|---|---|---|
| "Hey MiRo!" | greets | 0.38 s |
| "Hi! What is your favourite band?" | starts answering | 0.91 s |
| "Can you spin around for me?" | starts answering (and spins) | 1.20 s |
| "Can you dance to Dancing Queen?" | starts dancing | 1.23 s |
| "Stop MiRo!" | stops and says goodbye | 0.46 s |

The HRI '25 demo waited a fixed 3 s of silence before it started transcribing, and the paper reports 0.5–5 s after
that.

**GPU memory** with Gazebo, the camera clients, YOLOv10m, Qwen3.5-4B + mmproj, Whisper and Kokoro all running:
7.9 GB of 12 GB (the desktop included).

**Docker and the GPU:** Docker 25+ exposes NVIDIA GPUs through CDI (`docker info` lists `nvidia.com/gpu=all`), and no
`nvidia` runtime entry is needed. `up.bash` and `run_docker.bash` accept either. Without a usable GPU, `up.bash`
starts the same services on the CPU (`compose.cpu.yaml`) with a warning.
