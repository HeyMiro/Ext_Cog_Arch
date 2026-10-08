# Security: API keys

## Rotate the old keys now

The previous version of the demo had API keys **hardcoded in the source**, and that source was baked into the
public Docker image `aung9htet/ros_miro:public`. Anyone who pulled that image can read the keys, so treat them
as compromised.

Revoke and re-create each of the following.

| Service | What to do |
|---|---|
| OpenAI | platform.openai.com → API keys: revoke the key that was in `core/action/api_key/api_key.txt`. Create a new **project-scoped** key with a monthly budget limit. |
| ElevenLabs | elevenlabs.io → Profile → API keys: delete every key that appeared in `action_llm.py`, `action_dance.py`, `action_gestures.py`, `action_obj_detect.py` and `make_clip.py` (several accounts were used). Create a new key. |
| Picovoice | console.picovoice.ai → AccessKey: regenerate the AccessKey of each account used for "Hey MiRo", "Dance MiRo" and "Stop MiRo" (four keys in total, one of them for Koala). Check that the `.ppn` files still load with the new key. |
| Spotify | developer.spotify.com → Dashboard → your app → Settings → **Rotate client secret**. |

After rotating, check each service's usage and billing pages for activity you don't recognise. Whether to make
the old Docker Hub image private or delete it is up to you. This repository does not modify it.

## How keys are handled now

- **One file, outside git.** Keys go in `~/.config/heymiro/secrets.env` with mode 600, created from
  `config/secrets.env.example`. `.gitignore` excludes `*.env`, `api_key*.txt` and `heymiro.local.yaml`.
- **Never in the image.** The Docker build context is restricted to `docker/` by `.dockerignore`. The smoke test
  fails if any key file or key variable ends up in the image.
- **Mounted read-only at run time.** `run_docker.bash` mounts the file at `/run/secrets/heymiro.env`. It is not
  passed with `-e` or `--env-file`, so the keys don't show up in `docker inspect`.
- **Read on demand.** `core/heymiro_config.py` reads the file when a cloud client is created. Keys are never
  stored on `pars`, put on the ROS parameter server (which anyone on the robot's network can read), published on
  a topic, or printed. The start-up banner only says `set` or `missing`.
- **Sanitised errors.** Failed HTTP calls report the exception class and status code only. Request headers are
  never logged.
- **CI secret scan.** gitleaks runs on every push with extra rules for Picovoice, ElevenLabs, Spotify and
  `api_key.txt` files (`.gitleaks.toml`).

## Recommended GitHub settings

- Enable **Secret scanning** and **Push protection** under Settings → Code security. Both are free for public
  repositories, and push protection blocks a push that contains a known key format.
- Keep the GHCR package `ghcr.io/heymiro/ext_cog_arch` **private** unless you have checked that the MiRo MDK
  licence allows redistribution. Users then pull with a token that has `read:packages`.

## Don'ts

- Don't paste keys into code, YAML, notebooks or issue comments.
- Don't run the entrypoint or launch scripts with `bash -x`, because it prints environment values.
- Don't commit `docker/mdk/*.tgz` (the MDK is licensed by Consequential Robotics).
