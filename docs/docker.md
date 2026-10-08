# The Docker environment

The image contains only the environment:

- Ubuntu 20.04 with ROS Noetic (Python 3.8) and Gazebo 11 (`osrf/ros:noetic-desktop-full`)
- the **MiRo MDK 2-230105**, installed at `/root/mdk` with its catkin workspace built
- the Python packages in [`docker/requirements.txt`](../docker/requirements.txt): OpenAI SDK, Porcupine/Cobra,
  sounddevice, ultralytics (YOLO), mediapipe, OpenCV, apriltag, and torch (CUDA 12.1 build, which also runs on the
  CPU)
- ffmpeg, PortAudio, and an ALSA→PulseAudio bridge (`docker/asound.conf`)

It does **not** contain the project code, assets, or any keys. The projects are bind-mounted by
`docker/run_docker.bash`:

| Host | Container |
|---|---|
| `src/cloud_demo_robot` | `/root/mdk/catkin_ws/src/cloud_demo_robot` |
| `src/cloud_demo_sim` | `/root/mdk/catkin_ws/src/cloud_demo_sim` |
| `~/.config/heymiro/secrets.env` | `/run/secrets/heymiro.env` (read-only) |
| Docker volume `heymiro-cache` | `/root/.cache` (YOLO weights, ultralytics settings) |

A `CATKIN_IGNORE` file in each project keeps `catkin build` from treating it as a package.

## Images

CI ([`.github/workflows/docker-image.yml`](../.github/workflows/docker-image.yml)) builds the image whenever
`docker/` changes and pushes it to the GitHub Container Registry:

| Tag | When |
|---|---|
| `ghcr.io/heymiro/ext_cog_arch:latest` | default branch |
| `ghcr.io/heymiro/ext_cog_arch:<branch>` | every branch |
| `ghcr.io/heymiro/ext_cog_arch:sha-<commit>` | every build |
| `ghcr.io/heymiro/ext_cog_arch:<x.y.z>` | git tags `vX.Y.Z` |

The Dockerfile has a `test` stage that runs [`docker/smoke_test.sh`](../docker/smoke_test.sh). It checks:

- Python 3.8, and that every module imports
- pvporcupine 3.0.x, which matches the `*_v3_0_0.ppn` wake words
- the MDK is installed and roscore starts
- **no** project code, `.ppn`, key files or key variables are inside the image

The image is pushed only if the test stage passes.

GHCR packages of an organisation are **private** by default. To pull, log in with a GitHub personal access
token (classic) that has the `read:packages` scope:

```bash
echo <token> | docker login ghcr.io -u <github-user> --password-stdin
docker pull ghcr.io/heymiro/ext_cog_arch:latest
```

To make the package public (only if the MDK licence allows it), go to GitHub → HeyMiro → Packages →
ext_cog_arch → Package settings → Change visibility.

## Building locally

```bash
docker/build_image.bash                              # MDK downloaded from Google Drive
docker/build_image.bash --mdk ~/Downloads/mdk_2-230105.tgz
docker/build_image.bash --cpu --tag heymiro-env:cpu  # CPU-only torch build
docker/build_image.bash --test                       # build and run the smoke test
./docker/run_docker.bash --image heymiro-env:local --sim
```

The build needs BuildKit (the default in current Docker) and about 25 GB of free disk space. The MDK tarball is
taken from the first of these that exists:

1. `docker/mdk/mdk_2-230105.tgz` (gitignored)
2. `--build-arg MDK_URL=...`
3. the Google Drive copy used by the University of Sheffield COM3528 install script

The tarball is bind-mounted during the build, so it never becomes an image layer. Set the repository variable
`MDK_SHA256` to pin its checksum. The first CI build prints it.

## Changing dependencies

Edit [`docker/requirements.txt`](../docker/requirements.txt) (Python) or the Dockerfile (apt packages) and open
a pull request. CI builds the image and runs the smoke test. Once it is merged, `latest` is updated. This
replaces `docker commit` / `save_docker.bash`: the container is started with `--rm` and holds no state. The
exact installed versions are written to `/opt/heymiro/pip-freeze.txt` in every image.

## The ROS network

[`docker/heymiro_env.bash`](../docker/heymiro_env.bash) runs in every shell inside the container. It sources ROS
and the MDK, then sets up the network from the variables passed by `run_docker.bash`:

| Mode | `ROS_MASTER_URI` | `ROS_IP` |
|---|---|---|
| `MIRO_MODE=sim` | `http://localhost:11311` | `127.0.0.1` |
| `MIRO_MODE=robot` | `http://$MIRO_ROBOT_IP:11311` | your address on the route to the robot (override with `--host-ip`) |

It then prints a status banner showing the mode, ROS settings, GPU, audio, and which secrets are **set**. It
never prints their values.
