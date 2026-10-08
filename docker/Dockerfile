# syntax=docker/dockerfile:1.7
#
# Hey MiRo - runtime ENVIRONMENT for the cloud demo (no project code inside).
#
# ROS Noetic (Ubuntu 20.04, Python 3.8) + Gazebo 11 + MiRo MDK 2-230105 + the
# Python packages used by src/cloud_demo_robot and src/cloud_demo_sim. The two
# project folders are bind-mounted at run time by docker/run_docker.bash, so
# code changes come from git, never from `docker commit`.
#
# Build (from the repository root; .dockerignore only lets docker/ in):
#   docker/build_image.bash                    # or:
#   DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile -t heymiro-env:local .
#
# MDK source, first match wins:
#   1. docker/mdk/mdk_2-230105.tgz            (put it there yourself; gitignored)
#   2. --build-arg MDK_URL=https://...        (any direct download link)
#   3. Google Drive copy used by the University of Sheffield install script
#
# Stages: "env" is the image you run; "test" runs docker/smoke_test.sh on top
# of it (CI builds --target test before pushing "env").

ARG BASE_IMAGE=osrf/ros:noetic-desktop-full

FROM ${BASE_IMAGE} AS env

ARG TORCH_VARIANT=cu121
ARG MDK_VERSION=230105
ARG MDK_GDRIVE_ID=1vNODaenljocVWalM4cOW4Kax-RB4U3nh
ARG MDK_URL=""
ARG MDK_SHA256=""

ENV DEBIAN_FRONTEND=noninteractive \
	USER=root \
	HOME=/root \
	LANG=C.UTF-8 \
	LC_ALL=C.UTF-8 \
	PYTHONDONTWRITEBYTECODE=1 \
	PIP_NO_CACHE_DIR=1 \
	PIP_DISABLE_PIP_VERSION_CHECK=1 \
	NVIDIA_VISIBLE_DEVICES=all \
	NVIDIA_DRIVER_CAPABILITIES=all \
	YOLO_CONFIG_DIR=/root/.cache/ultralytics \
	MPLCONFIGDIR=/root/.cache/matplotlib \
	HEYMIRO_CACHE_DIR=/root/.cache/heymiro

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# --- 1. ROS apt key ------------------------------------------------------------
# keys baked into older osrf/ros images expired on 2025-06-01; refresh it so
# apt-get update works (the archive still serves noetic/focal)
RUN mv /etc/apt/sources.list.d/ros1-latest.list /tmp/ 2>/dev/null || true; \
	rm -f /etc/apt/sources.list.d/ros*.list; \
	apt-get update && apt-get install -y --no-install-recommends curl gnupg ca-certificates && \
	curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
		| gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg && \
	echo "deb [signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros/ubuntu focal main" \
		> /etc/apt/sources.list.d/ros1.list && \
	rm -rf /var/lib/apt/lists/*

# --- 2. system packages ------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
		python3-pip python3-dev python3-yaml python3-catkin-tools \
		build-essential cmake git wget sudo lsb-release iproute2 iputils-ping net-tools nano less procps \
		ffmpeg libportaudio2 portaudio19-dev libasound2-dev libasound2-plugins alsa-utils \
		pulseaudio-utils libsndfile1 \
		libgl1 libglib2.0-0 mesa-utils x11-xserver-utils xauth \
	&& rm -rf /var/lib/apt/lists/*

# ALSA default -> PulseAudio (the host's sound server socket is mounted at run time)
COPY docker/asound.conf /etc/asound.conf

# --- 3. MiRo MDK -----------------------------------------------------------------------
# the tarball is bind-mounted or downloaded into a temp dir, so it never
# becomes an image layer
RUN --mount=type=bind,source=docker/mdk,target=/tmp/mdk-local \
	set -e; \
	tgz=/tmp/mdk.tgz; \
	if [ -f "/tmp/mdk-local/mdk_2-${MDK_VERSION}.tgz" ]; then \
		echo "MDK: using docker/mdk/mdk_2-${MDK_VERSION}.tgz"; cp "/tmp/mdk-local/mdk_2-${MDK_VERSION}.tgz" "$tgz"; \
	elif [ -n "${MDK_URL}" ]; then \
		echo "MDK: downloading from MDK_URL"; curl -fL "${MDK_URL}" -o "$tgz"; \
	else \
		echo "MDK: downloading from Google Drive"; \
		python3 -m pip install -q "gdown==5.2.0"; \
		python3 -m gdown "https://drive.google.com/uc?id=${MDK_GDRIVE_ID}" -O "$tgz"; \
		python3 -m pip uninstall -q -y gdown; \
	fi; \
	if [ -n "${MDK_SHA256}" ]; then echo "${MDK_SHA256}  $tgz" | sha256sum -c -; else sha256sum "$tgz"; fi; \
	mkdir -p /root/pkgs && tar -xzf "$tgz" -C /root/pkgs && rm -f "$tgz"; \
	test -x "/root/pkgs/mdk-${MDK_VERSION}/bin/deb64/install_mdk.sh"; \
	rm -f /root/mdk; \
	cd "/root/pkgs/mdk-${MDK_VERSION}/bin/deb64" && ./install_mdk.sh </dev/null; \
	sed -i '/# MDK/d; /mdk\/setup.bash/d' /root/.bashrc; \
	test -e /root/mdk/setup.bash

# build the MDK catkin workspace (miro2_msg); the setup script probes the
# network and prints warnings in a container, which we tolerate
RUN set -e; \
	source /opt/ros/noetic/setup.bash; \
	source /root/mdk/setup.bash || true; \
	cd /root/mdk/catkin_ws; \
	echo "catkin_ws/src:"; ls src; \
	[ -d .catkin_tools ] || catkin init; \
	catkin build --no-status; \
	if [ -d build/miro2_msg ]; then (cd build/miro2_msg && make install); fi; \
	source /root/mdk/catkin_ws/devel/setup.bash || true; \
	source /root/mdk/setup.bash || true; \
	python3 -c "import miro2, miro2_msg.msg; print('MDK python OK')" \
		|| { echo "PYTHONPATH=$PYTHONPATH"; ls /root/mdk/share/python /root/mdk/catkin_ws/devel/lib/python3/dist-packages || true; exit 1; }

# --- 4. Python packages ----------------------------------------------------------------
# Installed into /usr/local with --ignore-installed: apt's Python packages
# (psutil, numpy, Pillow, ... installed by distutils) cannot be uninstalled
# by pip, so ours shadow them instead. torch 2.4.1 from PyPI is the CUDA
# 12.1 build; another TORCH_VARIANT (e.g. cpu) is swapped in afterwards.
COPY docker/requirements.txt docker/constraints.txt /opt/heymiro/
RUN python3 -m pip install -U "pip<25.1" "setuptools<70" wheel && \
	python3 -m pip install --ignore-installed \
		-r /opt/heymiro/requirements.txt -c /opt/heymiro/constraints.txt \
		torch==2.4.1 torchvision==0.19.1 && \
	if [ "${TORCH_VARIANT}" != "cu121" ]; then \
		python3 -m pip install --force-reinstall --no-deps \
			--index-url "https://download.pytorch.org/whl/${TORCH_VARIANT}" torch==2.4.1 torchvision==0.19.1; \
	fi && \
	python3 -m pip install --force-reinstall --no-deps opencv-contrib-python==4.10.0.84 && \
	(python3 -m pip check || true) && \
	python3 -m pip freeze > /opt/heymiro/pip-freeze.txt

# --- 5. container tooling ------------------------------------------------------------
COPY docker/heymiro_env.bash docker/entrypoint.sh docker/smoke_test.sh /opt/heymiro/
RUN chmod +x /opt/heymiro/*.sh && \
	echo 'source /opt/heymiro/heymiro_env.bash' >> /root/.bashrc && \
	git config --system --add safe.directory '*' && \
	mkdir -p /root/mdk/catkin_ws/src /root/.cache/heymiro

LABEL org.opencontainers.image.source="https://github.com/HeyMiro/Ext_Cog_Arch" \
	org.opencontainers.image.description="Hey MiRo environment: ROS Noetic + MiRo MDK + cloud demo dependencies (code is bind-mounted)" \
	org.opencontainers.image.licenses="NOASSERTION"

WORKDIR /root/mdk/catkin_ws/src
ENTRYPOINT ["/opt/heymiro/entrypoint.sh"]
CMD ["bash"]


# --- test stage (CI): fail the build if the environment is broken --------------------
FROM env AS test
RUN MIRO_MODE=sim /opt/heymiro/smoke_test.sh
