# Getting started: run Hey MiRo step by step

This guide is for anyone, including people who have never used a terminal. Every command is in its own grey box.
For each one, the guide says what it does and what you should see afterwards.

**How to use this guide:**

- Copy one box at a time, paste it into the terminal, and press **Enter**.
- Wait until the command has finished before pasting the next one. It has finished when the line ending in `$`
  appears again.
- Lines starting with `#` are comments. Copying them is harmless.

**Contents**

1. [What you need](#1-what-you-need)
2. [Terminal basics (2 minutes)](#2-terminal-basics-2-minutes)
3. [One-time setup](#3-one-time-setup) (about 1 hour, mostly waiting for downloads)
4. [Run the demo in the simulator](#4-run-the-demo-in-the-simulator)
5. [Run the demo on the real MiRo robot](#5-run-the-demo-on-the-real-miro-robot)
6. [Stop everything](#6-stop-everything)
7. [Get the newest version](#7-get-the-newest-version)
8. [Check that everything works](#8-check-that-everything-works)
9. [Problems and fixes](#9-problems-and-fixes)
10. [Cheat sheet](#10-cheat-sheet)

---

## 1. What you need

| | |
|---|---|
| **Computer** | Ubuntu Linux 22.04 or 24.04 (tested on 24.04) |
| **Graphics card** | An NVIDIA graphics card with at least 8 GB of memory (tested on an RTX 4070 Ti, 12 GB; the demo uses about 8 GB). Without one, everything still works automatically on the processor, but MiRo answers in about 15 s instead of about 1 s. |
| **Disk space** | About 40 GB free |
| **Internet** | Needed for the one-time setup only. After that, the demo runs offline. |
| **Microphone and speaker** | For talking to MiRo. Headphones are best in the simulator, so MiRo doesn't hear itself. |
| **For the real robot** | A MiRo-e on the same Wi-Fi network as the computer, and the MiRo app on a phone |
| **Password** | Your computer password, for the commands that start with `sudo` |

No API keys or paid accounts are needed. All the AI (speech recognition, the language model and MiRo's voice) runs
on your own computer.

---

## 2. Terminal basics (2 minutes)

- **Open a terminal:** press **Ctrl + Alt + T**. A window with a line ending in `$` appears.
- **Paste into the terminal:** press **Ctrl + Shift + V** (plain Ctrl + V does not work in a terminal), or
  right-click and choose *Paste*.
- **Stop a running program:** press **Ctrl + C**.
- **Typing your password:** after a `sudo` command the terminal asks for your password. **Nothing appears while you
  type**, not even dots. This is normal. Type it and press Enter.
- **"Permission denied" or "command not found":** see [Problems and fixes](#9-problems-and-fixes).
- **Two kinds of terminal:**
  - The **computer terminal** shows a prompt like `name@computer:~$`.
  - The **MiRo container terminal** shows a prompt like `root@computer:/root/mdk/catkin_ws/src#`.

  The guide always says which one a command is for.

---

## 3. One-time setup

Do this once per computer. If a step says *already done* or *already installed*, just go on to the next one.

### 3.1 Install basic tools

```bash
sudo apt-get update && sudo apt-get install -y git curl
```

**What it does:** installs `git` (downloads the MiRo code) and `curl` (downloads other files).
**You should see:** lots of text, ending with your prompt again. *"is already the newest version"* is fine.

### 3.2 Install Docker

Docker runs MiRo's software in a ready-made "container", so you don't have to install ROS, the MiRo kit or the AI
libraries yourself.

```bash
curl -fsSL https://get.docker.com -o /tmp/get-docker.sh
```

**What it does:** downloads Docker's official installer.
**You should see:** nothing; the prompt comes back.

```bash
sudo sh /tmp/get-docker.sh
```

**What it does:** installs Docker and Docker Compose. It takes a few minutes.
**You should see:** a long list ending with information about Docker. If Docker was already installed, it says so.

### 3.3 Use Docker without typing `sudo` every time

```bash
sudo usermod -aG docker $USER
```

**What it does:** adds you to the `docker` group, so you are allowed to use Docker.
**You should see:** nothing.

**Now log out of Ubuntu and log back in** (or restart the computer). The change only applies after that. Then open
a new terminal (**Ctrl + Alt + T**) and check it worked:

```bash
docker run --rm hello-world
```

**What it does:** runs a tiny test container.
**You should see:** *"Hello from Docker!"* If you see *"permission denied"*, you have not logged out and in yet.

### 3.4 Check the NVIDIA graphics driver

```bash
nvidia-smi
```

**What it does:** shows your graphics card and its driver.
**You should see:** a table with your card's name (e.g. *NVIDIA GeForce RTX 4070 Ti*) and a *Driver Version*.

**If you see "command not found" instead**, install the driver, then restart:

```bash
sudo ubuntu-drivers install
```

```bash
sudo reboot
```

**What they do:** the first command installs the recommended NVIDIA driver; the second restarts the computer.
Afterwards, run `nvidia-smi` again to check.

*No NVIDIA card?* Skip to [3.7](#37-download-the-miro-code). Everything still works, just more slowly.

### 3.5 Check that Docker can use the graphics card

```bash
docker run --rm --gpus all ubuntu nvidia-smi
```

**What it does:** starts a tiny test container and asks it to show the graphics card.
**You should see:** the same table as in 3.4. **If you do, skip to [3.7](#37-download-the-miro-code).** Recent
Docker versions often work straight away.

**If you see an error instead** (for example *"could not select device driver"*), do 3.6.

### 3.6 Only if 3.5 failed: install the NVIDIA Container Toolkit

Run these five commands one by one.

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
```

**What it does:** adds NVIDIA's signing key, so Ubuntu trusts their software.
**You should see:** nothing (after your password).

```bash
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

**What it does:** tells Ubuntu where to download the toolkit from.
**You should see:** one or two lines starting with `deb [signed-by=...`.

```bash
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
```

**What it does:** installs the toolkit.
**You should see:** installation text ending with your prompt.

```bash
sudo nvidia-ctk runtime configure --runtime=docker
```

**What it does:** tells Docker that the toolkit exists.
**You should see:** a message mentioning `/etc/docker/daemon.json`.

```bash
sudo systemctl restart docker
```

**What it does:** restarts Docker so it picks up the change.
**You should see:** nothing. Now run the test from 3.5 again; it should show the table.

### 3.7 Download the MiRo code

```bash
cd ~
```

**What it does:** goes to your home folder (where the code will be stored).

```bash
git clone https://github.com/HeyMiro/Ext_Cog_Arch.git
```

**What it does:** downloads the code into a new folder `~/Ext_Cog_Arch`.
**You should see:** *"Cloning into 'Ext_Cog_Arch'..."*. If it says the folder *already exists*, you have done this
before; go on to the next command.

```bash
cd ~/Ext_Cog_Arch
```

**What it does:** goes into the code folder. **All the remaining commands in this guide are run from here.** If you
open a new terminal later, run this line first.

```bash
git switch claude/local-ai-pipeline
```

**What it does:** selects the version of the code with the local AI (no cloud, no API keys). Skip this line once that
version has been merged into `main`.
**You should see:** *"Switched to branch 'claude/local-ai-pipeline'"*.

### 3.8 Download the AI models

```bash
docker/local-ai/download_models.bash
```

**What it does:** downloads MiRo's AI models (about 5 GB) into Docker storage:
- the language model (Qwen3.5-4B);
- speech recognition (Whisper);
- MiRo's voice (Kokoro);
- end-of-turn detection (smart-turn).

All of them are free and need no account. **This takes 5–30 minutes**, depending on your internet.
**You should see:** lines starting with `==` for each model, then a list of folders with sizes (1.6G, 315M, 8.3M, 3.2G).

*Optional:* the larger, smarter language model (9B, another 6.6 GB). Only useful on the real robot, because in the
simulator the graphics card is also busy with the 3D view:

```bash
docker/local-ai/download_models.bash --9b
```

Setup is finished. 🎉 You never need to repeat these steps on this computer.

---

## 4. Run the demo in the simulator

The simulator shows a 3D MiRo on screen. You talk to it through your computer's microphone, and it answers through
your speakers.

### 4.1 Start MiRo's AI and the MiRo container

In a **computer terminal**:

```bash
cd ~/Ext_Cog_Arch
```

```bash
./docker/run_docker.bash --sim --local-ai
```

**What it does:**
1. Starts the two AI services: the language model, and the speech service (recognition plus MiRo's voice).
2. Waits until the models are loaded.
3. Opens the **MiRo container terminal**.

The first time, Docker also downloads the MiRo environment (about 11 GB) and builds the speech service (about 6 GB),
which takes 10–30 minutes. Later starts take under a minute.

**You should see:** *"waiting for the models to load.... ready"*, a status box (*mode: sim*, *GPU: NVIDIA ...*,
*audio: PulseAudio OK*), and the prompt changes to `root@...#`.

If you see *"WARNING: Docker cannot use an NVIDIA GPU, so the AI runs on the CPU"*, the demo still works, but
slowly. To use the graphics card, see [3.5](#35-check-that-docker-can-use-the-graphics-card).

### 4.2 Start the simulator and MiRo's brain

In the **MiRo container terminal** (the `root@...#` prompt):

```bash
cd $HEYMIRO_SIM
```

**What it does:** goes to the simulator version of the code.

```bash
./run_sim.sh
```

**What it does:**
- opens the 3D simulator window;
- connects your microphone to the simulated MiRo;
- starts MiRo's brain.

**You should see:** after 20–60 seconds, a 3D window with MiRo and a blue ball in a red-walled arena. The terminal
shows these lines:

```
[hearing] wake words: hey_miro, dance_miro, stop_miro (local ASR); VAD: silero; end of turn: smart-turn
[forebrain] local llm service ready
[forebrain] local speech service ready
[forebrain] LLM prompt cache warmed (...)
```

Leave this terminal open. MiRo runs as long as it does.

### 4.3 Talk to MiRo

Speak clearly towards your microphone (headphones help):

| Say | MiRo does |
|---|---|
| **"Hey MiRo!"** | wakes up, says hello, then listens |
| any question, e.g. "What's your favourite band?" | answers |
| "Can you spin around?" | spins |
| "Wiggle your ears!" | flaps its ears |
| "Do your Mickey Mouse impression!" | answers in a Mickey Mouse voice |
| "Can you dance to Dancing Queen?" | dances to the song |
| **"Dance MiRo!"** | dances straight away |
| "Do some tricks!" | watches your hand signs (needs a camera image; easier on the real robot) |
| "Go to sleep" | says goodbye and gets sleepy |
| **"Stop MiRo!"** | stops whatever it is doing |

If nobody speaks for about 3 seconds, MiRo says goodbye and goes back to exploring.

### 4.4 Optional: let a computer voice talk to MiRo

Use this for a hands-free demo, or to test without a microphone. Stop MiRo first: press **Ctrl + C** in the
MiRo container terminal. Then:

```bash
./run_sim.sh --no-bridge
```

**What it does:** starts the simulator and MiRo again, but without your microphone.

Now open a **second terminal** (**Ctrl + Alt + T**) and paste these lines one by one:

```bash
cd ~/Ext_Cog_Arch
```

```bash
./docker/run_docker.bash
```

**What it does:** opens a second MiRo container terminal, connected to the same running MiRo.
**You should see:** *"attaching a new terminal to running container 'heymiro'"*.

```bash
cd $HEYMIRO_SIM
```

```bash
python3 tools/fake_user.py --play
```

**What it does:** a synthetic voice says, one after the other: "Hey MiRo!", "What is your favourite band?",
"Can you spin around?", "Can you dance to Dancing Queen?" and "Stop MiRo!". Each line waits until MiRo is ready.
You hear both voices on your speakers.
**You should see:** each line printed as it is said, then a table of how many seconds MiRo took to answer.

### 4.5 Optional: give commands without speaking

In a second MiRo container terminal (as in 4.4: `./docker/run_docker.bash`), you can send commands directly:

```bash
rostopic pub -1 /miro/core/forebrain/command std_msgs/String "data: 'dance abba'"
```

**What it does:** makes MiRo dance to ABBA. Swap the word after `data:` for any of these:

| Command | MiRo does |
|---|---|
| `converse` | starts a conversation, like "Hey MiRo" |
| `dance` | dances to a random song |
| `tricks` | does tricks |
| `identify` | plays the object-guessing game |
| `say Hello there!` | says the words after `say` |
| `stop` | stops everything |

---

## 5. Run the demo on the real MiRo robot

### 5.1 Prepare the robot

1. Switch MiRo on and connect it to the **same Wi-Fi** as the computer.
2. Open the MiRo app on your phone and find MiRo's **IP address**. It looks like `192.168.0.42`.
3. In the app, **switch Demo mode OFF**. Otherwise MiRo's built-in demo fights this one for the motors.

### 5.2 Start everything

In a **computer terminal**. Replace `192.168.0.42` with your MiRo's IP address:

```bash
cd ~/Ext_Cog_Arch
```

```bash
./docker/run_docker.bash --robot-ip 192.168.0.42 --local-ai
```

**What it does:** starts the AI services, then opens the MiRo container terminal, set up to talk to your robot over
Wi-Fi.
**You should see:** the status box with *mode: robot* and *ROS_MASTER_URI: http://192.168.0.42:11311*.

On the real robot the graphics card has nothing else to do, so you can use the larger language model if you
downloaded it (3.8): write `--local-ai 9b` instead of `--local-ai`.

In the **MiRo container terminal**:

```bash
cd $HEYMIRO_ROBOT
```

Then **one** of these two:

```bash
./run_robot.sh
```

**What it does:** starts MiRo's brain, using **MiRo's own microphones**. Best in a quiet room.

```bash
./run_robot.sh --ext-mic
```

**What it does:** the same, but uses **the computer's microphone** (or a USB microphone plugged into it). Best in
noisy rooms, such as at an event.

**You should see:** MiRo starts moving and looking around. Talk to it as in [4.3](#43-talk-to-miro). Its voice comes
out of the **computer's speaker**. Put a Bluetooth speaker next to MiRo for the best effect, as at the HRI '25
demo.

---

## 6. Stop everything

1. In the terminal where MiRo is running, press **Ctrl + C**. This stops MiRo and closes the simulator.
2. Leave the MiRo container. In the MiRo container terminal:

   ```bash
   exit
   ```

   You are back at the computer terminal.
3. Stop the AI services (this frees the graphics card memory):

   ```bash
   docker/local-ai/up.bash --down
   ```

   **You should see:** lines ending in *Stopped* and *Removed* for `heymiro-local-ai-llm-1` and
   `heymiro-local-ai-speech-1`.

If something seems stuck, this stops every MiRo container at once:

```bash
docker stop heymiro; docker/local-ai/up.bash --down
```

---

## 7. Get the newest version

In a **computer terminal**, with MiRo stopped:

```bash
cd ~/Ext_Cog_Arch
```

```bash
git pull
```

**What it does:** downloads the latest code changes.
**You should see:** a list of changed files, or *"Already up to date."*

```bash
./docker/run_docker.bash --pull
```

**What it does:** downloads the newest MiRo environment that matches this code (only the parts that changed).
**You should see:** download progress, then *"Status: Image is up to date"* or *"Downloaded newer image"*.

```bash
docker/local-ai/download_models.bash
```

**What it does:** downloads any AI models that were added. It skips the ones you already have, so this is quick.

The speech service is rebuilt automatically the next time you start with `--local-ai`, if it changed.

---

## 8. Check that everything works

These commands give a quick health check. They are optional.

**The AI services are running** (computer terminal, while they are started):

```bash
curl -s http://127.0.0.1:8090/health; echo
```

**You should see:** `{"ok":true,"device":"cuda",...}`. `"device":"cpu"` means the graphics card is not being used;
see section 9.

```bash
curl -s http://127.0.0.1:8080/health; echo
```

**You should see:** `{"status":"ok"}`.

**The software's self-tests pass** (MiRo container terminal):

```bash
cd $HEYMIRO_SIM && python3 -m unittest discover -s tests 2>&1 | grep -E "^(Ran|OK|FAILED)"
```

**What it does:** runs the software's built-in checks (about 10 seconds), and shows only the result.
**You should see:** two lines, *"Ran 292 tests in ..."* and **OK**. The number grows as tests are added. *FAILED*
means something is broken: send the team the output of the same command without everything after `2>&1`.

**The language model answers well and quickly** (MiRo container terminal, AI services running):

```bash
cd $HEYMIRO_SIM && python3 tools/eval_dialogue.py
```

**What it does:** asks MiRo's brain 40 test questions and checks every answer. Is the reply valid? Did it choose the
right action, such as dancing only when asked? How fast was it?
**You should see:** one line per question (*ok* or *BAD*), then a summary. A good result is: valid JSON 40/40, F1
above 0.9.

---

## 9. Problems and fixes

| What you see | What to do |
|---|---|
| `permission denied ... docker.sock` | You are not in the docker group yet: do [3.3](#33-use-docker-without-typing-sudo-every-time), then log out and in. |
| `cannot attach stdin to a TTY` | You are running an old copy of the code: do [7](#7-get-the-newest-version). |
| `WARNING: Docker cannot use an NVIDIA GPU` | MiRo still works, slowly. Do [3.5](#35-check-that-docker-can-use-the-graphics-card) (and 3.6 if needed), then stop and start again. |
| `no models yet: run .../download_models.bash first` | Do [3.8](#38-download-the-ai-models). |
| `local llm service NOT reachable` | The AI services are not running. Start MiRo with `--local-ai`, as in 4.1. |
| `container 'heymiro' is already running` | MiRo is already open in another terminal. Use that one, or stop it: `docker stop heymiro`. |
| MiRo never reacts to "Hey MiRo" | Is the right microphone selected in Ubuntu's *Settings → Sound → Input*? Check the level: in the MiRo container terminal, run `python3 tools/mic_check.py --topic host/mics` (simulator) and speak. |
| MiRo answers itself, or keeps talking | It hears its own voice from the speakers. Use headphones, or turn the speaker volume down. |
| No sound from MiRo | Check Ubuntu's *Settings → Sound → Output*. In the MiRo container terminal, `pactl info` should show a *Server Name*. |
| The 3D window does not open | In a computer terminal, run `xhost +local:root`, then start again. |
| Everything works but answers take 10–20 s | The graphics card is not being used (`"device":"cpu"` in section 8). Do 3.5 (and 3.6), then [stop](#6-stop-everything) and start again. |
| `No space left on device` | Free disk space. `docker system prune` removes old, unused Docker data (it asks first). |
| Anything else | Copy the last 30 lines of the terminal and send them to the team. The full logs are in the MiRo container, in `~/.ros/heymiro/<date>/`. |

---

## 10. Cheat sheet

Once setup is done, this is all you need:

```bash
# simulator
cd ~/Ext_Cog_Arch
./docker/run_docker.bash --sim --local-ai
cd $HEYMIRO_SIM
./run_sim.sh
```

```bash
# real robot (put in MiRo's IP address)
cd ~/Ext_Cog_Arch
./docker/run_docker.bash --robot-ip 192.168.0.42 --local-ai
cd $HEYMIRO_ROBOT
./run_robot.sh            # or: ./run_robot.sh --ext-mic
```

```bash
# stop: Ctrl + C, then
exit
docker/local-ai/up.bash --down
```

```bash
# update
cd ~/Ext_Cog_Arch
git pull
./docker/run_docker.bash --pull
docker/local-ai/download_models.bash
```

More detail for developers: [README](../README.md), [local AI design](local_ai.md),
[configuration](configuration.md), [simulator](simulation.md), [comparison with the HRI '25 demo](hri25_parity.md).
