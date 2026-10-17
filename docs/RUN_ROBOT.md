# Run Hey MiRo on the real MiRo-e robot

MiRo's brain runs on this computer and drives the robot over Wi-Fi. All the AI runs on this computer; no API keys are
needed. For the simulator, see [RUN_SIMULATOR.md](RUN_SIMULATOR.md).

**You need:**

- Ubuntu 22.04/24.04 with an NVIDIA GPU of 8 GB or more (without one it still works, with ~15 s answers);
- 40 GB of free disk;
- a MiRo-e and the MiRo app on a phone;
- **the computer and MiRo on the same Wi-Fi**;
- a speaker for MiRo's voice (a Bluetooth speaker next to MiRo works best).

Paste commands with **Ctrl+Shift+V**. Run them from the `Ext_Cog_Arch` folder unless a step says otherwise.

---

## Setup (once per computer)

**1. Install Docker** and allow your user to run it:

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
```

The first line installs Docker; the second lets you use it without `sudo`. **Log out and back in** afterwards.

**2. Check that Docker can see the GPU:**

```bash
docker run --rm --gpus all ubuntu nvidia-smi
```

It should print a table with your graphics card.
- If it says `nvidia-smi: not found`, install the driver with `sudo ubuntu-drivers install` and reboot.
- If it says `could not select device driver`, see [GPU not found](#if-something-goes-wrong).

**3. Download the code:**

```bash
cd ~
git clone https://github.com/HeyMiro/Ext_Cog_Arch.git
cd Ext_Cog_Arch
git switch claude/local-ai-pipeline
```

This downloads the code into `~/Ext_Cog_Arch` and selects the local-AI version. Skip the last line once that
version is merged into `main`.

**4. Download the AI models** (about 5 GB; takes 5–30 minutes):

```bash
docker/local-ai/download_models.bash
```

This fetches the language model, speech recognition, MiRo's voice and end-of-turn detection. They are free and need
no account. Re-running it only fetches what is missing.

---

## Prepare the robot (every time)

1. Switch MiRo on and connect it to the same Wi-Fi as the computer.
2. In the MiRo app, find **MiRo's IP address**. It looks like `192.168.0.42`.
3. In the MiRo app, **turn Demo mode OFF**. Otherwise MiRo's own demo fights this one for the motors.
4. Connect the speaker to the **computer**, not to MiRo, and choose it in *Settings → Sound → Output*.

---

## Run

**1. Start the AI and open the MiRo container.** Put your MiRo's IP address instead of `192.168.0.42`:

```bash
cd ~/Ext_Cog_Arch
./docker/run_docker.bash --robot-ip 192.168.0.42 --local-ai
```

This starts the AI services, then opens a terminal *inside* the MiRo container, connected to the robot. The prompt
changes to `root@...#`, and the status box shows `mode: robot`.

**2. Start MiRo's brain** (inside the container). Pick the line for your room:

```bash
cd $HEYMIRO_ROBOT
./run_robot.sh
```

This uses **MiRo's own microphones**, which is best in a quiet room.

```bash
cd $HEYMIRO_ROBOT
./run_robot.sh --ext-mic
```

This uses **the computer's microphone** (or a USB microphone plugged into it). Use it in noisy rooms, such as at
an event.

MiRo starts moving and looking around. Keep this terminal open while you use MiRo.

**3. Talk to it:** stand in front of MiRo and say **"Hey MiRo!"**. MiRo turns towards you if you hold up an AprilTag,
leans in and listens. Things to try:

- "What's your favourite band?"
- "Can you spin around?"
- "Wiggle your ears!"
- "Do your Mickey Mouse impression!"
- "Can you dance to Dancing Queen?"
- "Do some tricks!" (then show hand signs: open palm, pointing up, victory, "I love you")
- "Guess what I'm holding!" (hold an object up to MiRo's camera)
- **"Dance MiRo!"**
- **"Stop MiRo!"**

Stroking MiRo's head or back makes it happy. If nobody speaks for about 3 s, MiRo says goodbye and goes back to
exploring.

---

## Optional

**Send a command without speaking.** Open a second terminal:

```bash
cd ~/Ext_Cog_Arch
./docker/run_docker.bash
```

Running it a second time attaches a new terminal to the same MiRo container.

```bash
rostopic pub -1 /miro/core/forebrain/command std_msgs/String "data: 'dance abba'"
```

Instead of `dance abba` you can use:

| Command | MiRo does |
|---|---|
| `converse` | starts a conversation, like "Hey MiRo" |
| `dance` | dances to a random song |
| `tricks` | does tricks |
| `identify` | plays the object game |
| `say Hello!` | says the words |
| `stop` | stops everything |

**A bigger language model (9B).** It gives slightly richer conversation, but answers about 0.25 s slower and uses
9.5 GB of GPU memory instead of 7 GB. It picked the right action equally often in our tests. Download it once with
`docker/local-ai/download_models.bash --9b`, then start with `--local-ai 9b` instead of `--local-ai`.

**MiRo's voice from MiRo's own speaker** (quieter, but no extra speaker needed): inside the container, start with:

```bash
HEYMIRO_SET="voice.backend=miro" ./run_robot.sh
```

---

## Stop

1. Press **Ctrl+C** in the terminal running `run_robot.sh`. MiRo stops moving.
2. Leave the container, then stop the AI services to free the GPU:

```bash
exit
docker/local-ai/up.bash --down
```

3. Turn Demo mode back on in the MiRo app if you want MiRo to play on its own.

## Update to the newest version

```bash
cd ~/Ext_Cog_Arch
git pull
./docker/run_docker.bash --pull
docker/local-ai/download_models.bash
```

These get the newest code, the matching MiRo environment, and any new models.

## Check it works

```bash
ping -c 3 192.168.0.42
```

Run this on the computer, with your MiRo's IP. It checks the computer can reach MiRo: you should see three
`bytes from` lines.

```bash
curl -s http://127.0.0.1:8090/health; echo
```

Run this on the computer while the AI is running. It should print `"ok":true,"device":"cuda"`.

```bash
rostopic hz /miro/sensors/package
```

Run this inside the container. It shows MiRo's sensor data arriving: about 50 per second. Press Ctrl+C to stop it.

---

## If something goes wrong

| Problem | Fix |
|---|---|
| `permission denied ... docker.sock` | Log out and back in after setup step 1. |
| `WARNING: Docker cannot use an NVIDIA GPU` / slow answers | Setup step 2 failed. Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), then run `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`. |
| `ping` fails | MiRo and the computer are not on the same Wi-Fi, or the IP address is wrong (check the app). |
| `run_robot.sh` waits forever, or `rostopic hz` shows nothing | Is MiRo on? Does `ping` work? Some networks block the robot from reaching the computer: stop, then start again with your computer's IP address added: `./docker/run_docker.bash --robot-ip 192.168.0.42 --host-ip <your computer's IP> --local-ai`. `hostname -I` shows it. |
| MiRo moves strangely or twitches | Demo mode is still on in the MiRo app. Turn it off. |
| MiRo doesn't react to "Hey MiRo" | Speak closer and louder, or use `--ext-mic` with a microphone near you. Inside the container, `python3 tools/mic_check.py --topic sensors/mics` shows the microphone level. |
| No sound | Check *Settings → Sound → Output* on the computer. |
| `container 'heymiro' is already running` | Use the terminal you already have open, or run `docker stop heymiro`. |
| GPU out of memory with `9b` | Use `--local-ai` (the 4B model) instead. |
| Anything else | The logs are in the container, in `~/.ros/heymiro/<date>/`. Send the last lines of `main.log` to the team. |

**Short version, once set up (with your MiRo's IP):**

```bash
cd ~/Ext_Cog_Arch && ./docker/run_docker.bash --robot-ip 192.168.0.42 --local-ai
cd $HEYMIRO_ROBOT && ./run_robot.sh
```
