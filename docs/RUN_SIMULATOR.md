# Run Hey MiRo in the Gazebo simulator

A 3D MiRo on screen that you talk to through the computer's microphone. All the AI runs on this computer; no API
keys are needed. For the physical robot, see [RUN_ROBOT.md](RUN_ROBOT.md).

**You need:** Ubuntu 22.04/24.04, an NVIDIA GPU with 8 GB or more (without one it still works, with ~15 s answers),
40 GB of free disk, a microphone, and **headphones** (so MiRo doesn't hear itself).

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

## Run

**1. Start the AI and open the MiRo container:**

```bash
cd ~/Ext_Cog_Arch
./docker/run_docker.bash --sim --local-ai
```

This starts the AI services, waits until the models are loaded, then opens a terminal *inside* the MiRo container.
The prompt changes to `root@...#`. The first time, it also downloads the MiRo environment (~11 GB).

**2. Start the simulator and MiRo's brain** (inside the container):

```bash
cd $HEYMIRO_SIM
./run_sim.sh
```

After 20–60 s the 3D window opens and the terminal shows:

```
[forebrain] local llm service ready
[forebrain] local speech service ready
```

Keep this terminal open while you use MiRo.

**3. Talk to it:** say **"Hey MiRo!"**, then talk normally. Things to try:

- "What's your favourite band?"
- "Can you spin around?"
- "Wiggle your ears!"
- "Do your Mickey Mouse impression!"
- "Can you dance to Dancing Queen?"
- **"Dance MiRo!"**
- **"Stop MiRo!"**

If nobody speaks for about 3 s, MiRo says goodbye and goes back to exploring.

---

## Optional

**A computer voice talks to MiRo** (a hands-free demo, or no microphone needed). In step 2, run `./run_sim.sh
--no-bridge` instead of `./run_sim.sh`. Then open a **second terminal**:

```bash
cd ~/Ext_Cog_Arch
./docker/run_docker.bash
```

Running it a second time attaches a new terminal to the same MiRo container.

```bash
cd $HEYMIRO_SIM
python3 tools/fake_user.py --play
```

A synthetic voice says "Hey MiRo!", asks a question, asks for a spin and a dance, then says "Stop MiRo!". You hear
both voices. At the end it prints how quickly MiRo reacted to each line.

**Send a command without speaking** (in a second container terminal, as above):

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

---

## Stop

1. Press **Ctrl+C** in the terminal running `run_sim.sh`. This stops MiRo and closes the simulator.
2. Leave the container, then stop the AI services to free the GPU:

```bash
exit
docker/local-ai/up.bash --down
```

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
curl -s http://127.0.0.1:8090/health; echo
```

Run this on the computer while the AI is running. It should print `"ok":true,"device":"cuda"`. `"device":"cpu"`
means the GPU is not being used.

```bash
cd $HEYMIRO_SIM && python3 -m unittest discover -s tests 2>&1 | grep -E "^(Ran|OK|FAILED)"
```

Run this inside the container. It runs the built-in tests and should end with `OK`.

```bash
cd $HEYMIRO_SIM && python3 tools/eval_dialogue.py
```

Run this inside the container while the AI is running. It asks the AI 40 test questions and scores the format,
the choice of action and the speed. Expect 40/40 valid answers and F1 above 0.9.

---

## If something goes wrong

| Problem | Fix |
|---|---|
| `permission denied ... docker.sock` | Log out and back in after setup step 1. |
| `WARNING: Docker cannot use an NVIDIA GPU` / slow answers | Setup step 2 failed. Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), then run `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`. |
| `no models yet` | Do setup step 4. |
| `local llm service NOT reachable` | You started without `--local-ai`. Stop and start again with it. |
| `container 'heymiro' is already running` | Use the terminal you already have open, or run `docker stop heymiro`. |
| MiRo doesn't react to "Hey MiRo" | Check *Settings → Sound → Input* is the right microphone. Inside the container, `python3 tools/mic_check.py --topic host/mics` shows the level. |
| MiRo talks to itself | Use headphones. |
| No 3D window | Run `xhost +local:root` on the computer, then start again. |
| Anything else | The logs are in the container, in `~/.ros/heymiro/<date>/`. Send the last lines of `main.log` to the team. |

**Short version, once set up:**

```bash
cd ~/Ext_Cog_Arch && ./docker/run_docker.bash --sim --local-ai
cd $HEYMIRO_SIM && ./run_sim.sh
```
