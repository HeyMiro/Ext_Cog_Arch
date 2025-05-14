# Docker_machine

This repo contains the guide on how to build the docker image and run it with a local folder.

# Install Docker and Pull Repo/Image:

1. Install Docker on your local Ubuntu machine: ```https://docs.docker.com/engine/install/ubuntu/```
2. Install Visual Studio: ```https://code.visualstudio.com/``` (I recommend using VS code for code development).
3. Add Docker plugin in VS code, and I use the 'Dev Containers' plugin to write and debug codes within Docker.
4. Pull the current repo: ```git clone https://github.com/HeyMiro/Ext_Cog_Arch.git```
5. Run command in one terminal： ```docker pull aung9htet/ros_miro:public```

# To Run Docker with GPU (Optional, Dependent on your laptop):

1. You should have a Nividia Driver installed in your local machine.
2. Install the Nvidia docker toolkit on your local machine: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html.

# Run Docker without sudo (non-root user):

```
https://docs.docker.com/engine/install/linux-postinstall/
```

# Running And Saving Docker Image

To run the docker image:
```
./run_docker.bash
```
1. The run_docker.bash contains the commands to allow the Docker to better run on your local machine with your folder. 

2. Some commands you should modify:

   1. --volume="/home/cavaa/docker_file:/docker-ros" needs to be changed for part of it, i.e. '/home/cavaa/docker_file', to your absolute local folder path.
   2.  ```aung9htet/ros_miro:public```:  This is the Docker image name, you can change it.

2. save_docker.bash:

    1. This helps to save the changes you have made in the Image (for example, you install some new packages). For a separate image, you can modify line, ```docker commit $CONTAINER_ID aung9htet/ros_miro:public```, to ```docker commit $CONTAINER_ID yourusername/yourimage:tag```.

# Running MiRo Demo

1. To run the code, you will need to setup the MiRo first. Through the MiRo app, you can find the IP address for your MiRo. This IP will need to be replaced for ```ROS_MASTER_IP``` in ```/root/mdk/share/config/setup.bash```.

2. The next step will be to run the MiRo demo mode. This can be found and run in:
```
cd ~/mdk/catkin_ws/src/New_Miro_Demo/core/
./client_demo.py
```

   
   

