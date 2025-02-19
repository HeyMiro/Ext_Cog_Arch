# Docker_machine
This repo contains the guide on how to build the docker image and run it with a local folder.
# Install Docker and pull Image:
1. Install Docker on your local Ubuntu machine: https://docs.docker.com/engine/install/ubuntu/
2. Install Visual Studio: https://code.visualstudio.com/. (I recommend using VS code for code development).
3. Add Docker plugin in VS code, and I use the 'Dev Containers' plugin to write and debug codes within Docker.
4. Run command in one terminal： docker pull ruidong14/ros_torch
# To run Docker with GPU:
1. You should have a Nividia Driver installed in your local machine.
2. Install the Nvidia docker toolkit on your local machine: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html.
# Run Docker without sudo (non-root user):
1. https://docs.docker.com/engine/install/linux-postinstall/
# Run Docker Image:
1. Create a folder named docker_file and a subfolder called local_ws in your local machine.
2. You will need to remove the container that already exists with the same name before you run Docker.
3. cd docker_file
4. ./run_docker.bash 
# Build ROS workspace in the local_ws folder: 
https://wiki.ros.org/catkin/Tutorials/create_a_workspace
# Follow the README.md in docker_file/local_ws. 
