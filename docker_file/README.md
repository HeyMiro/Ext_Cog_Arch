This is the local folder that contains the ROS workspace and some codes. Because Docker can't save any changes you make automatically after you close it, I built this folder that can allow you to write and save the code locally, and thus run it within Docker.


#run_docker.bash:
1. The run_docker.bash contains the commands to allow the Docker to better run on your local machine with your folder. 
2. some commands you should modify:
   1). --volume="/home/primi/docker_file:/docker-ros" \     you need to change '/home/primi/docker_file' to your absolute local folder path
   2). ros_torch:v2.0    This is the Docker image name, you can change it.

#save_docker.bash:
1. This helps to save the changes you have made in the Image (for example, you install some new packages).

   
   
