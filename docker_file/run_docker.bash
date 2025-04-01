xhost +local:root

XAUTH=/tmp/.docker.xauth

docker run -it \
       --name=ros_container \
       --env="DISPLAY=$DISPLAY" \
       --env="QT_X11_NO_MITSHM=1" \
       --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
       --volume="/home/cavaa/docker_file:/docker-ros" \
       --env="XAUTHORITY=$XAUTH" \
       --volume="$XAUTH:$XAUTH" \
       --net=host \
       --privileged \
       --gpus all \
       --device /dev/snd \
       --group-add audio \
       --env="PULSE_SERVER=unix:/run/user/1000/pulse/native" \
       --volume="/run/user/1000/pulse:/run/user/1000/pulse" \
       --volume="$HOME/.config/pulse/cookie:/root/.config/pulse/cookie" \
       aung9htet/ros_miro:public \
       bash

echo "Done"
