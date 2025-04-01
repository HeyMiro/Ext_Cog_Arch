export CONTAINER_ID=`docker ps -lq`
docker commit $CONTAINER_ID aung9htet/ros_miro:public
echo "done"
