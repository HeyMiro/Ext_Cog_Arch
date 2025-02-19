export CONTAINER_ID=`docker ps -lq`
docker commit $CONTAINER_ID ros_miro:v2.0
echo "done"
