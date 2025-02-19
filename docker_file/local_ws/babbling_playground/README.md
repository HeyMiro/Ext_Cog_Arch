# babbling_playground


## Requeriments: 
ROS 1 Noetic, Gazebo, Catkin

## To setup:

1. Create a catkin workspace
2. Clone the repository into the src folder
3. Outside the src directory, install the dependencies: 
```
rosdep install -i --from-path src
```
4. Build the workspace 
```
catkin build babbling_example
```
5. Source the workspace
```
source devel/setup.bash 
```

## To run

1. Run the simulator:
```
roslaunch babbling_example simulation.launch
```

2. Run the babbling routine: 
```
rosrun babbling_example mirror_babbling.py
```

This routine, just moves the robot and takes a snapshot, storing the (image, state) pair as a training example.
