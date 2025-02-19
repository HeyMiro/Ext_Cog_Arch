# This contains the codes for ROS, I made changes based on Alejandro's repo: 
1. Download and extract the babbling_playground folder into your catkin_ws's src folder.
2. You should also git clone the UR robot driver into the src folder:https://github.com/ros-industrial/universal_robot
3. Once you download this, I have made some changes in the ur controller, please replace the catkin_ws/src/universal_robot/ur_gazebo/launch/inc/ur_control.launch.xml with the ur_control.launch.xml I provided. 
4. Build the workspace:
   catkin_make
5. source the workspace:
   source devel/setup.bash
# Run simulation and data collection
1. In the catkin_ws:
   roslaunch babbling_example sim_ur5.launch
2. Run data collection:
   rosrun babbling_example data_collection.py
