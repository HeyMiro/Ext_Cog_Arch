#!/usr/bin/env python3
import os
from moveit_msgs.msg import RobotState, Constraints, OrientationConstraint
import rospy, sys
import geometry_msgs.msg
from geometry_msgs.msg import Vector3, Quaternion, Transform, Pose, Point, Twist, Accel,PoseStamped
import rospy
from std_msgs.msg import String,Float64MultiArray,MultiArrayDimension
import numpy as np
import math
import moveit_commander
import time
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CameraInfo, Image
import cv2
from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import SetModelState

class Robot:
    def __init__(self,arm) -> None:
        self.arm=arm
        self.bridge = CvBridge()
        self.image=None
        self.robot_config=arm.get_current_joint_values()
    def get_images(self,data):
        self.image=data
    
    def return_configure(self):
        pos=self.arm.get_current_pose().pose
        self.robot_config=arm.get_current_joint_values()
        tip=[pos.position.x,pos.position.y,pos.position.z]
        return tip,self.robot_config
    
    def actions(self,rand):
       
        
        
        pos1=np.array([0.125570,4.073,-1.02,-0.62,3.2,1.58])
        #right buttom
        if rand==0:
            target_pos=[-0.345,-0.539,0.726]
            limit=[-0.62,4.07,-1.71,-1.24,2.82]
            max_action=[0,0,0,0,0]
            min_action=[-4,0,-4,-4,-2]
            actions=(np.array(max_action)-np.array(min_action))*np.random.random((1,5))+np.array(min_action)
        #right top
        if rand==1:
            target_pos=[-0.5,-0.76,0.726]
            limit=[-0.62,3.18,-0.203,-2,2.94]
            max_action=[0,0,4.7,0,0]
            min_action=[-4,-4.5,0,-9,-2]
            actions=(np.array(max_action)-np.array(min_action))*np.random.random((1,5))+np.array(min_action)
        
        #left top
        if rand==2:
            target_pos=[0.656,-0.654,0.726]
            limit=[0.75,3.18,-0.203,-2,2.94]
            max_action=[3.6,0,4.7,0,0]
            min_action=[0,-4.5,0,-8,-2]
            actions=(np.array(max_action)-np.array(min_action))*np.random.random((1,5))+np.array(min_action)
        
        #left buttom
        if rand==3:
            target_pos=[0.46,-0.441,0.7259]
            limit=[0.75,4.07,-1.71,-1.24,2.82]
            max_action=[3.6,0,0,0,0]
            min_action=[0,0,-4,-4,-2]
            actions=(np.array(max_action)-np.array(min_action))*np.random.random((1,5))+np.array(min_action)
        
        # self.set_pose(target_pos)
        return actions
        
        
        
    def move(self,actions):
       joints=arm.get_current_joint_values()
       new_joints=np.array(joints)
       new_joints[0:5]=new_joints[0:5]+actions*math.pi/180
       
       self.arm.set_joint_value_target(new_joints)
       self.arm.go()
       self.return_configure()
       
    def reset(self):
        pos1=np.array([0.125570,4.073,-1.02,-0.62,3.2,1.58])
       
        arm.set_joint_value_target(pos1)
        arm.go()
    
    def collectDataSample( self, idx,path):
        # Collects and savean image to the dataset
        im_path=os.path.join(path,f'{idx}.png')
        if self.image is not None:
          
            img = self.bridge.imgmsg_to_cv2(self.image, desired_encoding="bgr8")
           
            cv2.imwrite(im_path,img)
    def set_pose(self,pose):
        

        state_msg = ModelState()
        state_msg.model_name = 'unit_cylinder'
        state_msg.pose.position.x = pose[0]
        state_msg.pose.position.y = pose[1]
        state_msg.pose.position.z = pose[2]
        state_msg.pose.orientation.x = 0
        state_msg.pose.orientation.y = 0
        state_msg.pose.orientation.z = 0
        state_msg.pose.orientation.w = 0

        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            resp = set_state( state_msg )

        except rospy.ServiceException:
               print ('Service call failed')
           
        
        
                
            
            



if __name__ == "__main__":
    
    camera_topic = "/camera/color/image_raw"
    nh=rospy.init_node('agent',anonymous=True)
    parent_d='/docker-ros/local_ws/catkin_ws/src/babbling_playground/data_no_goal'
    moveit_commander.roscpp_initialize(sys.argv)
    arm=moveit_commander.MoveGroupCommander('manipulator')
    reference_frame = 'base_link'
    arm.set_pose_reference_frame(reference_frame)
    arm.set_goal_joint_tolerance(0.001)
    arm.set_max_acceleration_scaling_factor(1)
    arm.set_max_velocity_scaling_factor(1)
    arm.set_planer_id = "RRTkConfigDefault"
    arm.set_planning_time(50)
    robot=Robot(arm)
    rospy.Subscriber(name=camera_topic,
                        data_class=Image,
                        callback=robot.get_images, 
                        queue_size=1)
    epoch=1000
    steps=10
    for i in range(epoch):
         robot.reset()
         observations={'tip':[],'joints':[],'actions':[]}
         
         dire=str(i)
         path=os.path.join(parent_d,dire)
         rand=np.random.randint(0,4)
         if not os.path.exists(path):
             os.mkdir(path)
         time.sleep(1)
         for j in range(steps):
             robot.collectDataSample(j,path)
             tip,joints=robot.return_configure()
             action=robot.actions(rand)
        
            #  print(action)
             observations['tip'].append(tip)
             observations['joints'].append(joints)
             observations['actions'].append(list(action[0]))
             robot.move(action[0])
             time.sleep(1)
         np.save(os.path.join(path,'obs.npy'),observations) 
         print(f'Collected data sample {i}')       
            
            
    
    
    