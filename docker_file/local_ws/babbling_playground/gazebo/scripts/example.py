#!/usr/bin/env python3


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

import cv2



def return_tip():
    pos=arm.get_current_pose().pose
    
   
    tip=[pos.position.x,pos.position.y,pos.position.z]
    
def move(angle):
    target_pose = PoseStamped()
    target_pose.header.frame_id = reference_frame
    target_pose.header.stamp = rospy.Time.now()
    target_pose.pose.position.x=float(angle[0])
    target_pose.pose.position.y=float(angle[1])
    target_pose.pose.position.z=float(angle[2])
    target_pose.pose.orientation.x = 0
    target_pose.pose.orientation.y = 1
    target_pose.pose.orientation.z = 0
    target_pose.pose.orientation.w = 0
    
    arm.set_joint_value_target(target_pose,True)
    plan_=arm.plan()
    if type(plan_) is tuple:
        # noetic
        success, plan, planning_time, error_code = plan_
    arm.execute(plan)
def reset():
   
    pos1=np.array([85.27,-41.81,-124.34,-99.20,89.99,0])*math.pi/180
    print(pos1)
    arm.set_joint_value_target(pos1)
    arm.go()
def move_robot(pos1,grab_pos):
    
    move(pos1)
    
   
    
    
    pos1[2]=pos1[2]-0.04
    move(pos1)
    
    pos1[2]=pos1[2]+0.18
    move(pos1)
    
    
    
    
    grab_pos[2]=grab_pos[2]+0.18
    move(grab_pos)
    # grab_pos[2]=0.365
    # grab_pos[2]=0.415
    grab_pos[2]=0.385
    move(grab_pos)
    
    
    grab_pos[2]=0.51
    move(grab_pos)
    reset()

ip = "192.168.0.131"
if __name__ == "__main__":
        pos1=np.array([0.125570,4.073,-1.02,-0.62,3.2,1.58])
        max_action=[0,0,0,0,0]
        min_action=[-4,0,-4,-4,-2]
        actions=(np.array(max_action)-np.array(min_action))*np.random.random((1,5))+np.array(min_action)
        new_pos=pos1[0:5]+actions*math.pi/180
        print(new_pos)
        print(np.random.randint(0,4))