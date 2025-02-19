#!/usr/bin/env python3
import sys
import rospy
import moveit_commander
import geometry_msgs.msg
import torch
from torch import nn
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CameraInfo, Image
import cv2
import datasets

N_JOINTS = 6
GENERATE_DATA = False


class Robot:
    def __init__(self, max_states, args):
        self.currentView = None
        # self.brain = Brain()
        moveit_commander.roscpp_initialize(args)
        self.robot = moveit_commander.RobotCommander()
        self.group_name = self.robot.get_group_names()[0]
        self.move_group = moveit_commander.MoveGroupCommander(self.group_name)
        self.scene = moveit_commander.PlanningSceneInterface()

        # Add the ground plane as collision object
        self.plane_pose = geometry_msgs.msg.PoseStamped()
        self.plane_pose.header.frame_id = "world"
        self.plane_pose.pose.orientation.w = 1.0
        self.scene.add_plane("ground_plane", self.plane_pose)

        self.move_group.set_max_acceleration_scaling_factor(1)
        self.move_group.set_max_velocity_scaling_factor(1)
        self.bridge = CvBridge()

        self.states = torch.zeros(max_states, 6)

    def babble(self):
        # Generate a randm goal position
        n_state = (torch.rand(1, N_JOINTS) - 0.5)*3.14*2

        success = False
        self.move_group.set_joint_value_target(n_state.numpy().tolist()[0])
        success, trajectory, planning_time, error_code = self.move_group.plan()
        self.move_group.execute(trajectory, wait=True)

        return n_state, success

    def getMirrorView( self, data ):        
        # Getting a snapshot of the robot
        # print(f'Getting a snapshot')
        self.currentView = data


    def collectDataSample( self, state, idx ):
        # Collects and savean image to the dataset
        print(f'Collecting data sample {idx}')
        if self.currentView is not None:
            img = self.bridge.imgmsg_to_cv2(self.currentView, desired_encoding="bgr8")
            name = f'view_{idx}.png'
            cv2.imwrite(name, img)
            cv2.waitKey(3)
            self.states[idx] = state
       
    def saveStates(self):
        torch.save(self.states, 'states.pt')

    def loadStates(self):
        self.states = torch.load('states.pt')
        print(f'Previous tensor: {self.states}')

    def inferState( self, view ):
        # Given a view generate a state
        print('Inferring states')

'''
Motor babbling example. Random explorations in the phase space
'''
def main():
    
    T = 500
    robot = Robot(T, sys.argv)
    camera_topic = "/static_camera/image_raw"
    # camera_topic = "/camera/rgb/image_raw"
    
    rospy.init_node('main', anonymous=True)

    rospy.Subscriber(name=camera_topic,
                        data_class=Image,
                        callback=robot.getMirrorView, 
                        queue_size=1)

    rospy.sleep(2.0)

    run = True
    idx = 0

    while(run):
        target_state, success = robot.babble()
        print('Executing movement')
        
        if success:
            print('Saving data sample')
            robot.collectDataSample(target_state, idx)
            idx = idx + 1

        if idx >= T:
            run = False

        rospy.sleep(0.5)
    
    robot.saveStates()


if __name__ == "__main__":
    main()
