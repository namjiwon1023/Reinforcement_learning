#!/usr/bin/env python
import rospy
import numpy as np
import math
from math import pi
from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty


class Env():
    def __init__(self, action_size):                                  
        self.action_size = action_size                                            
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)     
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)  
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty) 
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)     

    def getState(self, scan):   
        scan_range = []            
        min_range = 0.120         
        done = False       

        for i in range(len(scan.ranges)):        
            if scan.ranges[i] == float('Inf'):    
                scan_range.append(3.5)         
            elif np.isnan(scan.ranges[i]):      
                scan_range.append(0)               
            else:
                scan_range.append(scan.ranges[i])

        if min_range > min(scan_range) > 0:         
            done = True 

        return scan_range , done     

    def step(self, action):               

        rospy.wait_for_service('/gazebo/unpause_physics')    
        try:
            self.unpause_proxy()              
        except (rospy.ServiceException) as e:   
            print ("/gazebo/unpause_physics service call failed")

        max_angular_vel = 1.5
        ang_vel = ((self.action_size - 1)/2 - action) * max_angular_vel * 0.5

        vel_cmd = Twist()
        vel_cmd.linear.x = 0.15
        vel_cmd.angular.z = ang_vel
        self.pub_cmd_vel.publish(vel_cmd)

        data = None                                           
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass
        rospy.wait_for_service('/gazebo/pause_physics')   
        try:
            self.pause_proxy()             
        except (rospy.ServiceException) as e: 
            print ("/gazebo/pause_physics service call failed")

        state, done = self.getState(data)

        if not done:    
            reward = round(15*(max_angular_vel - abs(ang_vel) +0.0335), 2)           
        else:        
            reward = -500  

        return np.asarray(state), reward, done

    def reset(self):                                 
        rospy.wait_for_service('gazebo/reset_simulation')
        try:
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print("gazebo/reset_simulation service call failed")

        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause_proxy()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")
   
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause_proxy()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")


        state, done = self.getState(data)

        return np.asarray(state)
