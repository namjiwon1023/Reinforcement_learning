#!/usr/bin/env python
import rospy
import numpy as np
import cv2
import sys
import os
import random
import math
from math import pi
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from cv_bridge import CvBridge, CvBridgeError

class Env():
	def __init__(self, action_size):

		self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)     
		self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)  
		self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty) 
		self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)     
		self.action_size = action_size 

		self.last50actions = [0] * 50

		self.img_rows = 224      
		self.img_cols = 224      

	def getState(self, scan):   
		scan_range = []            
        	min_range = 0.11       
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

        	return done     

	def step(self, action):

		if action == 0: 
            		vel_cmd = Twist()
            		vel_cmd.linear.x = 0.10
            		vel_cmd.angular.z = 0.0
            		self.pub_cmd_vel.publish(vel_cmd)

		elif action == 1: 
            		vel_cmd = Twist()
            		vel_cmd.linear.x = 0.05
            		vel_cmd.angular.z = 0.2
            		self.pub_cmd_vel.publish(vel_cmd)

		elif action == 2: 
            		vel_cmd = Twist()
            		vel_cmd.linear.x = 0.05
            		vel_cmd.angular.z = -0.2
            		self.pub_cmd_vel.publish(vel_cmd)

		data = None
        	if data is None:
            		
                	data = rospy.wait_for_message('/scan', LaserScan, timeout=5)

        	done = self.getState(data)

        	image_data = None
        	success=False
        	cv_image = None

        	# if (image_data is None or success is False):
		if image_data is None :

                	image_data = rospy.wait_for_message('/camera/image', Image, timeout=5)
                	h = image_data.height
                	w = image_data.width
                	cv_image = CvBridge().imgmsg_to_cv2(image_data, "bgr8")
                
                	# if not (cv_image[h//2,w//2,0]==178 and cv_image[h//2,w//2,1]==178 and cv_image[h//2,w//2,2]==178):
                    		# success = True
                	# else:
                    		# print("/camera/image ERROR, retrying")

        	self.last50actions.pop(0) 

        	if action == 0:
            		self.last50actions.append(0)
        	else:
            		self.last50actions.append(1)

        	action_sum = sum(self.last50actions)

        	laser_len = len(data.ranges)
        	left_sum = sum(data.ranges[laser_len-(laser_len/5):laser_len-(laser_len/10)]) 
        	right_sum = sum(data.ranges[(laser_len/10):(laser_len/5)]) 

        	center_detour = abs(right_sum - left_sum)/5

        	if not done:
            		if action == 0:
                		reward = 1 / float(center_detour+1)
            		elif action_sum > 45:
                		reward = -0.5
            		else: 
                		reward = 0.5 / float(center_detour+1)
        	else:
            		reward = -100


        	# cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        	cv_image = cv2.resize(cv_image, (self.img_rows, self.img_cols))

        	state = cv_image.reshape(1, 3, cv_image.shape[0], cv_image.shape[1])
        	return state, reward, done 

	def reset(self):

		self.last50actions = [0] * 50 

		rospy.wait_for_service('/gazebo/reset_simulation')
		try:
			self.reset_proxy()
		except (rospy.ServiceException) as e:
			print ("/gazebo/reset_simulation service call failed")

		image_data = None
		success=False
		cv_image = None
		
		# if (image_data is None or success is False):
		if image_data is None :

                	image_data = rospy.wait_for_message('/camera/image', Image, timeout=5)
                	h = image_data.height
                	w = image_data.width
                	cv_image = CvBridge().imgmsg_to_cv2(image_data, "bgr8")
                
                	# if not (cv_image[h//2,w//2,0]==178 and cv_image[h//2,w//2,1]==178 and cv_image[h//2,w//2,2]==178):
                    		# success = True
                	# else:
                    		# print("/camera/image ERROR, retrying")


        	# cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        	cv_image = cv2.resize(cv_image, (self.img_rows, self.img_cols))

        	state = cv_image.reshape(1, 3, cv_image.shape[0], cv_image.shape[1])
        	return state
