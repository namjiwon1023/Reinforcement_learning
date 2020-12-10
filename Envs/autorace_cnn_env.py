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

import skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer
from cv_bridge import CvBridge, CvBridgeError

class Env():
	def __init__(self, action_size):

		self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)     
		self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)  
		self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty) 
		self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)     
		self.action_size = action_size 
		self.first_state_image = None

		# self.last50actions = [0] * 50

		self.img_rows = 224      
		self.img_cols = 224
		# self.channel = 4      

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

		max_angular_vel = 1.5   
		ang_vel = ((self.action_size - 1)/2 - action) * max_angular_vel * 0.5
	
		vel_cmd = Twist()         
		vel_cmd.linear.x = 0.08         
		vel_cmd.angular.z = ang_vel         
		self.pub_cmd_vel.publish(vel_cmd)

		# if action == 0: 
            		# vel_cmd = Twist()
            		# vel_cmd.linear.x = 0.10
            		# vel_cmd.angular.z = 0.0
            		# self.pub_cmd_vel.publish(vel_cmd)

		# elif action == 1: 
            		# vel_cmd = Twist()
            		# vel_cmd.linear.x = 0.05
            		# vel_cmd.angular.z = 0.3
            		# self.pub_cmd_vel.publish(vel_cmd)

		# elif action == 2: 
            		# vel_cmd = Twist()
            		# vel_cmd.linear.x = 0.05
            		# vel_cmd.angular.z = -0.3
            		# self.pub_cmd_vel.publish(vel_cmd)

		data = None
        	if data is None:
            		
                	data = rospy.wait_for_message('/scan', LaserScan, timeout=5)

        	done = self.getState(data)

        	next_image_data = None
        	# success=False
        	# cv_image = None

        	# if (image_data is None or success is False):
		if next_image_data is None :

                	next_image_data = rospy.wait_for_message('/camera/image', Image, timeout=5)
                	# h = image_data.height
                	# w = image_data.width
                	next_cv_image = CvBridge().imgmsg_to_cv2(next_image_data, "bgr8")
			next_cv_image = cv2.cvtColor(next_cv_image, cv2.COLOR_BGR2RGB)
			next_cv_image = next_cv_image.astype(np.float32)
			next_cv_image /= 255
			# x_t1 = skimage.color.rgb2gray(cv_image)
                
                	# if not (cv_image[h//2,w//2,0]==178 and cv_image[h//2,w//2,1]==178 and cv_image[h//2,w//2,2]==178):
                    		# success = True
                	# else:
                    		# print("/camera/image ERROR, retrying")

        	# self.last50actions.pop(0) 

        	# if action == 0:
            		# self.last50actions.append(0)
        	# else:
            		# self.last50actions.append(1)

        	# action_sum = sum(self.last50actions)

        	# laser_len = len(data.ranges)
        	# left_sum = sum(data.ranges[laser_len-(laser_len/5):laser_len-(laser_len/10)]) 
        	# right_sum = sum(data.ranges[(laser_len/10):(laser_len/5)]) 

        	# center_detour = abs(right_sum - left_sum)/5

        	# if not done:
            		# if action == 0:
                		# reward = 1 / float(center_detour+1)
				# reward = 0.03
            		# elif action_sum > 45:
			# elif action == 1:
                		# reward = 11.28
            		# else action == 2: 
                		# reward = 0.5 / float(center_detour+1)
				# reward = 22.53
			# elif action == 4:
				# reward = 11.28
			# elif action == 5:
				# reward = 0.03
        	# else:
            		# reward = -100

		
		if not done:
                 
			reward = round(15*(max_angular_vel - abs(ang_vel) +0.0335), 2)                    			
		else:                     
			
			reward = -500  

		x_t1 = skimage.color.rgb2gray(next_cv_image)
        	# cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
 		x_t1 = skimage.transform.resize(x_t1,(self.img_rows, self.img_cols),mode='constant')
        	# cv_image = cv2.resize(cv_image, (self.img_rows, self.img_cols))
		x_t1 = skimage.exposure.rescale_intensity(x_t1, out_range=(0, 255))
		x_t1 = x_t1.reshape(1, 1, x_t1.shape[0], x_t1.shape[1])
		next_state = np.append(x_t1, self.first_state_image[:, :3, :, :], axis=1)

        	# state = cv_image.reshape(1, self.channel, cv_image.shape[0], cv_image.shape[1])
        	return next_state, reward, done 

	def reset(self):

		# self.last50actions = [0] * 50 

		rospy.wait_for_service('/gazebo/reset_simulation')
		try:
			self.reset_proxy()
		except (rospy.ServiceException) as e:
			print ("/gazebo/reset_simulation service call failed")

		first_image_data = None
		# success=False
		# cv_image = None
		
		# if (image_data is None or success is False):
		if first_image_data is None :

                	first_image_data = rospy.wait_for_message('/camera/image', Image, timeout=5)
                	# h = image_data.height
                	# w = image_data.width
                	first_cv_image = CvBridge().imgmsg_to_cv2(first_image_data, "bgr8")
			first_cv_image = cv2.cvtColor(first_cv_image, cv2.COLOR_BGR2RGB)
			first_cv_image = first_cv_image.astype(np.float32)
			first_cv_image /= 255
                	# x_t = skimage.color.rgb2gray(cv_image)
                	# if not (cv_image[h//2,w//2,0]==178 and cv_image[h//2,w//2,1]==178 and cv_image[h//2,w//2,2]==178):
                    		# success = True
                	# else:
                    		# print("/camera/image ERROR, retrying")

		x_t = skimage.color.rgb2gray(first_cv_image)
        	# cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
		x_t = skimage.transform.resize(x_t,(self.img_rows, self.img_cols),mode='constant')
        	# cv_image = cv2.resize(cv_image, (self.img_rows, self.img_cols))
		x_t = skimage.exposure.rescale_intensity(x_t,out_range=(0,255))
		s_t = np.stack((x_t, x_t, x_t, x_t), axis=0)
		first_state = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])
		self.first_state_image = first_state
        	# state = cv_image.reshape(1, self.channel, cv_image.shape[0], cv_image.shape[1])
        	return first_state
