#!/usr/bin/env python
import rospy
import numpy as np
import math
from math import pi
from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from respawnGoal import Respawn

class Env():
    def __init__(self, action_size):
        self.goal_x = 0
        self.goal_y = 0                                     # 将宝箱的x , y 坐标进行初始化
        self.heading = 0                                    # 机器人与目标的角度初始化 
        self.action_size = action_size                      # 定义 action_size， 这个值是我们在强化学习算法的代码中，需要输入的，通过输入动作，来决定下一步的状态(state_ , reward , done )
        self.initGoal = True
        self.get_goalbox = False                            # 先将是否得到宝箱进行定义，为假
        self.position = Pose()
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)     # 定义 线速度角速度的发布者
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.getOdometry)   # 定义 odom的 接受者
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)  # 重置GAZEBO虚拟环境的位置
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty) # 暂停物理更新
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)     # 恢复物理更新       服务器的数据类型为 std_srvs.srv 中的 Empty 空类型 
        self.respawn_goal = Respawn()     # turtlebot3 自定义的宝箱位置的函数

    def getGoalDistace(self):     # 计算机器人到宝箱的距离
        goal_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2) # math.hypot(x,y)计算两向量（可以通过点之间的相减）之间的距离

        return goal_distance   # 返回距离值

    def getOdometry(self, odom):
        self.position = odom.pose.pose.position                 #  里程计 位置
        orientation = odom.pose.pose.orientation                #  里程计 方向
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]   # 方向的定义 为一个 列表
        _, _, yaw = euler_from_quaternion(orientation_list)     # 四圆数变换之后 才可以进行发布 ，  其中变换的结果 yaw 是 偏移角度

        goal_angle = math.atan2(self.goal_y - self.position.y, self.goal_x - self.position.x)  # atan2 计算（x，y）之间的 反正切值

        heading = goal_angle - yaw      # 与 宝箱的 角度 
        if heading > pi:
            heading -= 2 * pi

        elif heading < -pi:
            heading += 2 * pi

        self.heading = round(heading, 2)   # round（x，数字） x是输入的数值 ， 数字是 保留的小数后几位 ，默认值为0 

    def getState(self, scan):    # 获取状态值
        scan_range = []           # 创建的雷达的列表
        heading = self.heading       
        min_range = 0.13            
        done = False        #初始化 done的状态值   

        for i in range(len(scan.ranges)):        #scan.ranges:是雷达的距离数据 是 矩阵
            if scan.ranges[i] == float('Inf'):    #如果数据无限大 给自己创建的雷达值列表添加 3.5    Nan （float 数据）: Not a number(没有数字)  Inf : Infinity(无穷大) 
                scan_range.append(3.5)          # 在 雷达探测很远的 情况下 回来的数据 为 inf    在 近似为 无 距离的时候  返回的值 为 Nan    
            elif np.isnan(scan.ranges[i]):        # np.isnan(数值) return值为 True or Flase    
                scan_range.append(0)                # 如果 数据
            else:
                scan_range.append(scan.ranges[i])

        if min_range > min(scan_range) > 0:         # 如果 最小数值 大于  最小的雷达数值 并且 大于 0 的情况 done为
            done = True 

        current_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y),2)   # 计算 当前的距离 
        if current_distance < 0.2:   # 如果 当前机器人到宝箱的距离小于 0.2 时  
            self.get_goalbox = True     # 获得宝箱 为 True

        return scan_range + [heading, current_distance], done     #返回    雷达的数值 ， 【距离目标的角度 ， 距离目标的距离 】  ,  done

    def setReward(self, state, done, action):
        yaw_reward = []                          # 偏移角 获得的 奖励 
        current_distance = state[-1]             # 当前的 距离 为 state的 最后一位 
        heading = state[-2]                      # 当前的 角度 为 state的 倒数第二位

        for i in range(5):
            angle = -pi / 4 + heading + (pi / 8 * i) + pi / 2
            tr = 1 - 4 * math.fabs(0.5 - math.modf(0.25 + 0.5 * angle % (2 * math.pi) / math.pi)[0])
            yaw_reward.append(tr)

        distance_rate = 2 ** (current_distance / self.goal_distance)
        reward = ((round(yaw_reward[action] * 5, 2)) * distance_rate)

        if done:                              # 如果 结束 
            rospy.loginfo("Collision!!")        # 输出 撞墙了！！
            reward = -200                      #  给出个 最大的 惩罚值 
            self.pub_cmd_vel.publish(Twist())      

        if self.get_goalbox:
            rospy.loginfo("Goal!!")
            reward = 200
            self.pub_cmd_vel.publish(Twist())
            self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)
            self.goal_distance = self.getGoalDistace()
            self.get_goalbox = False

        return reward

    def step(self, action):                   # 返回 状态 ，奖励  ， done 
        max_angular_vel = 1.5
        ang_vel = ((self.action_size - 1)/2 - action) * max_angular_vel * 0.5

        vel_cmd = Twist()
        vel_cmd.linear.x = 0.15
        vel_cmd.angular.z = ang_vel
        self.pub_cmd_vel.publish(vel_cmd)

        data = None                                           # scan 的 data 
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass

        state, done = self.getState(data)
        reward = self.setReward(state, done, action)

        return np.asarray(state), reward, done

    def reset(self):                                 #  返回 状态值
        rospy.wait_for_service('gazebo/reset_simulation')
        try:
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print("gazebo/reset_simulation service call failed")

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass

        if self.initGoal:
            self.goal_x, self.goal_y = self.respawn_goal.getPosition()
            self.initGoal = False

        self.goal_distance = self.getGoalDistace()
        state, done = self.getState(data)

        return np.asarray(state)