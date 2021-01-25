#!/usr/bin/python
# -*- coding:utf8 -*-
import carla
import random
import time
import numpy as np
import cv2
# 创建服务器
client = carla.Client('localhost', 2000)
print ('create client')
client.set_timeout(2.0)  # 设置等待时间
world = client.get_world()   # 获得世界
print ('crated VR world')

blueprint_library = world.get_blueprint_library() # 获取蓝图库
bp = blueprint_library.filter('model3')[0]  # 创建一个汽车蓝图

# 生成汽车
spawn_point = random.choice(world.get_map().get_spawn_points()) # 随机选择出生点
vehicle = world.spawn_actor(bp, spawn_point) # 随机在出生点生成汽车

# 控制汽车
vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0)) # (vehicle.set_autopilot(True)自动驾驶)
actor_list.append(vehicle)

# 设置摄像头
cam_bp = blueprint_library.find('sensor.camera.rgb')
# 设置摄像头的分辨率
cam_bp.set_attribute('image_size_x', '{}'.format(IM_WIDTH))
cam_bp.set_attribute('image_size_y', '{}'.format(IM_HEIGHT))
cam_bp.set_attribute('fov', '110') # 'fov' feel of view


spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7)) # 相机位置
sensor = world.spawn_actor(cam_bp, spawn_point, attach_to = vehicle) # 将传感器连接(attach)到汽车上
actor_list.append(sensor)

# 获取摄像头图片 ， 这里通过 .listen 的 lambda 函数传回数据。当然不要忘了设计一个延时。
sensor.listen(lambda data: process_img(data))
time.sleep(25)


# 摄像头获得的图片有四个通道 “rgba”，需要将第四个通道去掉，并用opencv 的 cv2.imshow() 将摄像头捕捉到的图片 归一化之后回传回来。
def process_img(image):
    i = np.array(image.raw_data)
    # print(i.shape)
    i2 = i.reshape((IM_HEIGHT,IM_WIDTH,4))  #4 changels "rgba"
    i3 = i2[:,:,:3]  # 3 changels "rgb"

    cv2.imshow(', i3')
    cv2.waitKey(1)
    return i3/255.0

