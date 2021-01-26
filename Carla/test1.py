#!/usr/bin/python
# -*- coding:utf8 -*-
import carla
import random
import time
import numpy as np
import cv2
# 创建 Client 对象
client = carla.Client('localhost', 2000)
print ('create client')
client.set_timeout(2.0)  # 设置等待时间
world = client.get_world()   # 获得世界
print ('crated VR world')

# actor ： 模拟其中的演员对象-比如汽车
# blueprint ： actor对应的属性（比如颜色，车型等）， 在构造蓝图blueprint中设置。 --所有的属性都包含在一个库中。
blueprint_library = world.get_blueprint_library() # 获取蓝图库
'''blueprint API : carla.BlueprintLibrary, find(id), filter(wildcard_pattern), __getitem__(pos), __len__(), __iter__()'''

# 该库允许我们通过ID查找特定的蓝图， 使用通配符对其进行过滤， 或者只是随机选择一个
# 寻找具体的蓝图
collision_sensor_bp = blueprint_library.find('sensor.other.collision')
# 随机选择车辆蓝图
vehicle_bp = random.choice(blueprint_library.filter('vehicle.bmw.*'))

bp = blueprint_library.filter('model3')[0]  # 创建一个汽车蓝图

# 可以修改蓝图的某些属性， 而其他一些属性则是只读的。例如，我们无法修改车辆的车轮数，但可以更改其颜色。
vehicles = blueprint_library.filter('vehicle.*') # filter:过滤
# 汽车
cars = [x for x in vehicles if int(x.get_attribute('number_of_wheels')) == 4]
# 自行车是两个轮子的， 将自行车挑选出来，将自行车颜色进行修改
bikes = [x for x in vehicles if int(x.get_attribute('number_of_wheels')) == 2]   # evens = [x * 2 for x in range(11)] 迭代方式
for bike in bikes:
    bike.set_attribute('color', '255,0,0')

# 可以修改的属性还带有建议修改值列表
for attr in blueprint:
    if attr.is_modifiable:
        blueprint.set_attribute(attr.id, random.choice(attr.recommended_values))

# 已经设置好蓝图后， 生成actor
transform = Transform(Location(x=230, y=195, z=40), Rotation(yaw=180)) # 生成点坐标创建
actor = world.spawn_actor(blueprint, transform)  # 这个时候会检查生成殿坐标是否会发生碰撞。 如果有碰撞，会报错。

# 为了简化查找生成位置的任务， 每个地图都提供了建议的变换列表
spawn_points = world.get_map().get_spawn_points() # 返回所有可用的collision-free points.
number_of_spawn_points = len(spawn_points)
print(number_of_spawn_points)



# 生成汽车
spawn_point = random.choice(world.get_map().get_spawn_points()) # 随机选择出生点
vehicle = world.spawn_actor(bp, spawn_point) # 随机在出生点生成汽车, 可以在后面设置自动驾驶（world.spawn_actor().set_autopilot(enabled=True)）

# 控制汽车
vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0)) # (vehicle.set_autopilot(True)自动驾驶)
'''操控汽车的全部API:
carla.VehicleControl(
    throttle = 0.0  # 油门
    steer = 0.0     # 转向
    brake = 0.0     # 刹车
    hand_brake = False  # 手刹
    reverse = False   # 倒车
    manual_gear_shift = False   # 手动换档
    gear = 0   # 档)'''
actor_list.append(vehicle)

# 调整车辆的动力性特征--调整质量，力矩图，最大RPM
vehicle.apply_physics_control(carla.VehiclePhysicsControl(max_rpm = 5000.0,
    center_of_mass = carla.Vector3D(0.0, 0.0, 0.0), torque_curve=[[0,400],[5000,400]]))
'''动力性特征的全部API:
carla.VehiclePhysicsControl(
    torque_curve,
    max_rpm,
    moi,
    damping_rate_full_throttle,
    damping_rate_zero_throttle_clutch_engaged,
    damping_rate_zero_throttle_clutch_disengaged,
    use_gear_autobox,
    gear_switch_time,
    clutch_strength,
    mass,
    drag_coefficient,
    center_of_mass,
    steering_curve, # 转向maximum steering for a specific forward speed
    wheels)'''

'''轮胎的特性API：
carla.WheelPhysicsControl(
    tire_friction,
    damping_rate,
    steer_angle,
    disable_steering)'''

'''actor 自动驾驶
vehicle.set_autopilot(True)'''


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

# 显示位置，设置位置信息
location = actor.get_location()
location.z += 10.0
actor.set_location(location)
# 显示 速度，加速度信息
print(actor.get_acceleration())
print(actor.get_velocity())

# 摧毁目标
carla.command.DestroyActor(actor)

#获得车辆的bounding box
box = vehicle.bounding_box
print(box.location)         # Location relative to the vehicle.相对于车辆的位置。
print(box.extent)           # XYZ half-box extents in meters.XYZ半框范围（以米为单位）。

# 除了车辆和传感器之外，世界上还有其他的一些参与者。 可以向全世界索取完整的列表
actor_list = world.get_actors()
# 返回的参与者列表对象具有查找，过滤和迭代参与者的功能
# Find an actor by id.
actor = actor_list.find(id)
# Print the location of all the speed limit signs in the world.
for speed_sign in actor_list.filter('traffic.speed_limit.*'):
    print(speed_sign.get_location())
'''
Traffic(红绿灯) lights with a state property to check the light's current state.
Speed limit signs(限速标志) with the speed codified in their type_id.
The Spectator actor(观众) that can be used to move the view of the simulator window.
'''
'''
get_weather() 获取

set_weather(weather_parameters) 设置

Static presets 定义好的天气。

carla.WeatherParameters.ClearNoon

carla.WeatherParameters.CloudyNoon

carla.WeatherParameters.WetNoon

carla.WeatherParameters.WetCloudyNoon

carla.WeatherParameters.MidRainyNoon

carla.WeatherParameters.HardRainNoon

carla.WeatherParameters.SoftRainNoon

carla.WeatherParameters.ClearSunset

carla.WeatherParameters.CloudySunset

carla.WeatherParameters.WetSunset

carla.WeatherParameters.WetCloudySunset

carla.WeatherParameters.MidRainSunset

carla.WeatherParameters.HardRainSunset

carla.WeatherParameters.SoftRainSunset
'''

# 地图对象
# 生成点获取
client=carla.Client('127.0.0.1',2000) #默认地址
print ('create client')
client.set_timeout(2.0)
world=client.get_world()
print ('crated VR world')

map=world.get_map()
print ('map name--',map.name)
sps=map.get_spawn_points()
print (' map first avaliable spawn points--',sps[0])

# 获取路标
# 地图API可以为我们提供最接近车辆的道路上的路标
waypoint = map.get_waypoint(vehicle.get_location()) # 车辆的位置 最近的路标获取

# 查询下一个路标
# Retrieve the closest waypoint.检索最近的航路点。
waypoint = map.get_waypoint(vehicle.get_location())

# Disable physics, in this example we're just teleporting the vehicle.禁用物理学，在此示例中，我们只是传送车辆。
vehicle.set_simulate_physics(False)

while True:
    # Find next waypoint 2 meters ahead.在前方2米处找到下一个航路点。
    waypoint = random.choice(waypoint.next(2.0))
    # Teleport the vehicle.传送车辆。
    vehicle.set_transform(waypoint.transform)

# The map object also provides methods for generating in bulk waypoints all over the map at an approximated distance between them
# 地图对象还提供了用于在地图之间以近似距离生成大量航路点的方法
waypoint_list = map.generate_waypoints(2.0)

# For routing purposes, it is also possible to retrieve a topology graph of the roads
# 出于路由目的，还可以获取道路的拓扑图
# 此方法返回一对航路点（元组）列表，对于每对航路点，第一个元素与第二个元素连接。此方法仅生成用于定义拓扑的最小路标集，对于地图中每个路段的每个车道仅生成一个路标。
# 最后，为了允许访问整个道路信息，可以将地图对象转换为OpenDrive格式，并保存到磁盘上。
waypoint_tuple_list = map.get_topology()