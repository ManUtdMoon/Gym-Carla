#!/usr/bin/env python

# Copyright (c) 2020: Dongjie yu (yudongjie.moon@foxmail.com)
# This file is modified from <https://github.com/cjy1992/gym-carla>:
# Copyright (c) 2019:
# author: Jianyu Chen (jianyuchen@berkeley.edu)

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

from __future__ import division

import copy
import numpy as np
import random
import time
from collections import deque
# from skimage.transform import resize

import gym
from gym import spaces
from gym.utils import seeding
import carla
import cv2

from .coordinates import train_coordinates
from .planner.planner import Planner
from .misc import *
from .carla_logger import *

# REACH_GOAL = 0.0
# GO_STRAIGHT = 5.0
# TURN_RIGHT = 4.0
# TURN_LEFT = 3.0
# LANE_FOLLOW = 2.0


class CarlaEnv(gym.Env):
    """An OpenAI gym wrapper for CARLA simulator."""
    def __init__(self, params):
        self.logger = setup_carla_logger("output_logger", experiment_name=str(params['port']))
        self.logger.info("Env running in port {}".format(params['port']))
        # parameters
        self.dt = params['dt']
        self.port = params['port']
        self.task_mode = params['task_mode']
        self.code_mode = params['code_mode']
        self.max_time_episode = params['max_time_episode']
        self.obs_size = params['obs_size']
        self.desired_speed = params['desired_speed']
        self.max_ego_spawn_times = params['max_ego_spawn_times']
        self.route_id = 0

        # used for debugging
        self.instruction = {0.0: 'REACH_GOAL', 2.0: 'LANE_FOLLOW',
            3.0: 'TURN_LEFT',4.0: 'TURN_RIGHT',5.0: 'GO_STRAIGHT'}

        # Start and Destination
        self.starts, self.dests = train_coordinates(self.task_mode)
        self.start = self.starts[self.route_id]
        self.dest = self.dests[self.route_id]

        # action and observation space
        self.action_space = spaces.Box(np.array([-1.0, -1.0]),
            np.array([1.0, 1.0]), dtype=np.float32)
        self.state_space = spaces.Box(low=0.0, high=1.0,
            shape=(self.obs_size[1], self.obs_size[0], 3), dtype=np.float32)

        # Connect to carla server and get world object
        # print('connecting to Carla server...')
        self._make_carla_client('localhost', self.port)

        # Create the ego vehicle blueprint
        self.ego_bp = self._create_vehicle_bluepprint(params['ego_vehicle_filter'], color='49,8,8')

        # Collision sensor
        self.collision_hist = [] # The collision history
        self.collision_hist_l = 1 # collision history length
        self.collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')

        # Add the bp of camera sensor
        self.camera_img = np.zeros((self.obs_size[0], self.obs_size[1], 3), dtype=np.uint8)
        self.camera_trans = carla.Transform(carla.Location(x=1.8, z=1.5), carla.Rotation(pitch=-10))
        self.camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        # Modify the attributes of the blueprint to set image resolution and field of view.
        self.camera_bp.set_attribute('image_size_x', str(self.obs_size[0]))
        self.camera_bp.set_attribute('image_size_y', str(self.obs_size[1]))
        self.camera_bp.set_attribute('fov', '110')
        # Set the time in seconds between sensor captures
        self.camera_bp.set_attribute('sensor_tick', '0.02')

        # Lane Invasion Sensor
        self.lane_bp = self.world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.lane_invasion_hist = []

        # Set fixed simulation step for synchronous mode
        self.settings = self.world.get_settings()
        self.settings.fixed_delta_seconds = self.dt

        # Record the time of total steps and resetting steps
        self.reset_step = 0
        self.total_step = 0

        # A dict used for storing state data
        self.state_info = {}

        # A list stores the ids for each episode
        self.actors = []


    def step(self, action):
        # Assign acc/steer/brake to action signal
        throttle_or_brake, steer = action[0], action[1]
        if throttle_or_brake >= 0:
            throttle = throttle_or_brake
            brake = 0
        else:
            throttle = 0
            brake = -throttle_or_brake
        # print(action)
        # print(self.ego.get_velocity())
        # print(self.time_step)
        # Apply control
        act = carla.VehicleControl(throttle=float(throttle), steer=float(steer), brake=float(brake))
        self.ego.apply_control(act)

        self.world.tick()

        # Route Planner
        directions = self._get_directions(self.ego.get_transform(), self.dest)
        self.last_direction = directions
        # print("command is %s" % self.instruction[self.last_direction])

        # calculate reward
        ego_x, ego_y = self._get_ego_pos()
        self.new_dist = np.linalg.norm((ego_x-self.dest[0], ego_y-self.dest[1]))

        delta_yaw = self._get_delta_yaw()
        dyaw_dt = self.ego.get_angular_velocity().z
        isDone = self._terminal()
        current_reward = self._get_reward()
        # print("reward of current state:", current_reward)

        # Update State Info (Necessary?)
        velocity = self.ego.get_velocity()
        accel = self.ego.get_acceleration()
        self.v_t = np.array([[velocity.x], [velocity.y]])
        self.a_t = np.array([[accel.x], [accel.y]])

        self.state_info['dist_to_dest'] = self.new_dist
        self.state_info['direction'] = command2Vector(self.last_direction)
        self.state_info['velocity_t'] = self.v_t
        self.state_info['acceleration_t'] = self.a_t
        self.state_info['delta_yaw_t'] = delta_yaw
        self.state_info['dyaw_dt_t'] = dyaw_dt

        # Update timesteps
        self.time_step += 1
        self.total_step += 1
        self.last_action = self.current_action
        # print("time step %d" % self.time_step)

        return (self._get_obs(), current_reward, isDone, copy.deepcopy(self.state_info))


    def reset(self):
        while True:
            try:
                # Clear sensor objects
                self.camera_sensor = None
                self.collision_sensor = None
                self.lane_sensor = None

                # Delete sensors, vehicles and walkers
                while self.actors:
                    (self.actors.pop()).destroy()

                # Disable sync mode
                self._set_synchronous_mode(False)

                # Spawn the ego vehicle at a random position between start and dest
                ego_spawn_times = 0
                while True:
                    if ego_spawn_times > self.max_ego_spawn_times:
                        self.reset()
                    if self.task_mode == 'Straight':
                        # Code_mode == test or eval, spawn at start
                        transform = self._set_carla_transform(self.start)
                        # Code_mode == train, spwan randomly between start and destination
                        if self.code_mode == 'train':
                            transform.location = self._get_random_position_between(self.start, self.dest)
                    if self._try_spawn_ego_vehicle_at(transform):
                        break
                    else:
                        ego_spawn_times += 1
                        time.sleep(0.1)
                # print("Ego car spawn Success!")
                # self.logger.info("Ego car spawn Success!")

                # Add collision sensor
                self.collision_sensor = self.world.try_spawn_actor(self.collision_bp, carla.Transform(), attach_to=self.ego)
                self.actors.append(self.collision_sensor)
                self.collision_sensor.listen(lambda event: get_collision_hist(event))
                def get_collision_hist(event):
                    impulse = event.normal_impulse
                    intensity = np.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
                    self.collision_hist.append(intensity)
                    if len(self.collision_hist)>self.collision_hist_l:
                        self.collision_hist.pop(0)
                self.collision_hist = []

                # Add camera sensor
                self.camera_sensor = self.world.spawn_actor(self.camera_bp, self.camera_trans, attach_to=self.ego)

                self.actors.append(self.camera_sensor)
                self.camera_sensor.listen(lambda data: get_camera_img(data))
                def get_camera_img(data):
                    array = np.frombuffer(data.raw_data, dtype = np.dtype("uint8"))
                    array = np.reshape(array, (data.height, data.width, 4))
                    array = array[:, :, :3]
                    self.camera_img = array

                # Add lane invasion sensor
                self.lane_sensor = self.world.spawn_actor(self.lane_bp, carla.Transform(), attach_to=self.ego)
                self.actors.append(self.lane_sensor)
                self.lane_sensor.listen(lambda event: get_lane_invasion(event))
                def get_lane_invasion(event):
                    self.lane_invasion_hist = event.crossed_lane_markings
                    # print("length of lane invasion: %d" % len(self.lane_invasion_hist))
                self.lane_invasion_hist = []

                # Update timesteps
                self.time_step=0
                self.reset_step+=1

                # Enable sync mode
                self.settings.synchronous_mode = True
                self.world.apply_settings(self.settings)

                # Route teller
                directions = self._get_directions(self.ego.get_transform(), self.dest)
                self.last_direction = directions
                # print("command is %s" % self.instruction[self.last_direction])

                ego_x, ego_y = self._get_ego_pos()
                dest_x, dest_y = self.dest[0], self.dest[1]
                self.last_dist = np.linalg.norm((ego_x-dest_x, ego_y-dest_y))

                # Set the initial speed to desired speed
                yaw = (self.ego.get_transform().rotation.yaw) * np.pi / 180.0
                init_speed = carla.Vector3D(x=self.desired_speed * np.cos(yaw),
                                            y=self.desired_speed * np.sin(yaw))
                self.ego.set_velocity(init_speed)

                # Get dynamics infomation
                velocity = self.ego.get_velocity()
                accel = self.ego.get_acceleration()
                self.v_t = np.array([[velocity.x], [velocity.y]])
                self.a_t = np.array([[accel.x], [accel.y]])
                delta_yaw = self._get_delta_yaw()
                dyaw_dt = self.ego.get_angular_velocity().z

                # record the action of last time step
                self.last_action = np.array([0.0, 0.0])
                self.current_action = self.last_action.copy()

                self.state_info['velocity_t'] = self.v_t
                self.state_info['acceleration_t'] = self.a_t
                self.state_info['dist_to_dest'] = self.last_dist
                self.state_info['direction'] = command2Vector(self.last_direction)
                self.state_info['delta_yaw_t'] = delta_yaw
                self.state_info['dyaw_dt_t'] = dyaw_dt

                # End State variable initialized
                self.isCollided = False
                self.isTimeOut = False
                self.isSuccess = False
                self.isOutOfLane = False
                self.isSpecialSpeed = False

                return self._get_obs(), copy.deepcopy(self.state_info)
            except:
                self.logger.error("Env reset() error")
                time.sleep(2)
                self._make_carla_client('localhost', self.port)


    def render(self, mode='human'):
        pass


    def close(self):
        while self.actors:
            (self.actors.pop()).destroy()


    def _terminal(self):
        """Calculate whether to terminate the current episode."""
        # Get ego state
        ego_x, ego_y = self._get_ego_pos()

        # If collides
        if len(self.collision_hist) > 0:
            # print("Collision happened! Episode Done.")
            self.logger.debug('Collision happened! Episode Done.')
            self.isCollided = True
            return True

        # If reach maximum timestep
        if self.time_step > self.max_time_episode:
            # print("Time out! Episode Done.")
            self.logger.debug('Time out! Episode Done.')
            self.isTimeOut = True
            return True

        # If at destination
        dest = self.dest
        if np.sqrt((ego_x-dest[0])**2+(ego_y-dest[1])**2) < 2.0:
            # print("Get destination! Episode Done.")
            self.logger.debug('Get destination! Episode Done.')
            self.isSuccess = True
            return True

        # If out of lane
        if len(self.lane_invasion_hist) > 0:
            # print("lane invasion happened! Episode Done.")
            self.logger.debug('Lane invasion happened! Episode Done.')
            self.isOutOfLane = True
            return True

        # If speed is special
        velocity = self.ego.get_velocity()
        v_norm = np.linalg.norm(np.array((velocity.x, velocity.y)))
        if v_norm < 3:
            self.logger.debug("Speed too slow! Episode Done.")
            self.isSpecialSpeed = True
            return True
        elif v_norm > 13:
            self.logger.debug("Speed too fast! Episode Done.")
            self.isSpecialSpeed = True
            return True

        return False


    def _clear_all_actors(self, actor_filters):
        """Clear specific actors."""
        for actor_filter in actor_filters:
            for actor in self.world.get_actors().filter(actor_filter):
                if actor.is_alive:
                    if actor.type_id == 'controller.ai.walker' or actor.type_id == 'sensor.camera.rgb' or actor.type_id == 'sensor.other.collision':
                        actor.stop()
                    actor.destroy()


    def _create_vehicle_bluepprint(self, actor_filter, color=None, number_of_wheels=[4]):
        """Create the blueprint for a specific actor type.

        Args:
            actor_filter: a string indicating the actor type, e.g, 'vehicle.lincoln*'.

        Returns:
            bp: the blueprint object of carla.
        """
        blueprints = self.world.get_blueprint_library().filter(actor_filter)
        blueprint_library = []
        for nw in number_of_wheels:
            blueprint_library = blueprint_library + [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == nw]
        bp = random.choice(blueprint_library)
        if bp.has_attribute('color'):
            if not color:
                color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)
        return bp


    def _get_ego_pos(self):
        """Get the ego vehicle pose (x, y)."""
        ego_trans = self.ego.get_transform()
        ego_x = ego_trans.location.x
        ego_y = ego_trans.location.y
        return ego_x, ego_y


    def _set_carla_transform(self, pose):
        """Get a carla tranform object given pose.

        Args:
            pose: [x, y, z, pitch, roll, yaw].

        Returns:
            transform: the carla transform object
        """
        transform = carla.Transform()
        transform.location.x = pose[0]
        transform.location.y = pose[1]
        transform.location.z = pose[2]
        transform.rotation.pitch = pose[3]
        transform.rotation.roll = pose[4]
        transform.rotation.yaw = pose[5]
        return transform


    def _set_synchronous_mode(self, synchronous = True):
        """Set whether to use the synchronous mode.
        """
        self.settings.synchronous_mode = synchronous
        self.world.apply_settings(self.settings)


    def _try_spawn_ego_vehicle_at(self, transform):
        """Try to spawn the ego vehicle at specific transform.

        Args:
            transform: the carla transform object.

        Returns:
            Bool indicating whether the spawn is successful.
        """
        vehicle = self.world.spawn_actor(self.ego_bp, transform)
        self.actors.append(vehicle)
        if vehicle is not None:
            self.ego=vehicle
            return True
        return False


    def _get_obs(self):
        current_obs = self.camera_img.copy()
        # cv2.imshow("camera img", current_obs)
        # cv2.waitKey(1)
        # self.world.tick()
        return np.float32(current_obs / 255.0)


    def _get_reward(self):
        """
        calculate the reward of current state
        """
        # end state
        # reward for done: collision/out/SpecislSPeed & Success
        r_done = 0.0
        if self.isCollided or self.isOutOfLane or self.isSpecialSpeed:
            r_done = -300.0
        if self.isSuccess:
            r_done = 300.0

        # reward for speed
        v = self.ego.get_velocity()
        speed = np.sqrt(v.x**2 + v.y**2)
        delta_speed = speed - self.desired_speed
        r_speed = 1.0 - delta_speed**2 / 25.0
        # print(r_speed)

        # reward for steer
        delta_yaw = self._get_delta_yaw()
        r_steer = -10 * (delta_yaw * np.pi / 180)**2

        # reward for action smoothness
        current_action = self.ego.get_control()
        self.current_action = np.array([current_action.throttle-current_action.brake, current_action.steer])
        r_action_smooth = -0.5 * np.linalg.norm(self.current_action - self.last_action)
        # print(self.last_action, '---->', self.current_action, r_action_smooth)

        return r_done + r_speed + r_steer + r_action_smooth


    def _get_directions(self, current_point, end_point):
        '''
        params: current position; target_position
        return: command list
        '''
        directions = self._planner.get_next_command(
            (current_point.location.x,
             current_point.location.y, 0.22),
            (current_point.rotation.roll,
             current_point.rotation.pitch,
             current_point.rotation.yaw),
            (end_point[0], end_point[1], 0.22),
            (end_point[3], end_point[4], end_point[5]))
        return directions


    def _get_shortest_path(self, start_point, end_point):

        return self._planner.get_shortest_path_distance(
            [   start_point.location.x, start_point.location.y, 0.22], [
                start_point.rotation.roll, start_point.rotation.pitch, start_point.rotation.yaw], [
                end_point[0], end_point[1], 0.22], [
                end_point[3], end_point[4], end_point[5]])


    def _make_carla_client(self, host, port):
        while True:
            try:
                self.logger.info("connecting to Carla server...")
                self.client = carla.Client(host, port)
                self.client.set_timeout(10.0)

                # Set map
                self.world = self.client.load_world('Town01')
                self._planner = Planner('Town01')
                self.map = self.world.get_map()

                # Set weather
                self.world.set_weather(carla.WeatherParameters.ClearNoon)
                self.logger.info("Carla server port {} connected!".format(port))
                break
            except Exception:
                self.logger.error('Fail to connect to carla-server...sleeping for 1')
                time.sleep(1)


    def _get_random_position_between(self, start, dest):
        """
        get a random carla position on the line between start and dest
        """
        s_x, s_y, s_z = start[0], start[1], start[2]
        d_x, d_y, d_z = dest[0], dest[1], dest[2]

        ratio = np.random.rand()
        new_x = (d_x - s_x) * ratio + s_x
        new_y = (d_y - s_y) * ratio + s_y
        new_z = (d_z - s_z) * ratio + s_z

        return carla.Location(x=new_x, y=new_y, z=new_z)


    def _get_delta_yaw(self):
        """
        calculate the delta yaw between ego and current waypoint
        """
        current_wpt = self.map.get_waypoint(location=self.ego.get_location())
        if not current_wpt:
            wpt_yaw = self.start[5] % 360
        else:
            wpt_yaw = current_wpt.transform.rotation.yaw % 360
        ego_yaw = self.ego.get_transform().rotation.yaw % 360
        delta_yaw = ego_yaw - wpt_yaw
        if 180 <= delta_yaw and delta_yaw <= 360:
            delta_yaw -= 360
        elif -360 <= delta_yaw and delta_yaw <= -180:
            delta_yaw += 360

        return delta_yaw