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
import pygame
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

# REACH_GOAL = 0.0
# GO_STRAIGHT = 5.0
# TURN_RIGHT = 4.0
# TURN_LEFT = 3.0
# LANE_FOLLOW = 2.0

class CarlaEnv(gym.Env):
    """An OpenAI gym wrapper for CARLA simulator."""
    def __init__(self, params):
        # parameters
        self.display_size = params['display_size']  # rendering screen size
        # self.max_past_step = params['max_past_step']
        self.number_of_vehicles = params['number_of_vehicles']
        self.number_of_walkers = params['number_of_walkers']
        self.dt = params['dt']

        self.task_mode = params['task_mode']
        self.code_mode = params['code_mode']

        self.max_time_episode = params['max_time_episode']
        # self.max_waypt = params['max_waypt']
        # self.obs_range = params['obs_range']
        # self.lidar_bin = params['lidar_bin']
        # self.d_behind = params['d_behind']
        # self.obs_size = int(self.obs_range/self.lidar_bin)
        self.obs_size = params['obs_size']
        self.out_lane_thres = params['out_lane_thres']
        self.desired_speed = params['desired_speed']
        self.max_ego_spawn_times = params['max_ego_spawn_times']
        self.route_id = 0

        # used for debugging
        self.instruction = {0.0: 'REACH_GOAL', 2.0: 'LANE_FOLLOW',
            3.0: 'TURN_LEFT',4.0: 'TURN_RIGHT',5.0: 'GO_STRAIGHT'}

        # Start and Destination
        if self.code_mode == 'train' or self.code_mode == 'eval':
            # Town01
            self.starts, self.dests = train_coordinates(self.task_mode)
        elif self.code_mode == 'test':
            pass
            # Town02
            # TODO: type in the data

        self.start = self.starts[self.route_id]
        self.dest = self.dests[self.route_id]

        # action and observation space
        self.action_space = spaces.Box(np.array([-1.0, -1.0]),
            np.array([1.0, 1.0]), dtype=np.float32)
        self.state_space = spaces.Box(low=0, high=256,
            shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8)

        # Connect to carla server and get world object
        print('connecting to Carla server...')
        client = carla.Client('localhost', params['port'])
        client.set_timeout(10.0)
        if self.code_mode == 'train' or self.code_mode == 'eval':
            self.world = client.load_world('Town01')
            self._planner = Planner('Town01')
        elif self.code_mode == 'test':
            self.world = client.load_world('Town02')
            self._planner = Planner('Town02')
        print('Carla server connected!')

        # Set weather
        self.world.set_weather(carla.WeatherParameters.ClearNoon)

        # Get spawn points
        self.vehicle_spawn_points = list(self.world.get_map().get_spawn_points())
        self.walker_spawn_points = []
        for i in range(self.number_of_walkers):
            spawn_point = carla.Transform()
            loc = self.world.get_random_location_from_navigation()
            if (loc != None):
                spawn_point.location = loc
                self.walker_spawn_points.append(spawn_point)

        # Create the ego vehicle blueprint
        self.ego_bp = self._create_vehicle_bluepprint(params['ego_vehicle_filter'], color='49,8,8')

        # Collision sensor
        self.collision_hist = [] # The collision history
        self.collision_hist_l = 1 # collision history length
        self.collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')

        # Add the bp of camera sensor
        self.camera_img = np.zeros((self.obs_size, self.obs_size, 3), dtype=np.uint8)
        self.camera_trans = carla.Transform(carla.Location(x=0.8, z=1.7))
        self.camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        # Modify the attributes of the blueprint to set image resolution and field of view.
        self.camera_bp.set_attribute('image_size_x', str(self.obs_size))
        self.camera_bp.set_attribute('image_size_y', str(self.obs_size))
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


    def step(self, action):
        # Assign acc/steer/brake to action signal
        throttle_or_brake, steer = action[0], action[1]
        if throttle_or_brake >= 0:
            throttle = throttle_or_brake
            brake = 0
        else:
            throttle = 0
            brake = -throttle_or_brake

        # Apply control
        act = carla.VehicleControl(throttle=float(throttle), steer=float(steer), brake=float(brake))
        self.ego.apply_control(act)

        self.world.tick()

        # Append actors polygon list
        # vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
        # self.vehicle_polygons.append(vehicle_poly_dict)
        # while len(self.vehicle_polygons) > self.max_past_step:
        #     self.vehicle_polygons.pop(0)
        # walker_poly_dict = self._get_actor_polygons('walker.*')
        # self.walker_polygons.append(walker_poly_dict)
        # while len(self.walker_polygons) > self.max_past_step:
        #     self.walker_polygons.pop(0)

        # Route Planner
        directions = self._get_directions(self.ego.get_transform(), self.dest)
        self.last_direction = directions
        print("command is %s" % self.instruction[self.last_direction])

        # calculate reward
        ego_x, ego_y = self._get_ego_pos()
        dest_x, dest_y = self.dest[0], self.dest[1]
        self.new_dist = np.linalg.norm((ego_x-dest_x, ego_y-dest_y))

        isDone = self._terminal()
        current_reward = self._get_reward()
        print("reward of current state:", current_reward)

        # Update State Info (Necessary?)
        velocity = self.ego.get_velocity()
        accel = self.ego.get_acceleration()
        self.v_t = np.array([[velocity.x], [velocity.y]])
        self.a_t = np.array([[accel.x], [accel.y]])

        self.state_info['dist_to_dest'] = self.new_dist
        self.state_info['direction'] = command2Vector(self.last_direction)
        self.state_info['velocity_t'] = self.v_t
        self.state_info['acceleration_t'] = self.a_t

        # Update timesteps
        self.time_step += 1
        self.total_step += 1
        print("time step %d" % self.time_step)
        # speed = self.ego.get_velocity()
        # print(speed.x, speed.y, speed.z)

        return (self._get_obs(), current_reward, isDone, copy.deepcopy(self.state_info))

    def reset(self):
        # Clear sensor objects
        try:
            self.camera_sensor.destroy()
            self.collision_sensor.destroy()
            self.lane_sensor.destroy()
        except:
            print("Sensor not initialized")
        self.camera_sensor = None
        self.collision_sensor = None
        self.lane_sensor = None
        
        # Delete sensors, vehicles and walkers
        self._clear_all_actors(['sensor.other.collision', 'sensor.camera.rgb', \
            'vehicle.*', 'controller.ai.walker', 'walker.*', \
            'sensor.other.lane_invasion'])

        # Disable sync mode
        self._set_synchronous_mode(False)

        # Spawn surrounding vehicles
        random.shuffle(self.vehicle_spawn_points)
        count = self.number_of_vehicles
        if count > 0:
            for spawn_point in self.vehicle_spawn_points:
                if self._try_spawn_random_vehicle_at(spawn_point, number_of_wheels=[4]):
                    count -= 1
                if count <= 0:
                    break
        while count > 0:
            if self._try_spawn_random_vehicle_at(random.choice(self.vehicle_spawn_points), number_of_wheels=[4]):
                count -= 1

        # Spawn pedestrians
        random.shuffle(self.walker_spawn_points)
        count = self.number_of_walkers
        if count > 0:
            for spawn_point in self.walker_spawn_points:
                if self._try_spawn_random_walker_at(spawn_point):
                    count -= 1
                if count <= 0:
                    break
        while count > 0:
            if self._try_spawn_random_walker_at(random.choice(self.walker_spawn_points)):
                count -= 1

        # Get actors polygon list
        self.vehicle_polygons = []
        vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
        self.vehicle_polygons.append(vehicle_poly_dict)
        self.walker_polygons = []
        walker_poly_dict = self._get_actor_polygons('walker.*')
        self.walker_polygons.append(walker_poly_dict)

        # Spawn the ego vehicle
        ego_spawn_times = 0
        while True:
            # print(ego_spawn_times)
            if ego_spawn_times > self.max_ego_spawn_times:
                self.reset()

            if self.task_mode == 'Straight' or self.task_mode == 'One_curve':
                # transform = random.choice(self.starts)  # formal
                transform = self._set_carla_transform(self.start)
            if self._try_spawn_ego_vehicle_at(transform):
                break
            else:
                ego_spawn_times += 1
                time.sleep(0.1)
        print("Ego car spawn Success!")

        # Add collision sensor
        self.collision_sensor = self.world.spawn_actor(self.collision_bp, carla.Transform(), attach_to=self.ego)
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
        self.camera_sensor.listen(lambda data: get_camera_img(data))
        def get_camera_img(data):
            array = np.frombuffer(data.raw_data, dtype = np.dtype("uint8"))
            array = np.reshape(array, (data.height, data.width, 4))
            array = array[:, :, :3]
            # array = array[:, :, ::-1]
            self.camera_img = array

        # Add lane invasion sensor
        self.lane_sensor = self.world.spawn_actor(self.lane_bp, carla.Transform(), attach_to=self.ego)
        self.lane_sensor.listen(lambda event: get_lane_invasion(event))
        def get_lane_invasion(event):
            self.lane_invasion_hist = event.crossed_lane_markings
            print("length of lane invasion: %d" % len(self.lane_invasion_hist))
        self.lane_invasion_hist = []

        # Update timesteps
        self.time_step=0
        self.reset_step+=1

        # Enable sync mode
        self.settings.synchronous_mode = True
        self.world.apply_settings(self.settings)

        # Route teller
        # TODO: Decide how to use route teller
        directions = self._get_directions(self.ego.get_transform(), self.dest)
        self.last_direction = directions
        print("command is %s" % self.instruction[self.last_direction])

        ego_x, ego_y = self._get_ego_pos()
        dest_x, dest_y = self.dest[0], self.dest[1]
        self.last_dist = np.linalg.norm((ego_x-dest_x, ego_y-dest_y))

        velocity = self.ego.get_velocity()
        accel = self.ego.get_acceleration()
        self.v_t = np.array([[velocity.x], [velocity.y]])
        self.a_t = np.array([[accel.x], [accel.y]])

        self.state_info['velocity_t'] = self.v_t
        self.state_info['acceleration_t'] = self.a_t
        self.state_info['dist_to_dest'] = self.last_dist
        self.state_info['direction'] = command2Vector(self.last_direction)

        # End State variable initialized
        self.isCollided = False
        self.isTimeOut = False
        self.isSuccess = False
        self.isOutOfLane = False

        return self._get_obs(), copy.deepcopy(self.state_info)


    def render(self, mode='human'):
        pass

    def close(self):
        self.ego.destroy()

    def _terminal(self):
        """Calculate whether to terminate the current episode."""
        # Get ego state
        ego_x, ego_y = self._get_ego_pos()

        # If collides
        if len(self.collision_hist) > 0:
            print("Collision happened! Episode Done.")
            self.isCollided = True
            return True

        # If reach maximum timestep
        if self.time_step > self.max_time_episode:
            print("Time out! Episode Done.")
            self.isTimeOut = True
            return True

        # If at destination
        dest = self.dest
        if np.sqrt((ego_x-dest[0])**2+(ego_y-dest[1])**2) < 2.0:
            print("Get destination! Episode Done.")
            self.isSuccess = True
            return True

        # If out of lane
        if len(self.lane_invasion_hist) > 0:
            print("lane invasion happened! Episode Done.")
            self.isOutOfLane = True
            return True

        return False

    def _clear_all_actors(self, actor_filters):
        """Clear specific actors."""
        for actor_filter in actor_filters:
            for actor in self.world.get_actors().filter(actor_filter):
                if actor.is_alive:
                    if actor.type_id == 'controller.ai.walker':
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

    def _get_actor_polygons(self, filt):
        """Get the bounding box polygon of actors.

        Args:
            filt: the filter indicating what type of actors we'll look at.

        Returns:
            actor_poly_dict: a dictionary containing the bounding boxes of specific actors.
        """
        actor_poly_dict={}
        for actor in self.world.get_actors().filter(filt):
            # Get x, y and yaw of the actor
            trans=actor.get_transform()
            x=trans.location.x
            y=trans.location.y
            yaw=trans.rotation.yaw/180*np.pi
            # Get length and width
            bb=actor.bounding_box
            l=bb.extent.x
            w=bb.extent.y
            # Get bounding box polygon in the actor's local coordinate
            poly_local=np.array([[l,w],[l,-w],[-l,-w],[-l,w]]).transpose()
            # Get rotation matrix to transform to global coordinate
            R=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
            # Get global bounding box polygon
            poly=np.matmul(R,poly_local).transpose()+np.repeat([[x,y]],4,axis=0)
            actor_poly_dict[actor.id]=poly
        return actor_poly_dict

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
        vehicle = None

        if self.number_of_vehicles == 0 and self.number_of_walkers == 0:
            print("no dynamic")
            vehicle = self.world.spawn_actor(self.ego_bp, transform)
        else:
            for idx, poly in self.vehicle_polygons[0].items():
                poly_center = np.mean(poly, axis=0)
                ego_center = np.array([transform.location.x, transform.location.y])
                dis = np.linalg.norm(poly_center - ego_center)
                print(dis)
                if dis > 8:
                    vehicle = self.world.try_spawn_actor(self.ego_bp, transform)
                    break
                else:
                    return False

        if vehicle is not None:
            self.ego=vehicle
            return True
        return False

    def _try_spawn_random_vehicle_at(self, transform, number_of_wheels=[4]):
        """Try to spawn a surrounding vehicle at specific transform with random bluprint.

        Args:
            transform: the carla transform object.

        Returns:
            Bool indicating whether the spawn is successful.
        """
        blueprint = self._create_vehicle_bluepprint('vehicle.*', number_of_wheels=number_of_wheels)
        blueprint.set_attribute('role_name', 'autopilot')
        vehicle = self.world.try_spawn_actor(blueprint, transform)
        if vehicle is not None:
            vehicle.set_autopilot()
            return True
        return False

    def _try_spawn_random_walker_at(self, transform):
        """Try to spawn a walker at specific transform with random bluprint.

        Args:
            transform: the carla transform object.

        Returns:
            Bool indicating whether the spawn is successful.
        """
        walker_bp = random.choice(self.world.get_blueprint_library().filter('walker.*'))
        # set as not invencible
        if walker_bp.has_attribute('is_invincible'):
            walker_bp.set_attribute('is_invincible', 'false')
        walker_actor = self.world.try_spawn_actor(walker_bp, transform)

        if walker_actor is not None:
            walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
            walker_controller_actor = self.world.spawn_actor(walker_controller_bp, carla.Transform(), walker_actor)
            # start walker
            walker_controller_actor.start()
            # set walk to random point
            walker_controller_actor.go_to_location(self.world.get_random_location_from_navigation())
            # random max speed
            walker_controller_actor.set_max_speed(1 + random.random())    # max speed between 1 and 2 (default is 1.4 m/s)
            return True
        return False

    def _get_obs(self):
        current_obs = self.camera_img.copy()
        # cv2.imshow("camera img", current_obs)
        # cv2.waitKey(1)
        # self.world.tick()
        return self.camera_img

    def _get_reward(self):
        """
        calculate the reward of current state
        """
        # end state
        # reward for collision
        r_collision = 0.0
        if len(self.collision_hist) > 0:
            r_collision = -300.0

        # reward for out of lane
        r_out = 0.0
        if len(self.lane_invasion_hist) > 0:
            r_out = -300.0
        
        # reward for time out
        # reward for steering
        r_steer = 0.0
        if self.instruction[self.last_direction] == 'GO_STRATGHIT' or \
                self.instruction[self.last_direction] == 'LANE_FOLLOW':
            if abs(self.ego.get_control().steer) >= 0.2:
                r_steer = -1.0
        elif self.instruction[self.last_direction] == 'TURN_LEFT':
            if self.ego.get_control().steer > 0:
                r_steer = -1.0
        elif self.instruction[self.last_direction] == 'TURN_RIGHT':
            if self.ego.get_control().steer < 0:
                r_steer = -1.0

        # reward for speed tracking
        r_speed = 0.0
        v = self.ego.get_velocity()
        speed = np.sqrt(v.x**2 + v.y**2)
        if abs(speed - self.desired_speed) >= 3:
            r_speed = -1.0

        # reward for getting dest
        r_dist = 0.0
        last_dist = self.state_info['dist_to_dest']
        delta_dist = last_dist - self.new_dist
        if delta_dist <= 1e-2:
            r_dist = -1.0

        r_success = 0.0
        if self.isSuccess:
            r_success = 100

        return r_collision + r_out + r_steer + r_speed + r_dist + r_success


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
                start_point.rotation.roll, start_point.rotation.pitch, current_point.rotation.yaw], [
                end_point[0], end_point[1], 0.22], [
                end_point[3], end_point[4], end_point[5]])
