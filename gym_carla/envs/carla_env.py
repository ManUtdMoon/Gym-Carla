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
from skimage.transform import resize

import gym
from gym import spaces
from gym.utils import seeding
import carla
import cv2

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
		self.out_lane_thres = params['out_lane_thres']
		self.desired_speed = params['desired_speed']
		self.max_ego_spawn_times = params['max_ego_spawn_times']

		# Start and Destination
		if self.code_mode == 'train' or self.code_mode == 'eval':
            # Town01
            if self.task_mode == 'Straight':
                self.starts = [[322.09, 129.35, 180], 
                           [88.13, 4.32, 90],
                           [392.47, 87.41, 90],
                           [383.18, -2.20, 180],
                           [283.67, 129.48, 180]]
                self.dests = [[119.47, 129.75, 180], 
                          [88.13, 299.92, 90], 
                          [392.47, 308.21, 90],
                          [185.55, -1.95, 180],
                          [128.94, 129.75, 180]]
            elif self.task_mode == 'One curve':
                pass
            elif self.task_mode == 'Navigation':
                pass
		elif self.code_mode == 'test':
            # Town02
            # TODO: type in the data
			# if self.task_mode == 'Straight':
            #     self.starts = [[322.09, 129.35, 180], 
            #                [88.13, 4.32, 90],
            #                [392.47, 87.41, 90],
            #                [383.18, -2.20, 180],
            #                [283.67, 129.48, 180]]
            #     self.dests = [[119.47, 129.75, 180], 
            #               [88.13, 299.92, 90], 
            #               [392.47, 308.21, 90],
            #               [185.55, -1.95, 180],
            #               [128.94, 129.75, 180]]
            # elif self.task_mode == 'One curve':
            #     pass
            # elif self.task_mode == 'Navigation':
            #     pass

        # action and observation space
        # TODO: confirm the state and actione space

        # Connect to carla server and get world object
		print('connecting to Carla server...')
		client = carla.Client('localhost', params['port'])
		client.set_timeout(10.0)
        if self.code_mode == 'train' or self.code_mode == 'eval':
		    self.world = client.load_world('Town01')
        elif self.code_mode == 'test':
            self.world = client.load_world('Town02')
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

        # Add camera sensor
        # TODO:

        # Set fixed simulation step for synchronous mode
		self.settings = self.world.get_settings()
		self.settings.fixed_delta_seconds = self.dt

        # Record the time of total steps and resetting steps
		self.reset_step = 0
		self.total_step = 0

        # Initialize the render
        

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self, mode='human'):
        pass

    def close(self):
        pass

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
    
    def _set_carla_transform(self, pose):
		"""Get a carla tranform object given pose.

		Args:
			pose: [x, y, yaw].

		Returns:
			transform: the carla transform object
		"""
		transform = carla.Transform()
		transform.location.x = pose[0]
		transform.location.y = pose[1]
		transform.rotation.yaw = pose[2]
		return transform

    def _set_synchronous_mode(self, synchronous = True):
		"""Set whether to use the synchronous mode.
		"""
		self.settings.synchronous_mode = synchronous
		self.world.apply_settings(self.settings)