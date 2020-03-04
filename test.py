#!/usr/bin/env python

# Copyright (c) 2019: Jianyu Chen (jianyuchen@berkeley.edu).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import gym
import gym_carla
import carla
import cv2
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Queue
import time

def main():
    # parameters for the gym_carla environment
    params = {
        'agent_id': 0,
        'number_of_vehicles': 0,
        'number_of_walkers': 0,
        'display_size': 256,  # screen size of bird-eye render
        'obs_size': 256,  # screen size of cv2 window
        'max_past_step': 1,  # the number of past steps to draw
        'dt': 0.1,  # time interval between two frames
        # 'discrete': False,  # whether to use discrete control space
        # 'discrete_acc': [-3.0, 0.0, 3.0],  # discrete value of accelerations
        # 'discrete_steer': [-0.2, 0.0, 0.2],  # discrete value of steering angles
        # 'continuous_accel_range': [-3.0, 3.0],  # continuous acceleration range
        # 'continuous_steer_range': [-0.3, 0.3],  # continuous steering angle range
        'ego_vehicle_filter': 'vehicle.lincoln*',  # filter for defining ego vehicle
        'port': 2000,  # connection port
        # 'town': 'Town01',  # which town to simulate
        'task_mode': 'Straight',  # mode of the task, [random, roundabout (only for Town03)]
        'code_mode': 'train',
        'max_time_episode': 1000,  # maximum timesteps per episode
        'max_waypt': 12,  # maximum number of waypoints
        'obs_range': 32,  # observation range (meter)
        'lidar_bin': 0.125,  # bin size of lidar sensor (meter)
        'd_behind': 12,  # distance behind the ego vehicle (meter)
        'out_lane_thres': 2.0,  # threshold for out of lane
        'desired_speed': 8,  # desired speed (m/s)
        'max_ego_spawn_times': 100,  # maximum times to spawn ego vehicle
    }

    # Set gym-carla environment
    procs = []
    for idx in range(2):
        procs.append(Process(target=run, args=(params, idx)))
    for p in procs:
        p.start()
        time.sleep(1)
    for p in procs:
        p.join()

def run(params, idx):
    params['agent_id'] = int(idx)
    env = gym.make('carla-v0', params=params)
    if idx == 0:
        env.__world_reset__()
    obs = env.reset()
    while True:
        if idx == 0:
            action = [1.0, 0.0, 0.0]
        else:
            action = [1.0, 0.0, 0.0]
        obs, r, done, info = env.step(action)
        try:
            cv2.imshow("camera img", obs)
            cv2.waitKey(1)
        except:
            print("no image")
            pass

        if done:
            env.reset()

if __name__ == '__main__':
    main()