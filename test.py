#!/usr/bin/env python

# Copyright (c) 2019: Jianyu Chen (jianyuchen@berkeley.edu).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import gym
import gym_carla
import carla
import cv2
import time

def main():
    # parameters for the gym_carla environment
    params = {
        'number_of_vehicles': 0,
        'number_of_walkers': 0,
        'display_size': 256,  # screen size of bird-eye render
        'obs_size': 256,  # screen size of cv2 window
        # 'max_past_step': 1,  # the number of past steps to draw
        'dt': 0.1,  # time interval between two frames
        # 'discrete': False,  # whether to use discrete control space
        # 'discrete_acc': [-3.0, 0.0, 3.0],  # discrete value of accelerations
        # 'discrete_steer': [-0.2, 0.0, 0.2],  # discrete value of steering angles
        # 'continuous_accel_range': [-3.0, 3.0],  # continuous acceleration range
        # 'continuous_steer_range': [-0.3, 0.3],  # continuous steering angle range
        'ego_vehicle_filter': 'vehicle.lincoln*',  # filter for defining ego vehicle
        'port': 2000,  # connection port
        # 'town': 'Town01',  # which town to simulate
        'task_mode': 'One_curve',  # mode of the task, [random, roundabout (only for Town03)]
        'code_mode': 'train',
        'max_time_episode': 5000,  # maximum timesteps per episode
        'max_waypt': 12,  # maximum number of waypoints
        'obs_range': 32,  # observation range (meter)
        'lidar_bin': 0.125,  # bin size of lidar sensor (meter)
        'd_behind': 12,  # distance behind the ego vehicle (meter)
        'out_lane_thres': 2.0,  # threshold for out of lane
        'desired_speed': 8,  # desired speed (m/s)
        'max_ego_spawn_times': 100,  # maximum times to spawn ego vehicle
    }

    # Set gym-carla environment
    env = gym.make('carla-v0', params=params)
    obs = env.reset()
    tic = time.time()
    ret = 0
    start = carla.Location(x=env.start[0], y=env.start[1], z=0.22)
    end = carla.Location(x=env.dest[0], y=env.dest[1], z=0.22)

    while True:
        # tac = time.time()
        # if tac - tic <= 10:
        #     action = [0.0, 0.00, 0.0]
        # else:
        action = [0.8, 0.00, 0.0]
        obs, r, done, info = env.step(action)
        ret += r
        cv2.imshow("camera img", obs)
        cv2.waitKey(1)

        env.world.debug.draw_point(start)
        env.world.debug.draw_point(end)

        if done:
            toc = time.time()

            print("An episode took %f s" %(toc - tic))
            print("total reward is", ret)
            env.close()
            break

    # turn left
    # obs = env.reset()
    # tic = time.time()

    # start = carla.Location(x=env.start[0], y=env.start[1], z=0.22)
    # end = carla.Location(x=env.dest[0], y=env.dest[1], z=0.22)

    # while True:
    #     action = [0.5, 0.0, 0.0]
    #     obs, r, done, info = env.step(action)
    #     ret += r
    #     cv2.imshow("camera img", obs)
    #     cv2.waitKey(1)
    #     env.world.debug.draw_point(start)
    #     env.world.debug.draw_point(end)

    #     if done:
    #         toc = time.time()

    #         print("An episode took %f s" %(toc - tic))
    #         print("total reward is", ret)
    #         env.close()
    #         break

    # # Stay still
    # obs = env.reset()
    # tic = time.time()
    # while True:
    #     action = [0.0, 0.0, 0.0]
    #     obs, r, done, info = env.step(action)

    #     cv2.imshow("camera img", obs)
    #     cv2.waitKey(1)

    #     if done:
    #         toc = time.time()

    #         print("An episode took %f s" %(toc - tic))
    #         env.close()
    #         break



if __name__ == '__main__':
    main()