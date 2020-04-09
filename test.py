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
import numpy as np

def main():
    # parameters for the gym_carla environment
    params = {
        'display_size': 256,  # screen size of bird-eye render
        'obs_size': (160, 100),  # screen size of cv2 window  x@y = width @ height
        'dt': 0.1,  # time interval between two frames
        'ego_vehicle_filter': 'vehicle.lincoln*',  # filter for defining ego vehicle
        'port': 2000,  # connection port
        'task_mode': 'Straight',  # mode of the task, [random, roundabout (only for Town03)]
        'code_mode': 'test',
        'max_time_episode': 5000,  # maximum timesteps per episode
        'desired_speed': 8,  # desired speed (m/s)
        'max_ego_spawn_times': 100,  # maximum times to spawn ego vehicle
    }

    # Set gym-carla environment
    env = gym.make('carla-v0', params=params)
    obs, info = env.reset()
    # obs, r, done, info = env.step([0.0, 0.0])
    # print(obs.shape)
    # print(env.ego.get_location())
    tic = time.time()
    done = False
    ret = 0
    start = carla.Location(x=env.start[0], y=env.start[1], z=0.22)
    end = carla.Location(x=env.dest[0], y=env.dest[1], z=0.22)

    while not done:
        tac = time.time()
        if tac - tic <= 10:
            action = [0, 1]
            # throttle = np.random.rand(1) - 0.5
            # action = np.concatenate((throttle, np.random.uniform(low=-0.3, high=0.3, size=(1,))), axis=0)
        else:
            action = [0.0, 0.00]
        print(obs[6])
        obs, r, done, info = env.step(action)

        # print('delta', info['delta_yaw_t'], 'angular_speed', info['dyaw_dt_t'])
        # print(info['delta_yaw_t'], info['dyaw_dt_t'])
        # print(np.max(obs), np.min(obs))
        ret += r

        env.world.debug.draw_point(start)
        env.world.debug.draw_point(end)

        if done:
            toc = time.time()
            print("An episode took %f s" %(toc - tic))
            print("total reward is", ret)
            print("time steps", env.time_step)
            env.close()
            # env.reset()
            ret = 0
            # print(env.ego.get_location())
            # done = False
            # break

    # turn left
    # obs = env.reset()
    # tic = time.time()

    # start = carla.Location(x=env.start[0], y=env.start[1], z=0.22)
    # end = carla.Location(x=env.dest[0], y=env.dest[1], z=0.22)

    # while True:
    #     action = [0.8, 0.0]
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
    #         # env.close()
    #         break

    # # # Stay still
    # obs = env.reset()
    # tic = time.time()
    # while True:
    #     action = [0.8, 0.0]
    #     obs, r, done, info = env.step(action)

    #     cv2.imshow("camera img", obs)
    #     cv2.waitKey(1)

    #     if done:
    #         toc = time.time()

    #         print("An episode took %f s" %(toc - tic))
    #         # env.close()
    #         break



if __name__ == '__main__':
    main()