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
        'dt': 0.025,  # time interval between two frames
        'ego_vehicle_filter': 'vehicle.lincoln*',  # filter for defining ego vehicle
        'port': 2000,  # connection port
        'task_mode': 'Curve',  # mode of the task, [random, roundabout (only for Town03)]
        'code_mode': 'train',
        'max_time_episode': 200,  # maximum timesteps per episode
        'desired_speed': 15,  # desired speed (m/s)
        'max_ego_spawn_times': 100,  # maximum times to spawn ego vehicle
    }

    # Set gym-carla environment
    env = gym.make('carla-v0', params=params)
    # start0 = carla.Location(x=env.starts[2][0], y=env.starts[2][1], z=env.starts[2][2])
    # end0 = carla.Location(x=env.dests[2][0], y=env.dests[2][1], z=env.dests[2][2])
    # env.world.debug.draw_point(start0)
    # env.world.debug.draw_point(end0)

    # start1 = carla.Location(x=env.starts[3][0], y=env.starts[3][1], z=env.starts[3][2])
    # end1 = carla.Location(x=env.dests[3][0], y=env.dests[3][1], z=env.dests[3][2])
    # env.world.debug.draw_point(start1, color=carla.Color(0,0,255))
    # env.world.debug.draw_point(end1, color=carla.Color(0,0,255))

    # print(env.map.get_waypoint(location=start0))
    # print(env.map.get_waypoint(location=end0))
    # print(env.map.get_waypoint(location=start1))
    # print(env.map.get_waypoint(location=end1))
    obs, info = env.reset()

    # for dist in np.arange(0.1, 100.1, 1):
    #     wpt = env.map.get_waypoint(location=start).next(dist)[0]
    #     env.world.debug.draw_point(wpt.transform.location)

    # obs, r, done, info = env.step([0.0, 0.0])
    # print(obs.shape)
    # print(env.ego.get_location())
    tic = time.time()
    done = False
    ret = 0
    count = 1

    while not (done or env.isTimeOut):
        print(obs[0:4])
        print(obs[4]*2, 'deg', obs[5]*5, 'deg/s', obs[6]/10, 'm')
        print(obs[7:9]/10)
        # print(obs[9:12]/10)
        print('--------------------------------')

        tac = time.time()
        if tac - tic <= 10:
            action = [0.0, 0.]
            # throttle = np.random.rand(1) - 0.5
            # action = np.concatenate((throttle, np.random.uniform(low=-0.3, high=0.3, size=(1,))), axis=0)
        else:
            action = [0.0, 0]
        # print((obs[0]**2+obs[1]**2)**0.5)

        obs, r, done, info = env.step(action)
        count += 1
        ret += r
        print(obs[0:4])
        print(obs[4]*2, 'deg', obs[5]*5, 'deg/s', obs[6]/10, 'm')
        print(obs[7:9]/10)
        # print(obs[9:12]/10)
        print('--------------------------------')
        # env.world.debug.draw_point(start)
        # env.world.debug.draw_point(end)

        if done or env.isTimeOut:
            toc = time.time()
            print("An episode took %f s" %(toc - tic))
            print("total reward is", ret)
            print("time steps", env.time_step)
            # env.close()

            ret = 0
            # print(env.ego.get_location())
            done = False
            obs, info = env.reset()
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