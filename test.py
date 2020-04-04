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
        'display_size': 256,  # screen size of bird-eye render
        'obs_size': (160, 100),  # screen size of cv2 window  x@y = width @ height
        'dt': 0.1,  # time interval between two frames
        'ego_vehicle_filter': 'vehicle.lincoln*',  # filter for defining ego vehicle
        'port': 2000,  # connection port
        'task_mode': 'One_curve',  # mode of the task, [random, roundabout (only for Town03)]
        'code_mode': 'test',
        'max_time_episode': 5000,  # maximum timesteps per episode
        'desired_speed': 8,  # desired speed (m/s)
        'max_ego_spawn_times': 100,  # maximum times to spawn ego vehicle
    }

    # Set gym-carla environment
    env = gym.make('carla-v0', params=params)
    obs, info = env.reset()
    # print(obs.dtype)
    # print(env.ego.get_location())
    tic = time.time()
    done = False
    ret = 0
    
    while not done:
        start = carla.Location(x=env.start[0], y=env.start[1], z=0.22)
        end = carla.Location(x=env.dest[0], y=env.dest[1], z=0.22)

        tac = time.time()

        if tac - tic <= 10:
            action = [-0, 0]
        else:
            action = [0.0, 0.00]

        obs, r, done, info = env.step(action)

        # if info['direction'][2][0]:
        #     print('left')
        # elif info['direction'][3,0]:
        #     print('right')
        # elif info['direction'][1, 0]:
        #     print('lane follow')

        ret += r
        cv2.imshow("camera img", obs)
        cv2.waitKey(1)

        # ------------ debug and test -----------------
        env.world.debug.draw_point(start)
        env.world.debug.draw_point(end)

        location = env.ego.get_location()
        nearest_wpt = env.map.get_waypoint(location)
        # for dist in range(1, 11):
        list_of_next_wpts = nearest_wpt.next(3)
        # print(len(list_of_next_wpts))
        for wpt in list_of_next_wpts:
            env.world.debug.draw_point(wpt.transform.location, life_time = 1)
        # wpt_location = nearest_wpt.transform.location
        # wpt_location.z = 2
        # env.world.debug.draw_point(wpt_location, life_time = 1)

        # -------------- end of episode ---------------
        if done:
            toc = time.time()
            print("An episode took %f s" %(toc - tic))
            print("total reward is", ret)
            print("time steps", env.time_step)
            env.close()
            env.reset()
            ret = 0
            # print(env.ego.get_location())
            done = False
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