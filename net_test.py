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

from Model import PolicyNet, QNet, Args
import torch
import numpy as np

def main():
    # parameters for the gym_carla environment
    params = {
        'display_size': 256,  # screen size of bird-eye render
        'obs_size': 128,  # screen size of cv2 window
        'dt': 0.1,  # time interval between two frames
        'ego_vehicle_filter': 'vehicle.lincoln*',  # filter for defining ego vehicle
        'port': 2000,  # connection port
        # 'town': 'Town01',  # which town to simulate
        'task_mode': 'Straight',  # mode of the task, [random, roundabout (only for Town03)]
        'code_mode': 'test',
        'max_time_episode': 5000,  # maximum timesteps per episode
        'desired_speed': 8,  # desired speed (m/s)
        'max_ego_spawn_times': 100,  # maximum times to spawn ego vehicle
    }

    # Set gym-carla environment
    env = gym.make('carla-v0', params=params)

    # load net
    device = torch.device('cpu')
    args = Args()
    args.NN_type
    actor = PolicyNet(args).to(device)
    actor.load_state_dict(torch.load('./policy1_500000.pkl',map_location='cpu'))

    Q_net1 = QNet(args).to(device)
    Q_net1.load_state_dict(torch.load('./Q1_500000.pkl',map_location='cpu'))

    obs, info_dict = env.reset()
    info = info_dict_to_array(info_dict)

    state_tensor = torch.FloatTensor(obs.copy()).float().to(device)
    info_tensor = torch.FloatTensor(info.copy()).float().to(device)

    # print(env.ego.get_location())
    tic = time.time()
    done = False
    ret = 0
    start = carla.Location(x=env.start[0], y=env.start[1], z=0.22)
    end = carla.Location(x=env.dest[0], y=env.dest[1], z=0.22)

    if args.NN_type == "CNN":
        state_tensor = state_tensor.permute(2, 0, 1)

    while not done:
        tac = time.time()

        u, log_prob = actor.get_action(state_tensor.unsqueeze(0), info_tensor.unsqueeze(0), True)
        u = u.squeeze(0)

        obs, r, done, info = env.step(u)

        info = info_dict_to_array(info_dict)
        state_tensor = torch.FloatTensor(obs.copy()).float().to(device)
        if args.NN_type == "CNN":
            state_tensor = state_tensor.permute(2, 0, 1)
        info_tensor = torch.FloatTensor(info.copy()).float().to(device)

        ret += r
        cv2.imshow("camera img", obs)
        cv2.waitKey(1)
        # print(info['acceleration_t'].shape)
        env.world.debug.draw_point(start)
        env.world.debug.draw_point(end)

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

def info_dict_to_array(info):
    '''
    params: dict of ego state(velocity_t, accelearation_t, dist, command)
    type: np.array
    return: array of size[10,], torch.Tensor (v, a, d, c)
    '''
    velocity_t = info['velocity_t']
    accel_t = info['acceleration_t']
    dist_t = info['dist_to_dest'].reshape((1,1))
    command_t = info['direction']

    info_vec = np.concatenate([velocity_t, accel_t, dist_t, command_t], axis=0)
    info_vec = info_vec.squeeze()

    return  info_vec


if __name__ == '__main__':
    main()