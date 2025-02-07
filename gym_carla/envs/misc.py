#!/usr/bin/env python

# Copyright (c) 2020: Dongjie yu (yudongjie.moon@foxmail.com)
#
# This file is modified from <https://github.com/cjy1992/gym-carla>:
# Copyright (c) 2019:
# authors: Jianyu Chen (jianyuchen@berkeley.edu).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import math
import numpy as np
import carla

def command2Vector(command):
    """
    Convert command(scalar) to vector to be used in FC-net
    param: command(1, float)
        REACH_GOAL = 0.0
        GO_STRAIGHT = 5.0
        TURN_RIGHT = 4.0
        TURN_LEFT = 3.0
        LANE_FOLLOW = 2.0
    return: command vector(np.array, 5*1) [1 0 0 0 0]
        0-REACH_GOAL
        1-LANE_FOLLOW
        2-TURN_LEFT
        3-TURN_RIGHT
        4-GO_STRAIGHT
    """
    command_vec = np.zeros((5,1))
    REACH_GOAL = 0.0
    GO_STRAIGHT = 5.0
    TURN_RIGHT = 4.0
    TURN_LEFT = 3.0
    LANE_FOLLOW = 2.0
    if command == REACH_GOAL:
        command_vec[0] = 1.0
    elif command > 1 and command < 6:
        command_vec[int(command) - 1] = 1.0
    else:
        raise ValueError("Command Value out of bound!")

    return command_vec

if __name__ == '__main__':
    print(command2Vector(4.0))
    print(command2Vector(5.0))