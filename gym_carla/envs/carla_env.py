#!/usr/bin/env python

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

class CarlaEnv(gym.Env):
    """An OpenAI gym wrapper for CARLA simulator."""
    def __init__(self, params):
        pass

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self, mode='human'):
        pass

    def close(self):
        pass
