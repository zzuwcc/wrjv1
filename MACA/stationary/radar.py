
import numpy as np
import random

from MACA.stationary.stationary_type import STATIONARY_TYPE
from MACA.stationary.base import BaseStationary

class Radar(BaseStationary):
    def __init__(self, args):
        super(Radar, self).__init__(args)
        self.type = STATIONARY_TYPE['radar']

    def initialize(self, base_info):
        BaseStationary.initialize(self, base_info)
        # self.top = base_info['top']
        # self.bottom = base_info['bottom']
        # self.current = base_info['current']
        self.flag = base_info['flag']
        self.rad = base_info['rad']
        self.speed = base_info['speed']
        self.k1 = base_info['k1']
        self.b1 = base_info['b1']
        self.k2 = base_info['k2']
        self.b2 = base_info['b2']
        self.pivot = base_info['pivot']
        self.period = base_info['period']
        self.left = base_info['left']
        self.t = 0
        self.dt = 1.25
        # self.base_pos = base_info['base_pos']

    def step(self, direct, be_attacked, attack, attack_bias=1.0):
        # 位置更新
        bias_x = np.cos(direct) * self.dt * self.speed
        bias_y = np.sin(direct) * self.dt * self.speed

        # self.pos[0] += bias_x
        # self.pos[1] += bias_y
        if self.flag == 1:
            self.t += self.dt * self.speed / 40
            self.t = self.t % self.period
            x = self.t
            if x > self.period / 2:
                x = self.period - x
            x += self.left
            if x < self.pivot:
                y = self.k1 * x + self.b1
            else:
                y = self.k2 * x + self.b2
            self.pos[0] = x
            self.pos[1] = y

        # 位置限制
        self.pos[0] = np.clip(self.pos[0], 0, self.map_size[0])
        self.pos[1] = np.clip(self.pos[1], 0, self.map_size[1])
        # last action recorded
        self.last_action = [direct, attack]
    

    def script_action(self, enemies):
        if self.alive:
            # if self.pos[1] <= self.top:
            #     self.flag *= -1
            #     # print(self.pos[0], self.pos[1])
            # if self.pos[1] >= self.bottom:
            #     self.flag *= -1
            #     # print(self.pos[0], self.pos[1])
            
            # direct = self.flag * np.pi / 2 + self.rad
            # if self.top == 75:
            #     print(self.flag, direct, np.sin(direct), np.cos(direct))

            direct = 0
            # calc attack
            attack = 0 # default non attack
            if len(self.detect_enemies) != 0:
                if np.random.uniform(0, 1) < self.args.stationary.radar.attack_precent:
                    attack = self.detect_enemies[0]
                else:
                    attack = 0

            return [direct, attack]
        return [0, 0]







