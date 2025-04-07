
from enum import Enum
import numpy as np

from MACA.stationary.stationary_type import STATIONARY_TYPE

class BaseStationary():
    def __init__(self, args):
        self.args = args

        self.dt = args.simulator.dt

        # 位置相关信息
        self.id = None
        self.type = STATIONARY_TYPE['base']
        self.side = None
        self.alive = None
        self.pos = None
        self.initial_pos = None

        # 转交范围
        self.turn_range = None

        # 地图信息
        self.map_size = None

        # 探测范围
        self.detect_range = None

        # 探测的ally和enemy
        self.detect_allies = []
        self.detect_enemies = []

        # 攻击相关信息
        self.damage = None
        self.damage_range = None
        self.damage_turn_range = None

        # last action
        self.last_action = [0.0, 0]
    
    def initialize(self, base_info):
        self.id = base_info['id']
        self.side = base_info['side']
        self.alive = True

        self.pos = base_info['pos']
        self.initial_pos = [self.pos[0], self.pos[1]]
        self.ori = 0
        self.speed = 40
        self.bloods = 100

        self.map_size = [base_info['map_x_limit'], base_info['map_y_limit']]

        # 探测范围
        self.detect_range = base_info['detect_range']

        # 攻击相关信息
        self.damage = base_info['damage']
        self.damage_range = base_info['damage_range']
        self.damage_turn_range = base_info['damage_turn_range']

        # 检测清零
        self.detect_allies = []
        self.detect_enemies = []

        # last action
        self.last_action = [0.0, 0]

    def step(self, direct, be_attacked, attack, attack_bias=1.0):
        # last action recorded
        self.last_action = [direct, attack]

    def script_action(self, enemies):
        raise NotImplementedError

    def _angle_clip(self, angle):
        while angle > 2 * np.pi:
            angle -= 2 * np.pi
        while angle < 0:
            angle += 2 * np.pi
        return angle