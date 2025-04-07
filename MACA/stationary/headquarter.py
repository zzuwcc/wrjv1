
import numpy as np
import random

from MACA.stationary.stationary_type import STATIONARY_TYPE
from MACA.stationary.base import BaseStationary

class Headquarter(BaseStationary):
    def __init__(self, args):
        super(Headquarter, self).__init__(args)
        self.type = STATIONARY_TYPE['headquarter']

    def initialize(self, base_info):
        BaseStationary.initialize(self, base_info)
        # self.base_pos = base_info['base_pos']


    def script_action(self, enemies):
        return [0, 0]







