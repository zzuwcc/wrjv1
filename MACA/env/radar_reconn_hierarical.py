
import gym
from gym.spaces import Dict, Discrete, Box, Tuple
import numpy as np

from ray.rllib.env.multi_agent_env import MultiAgentEnv

from MACA.stationary.radar import Radar
from MACA.stationary.headquarter import Headquarter
from MACA.fighter.fighter_reconnaissance import ReconnaissanceFighter
from MACA.simulator.radar_detect import RadarSimulator
from MACA.render.pygame_render import PygameRender
from MACA.render.stage_gen import StageGenPIT
from MACA.utils.setting import get_args

from MACA.env_wrapper.radar_reconn_hierarical import RadarReconnHieraricalWrapper

class RaderReconnHieraricalEnv(MultiAgentEnv):
    def __init__(self, config):

        args = get_args("param_detect.yaml")
        args_map = get_args("param_map_3.yaml")
        args = merge_args(args, args_map)
        args_fighter = get_args("param_fighter.yaml")
        args = merge_args(args, args_fighter)

        if config and 'render' in config:
            args.render = config['render']
        self.args = args

        is_render = args.render
        self.random_side = args.env.random_side

        # total game status
        self.game_status = None

        # add fighter
        self.allies = [ReconnaissanceFighter(args) for _ in range(args.env.n_ally_reconn)]
                      
        self.enemies = [Radar(args) for _ in range(args.env.n_enemy_radar)] + [Headquarter(args)] 

        # set simulator
        self.simulator = RadarSimulator(args, self.allies, self.enemies)
        self.n_ally = self.simulator.n_ally
        self.n_enemy = self.simulator.n_enemy

        # set env_wrapper
        self.env_wrapper = RadarReconnHieraricalWrapper(args, self.simulator)

        # render
        self.render_sim = None
        if is_render:
            self.render_sim = PygameRender(args, self.allies, self.enemies)

        # RL wrapper
        self.observation_spaces = [
            Box(low=-999.9, high=999.9, shape=(6+(self.n_ally+self.n_enemy-1)*6,), dtype=np.float),
            Box(low=-999.9, high=999.9, shape=(6+(self.n_ally+self.n_enemy-1)*6,), dtype=np.float)
        ]
        self.action_spaces = [
            Box(low=-self.args.fighter.turn_range, 
                high=self.args.fighter.turn_range, 
                shape=(1,)),
            Tuple([
                Box(low=-self.args.fighter.turn_range, 
                    high=self.args.fighter.turn_range, 
                    shape=(1,)),
                Dict({
                    'is_attack': Discrete(2),
                    'attack_target': Discrete(self.n_enemy),
                })
            ])
        ]

        self.step_cnt = 0

    def reset(self):
        self.game_status = self.simulator.initialize(self.random_side)
        self.step_cnt = 0
        return self._get_obs()

    def step(self, actions):
        encoded_actions = self.env_wrapper.action_wrapper(actions, self.simulator)
        self.game_status = self.simulator.step(encoded_actions)

        obs = self._get_obs()
        reward = self._get_reward()
        done = self._get_done()
        info = self._get_info()

        self.step_cnt += 1
        return obs, reward, done, info

    def render(self, save_pic=False):
        if self.render_sim:
            self.render_sim.render(save_pic=save_pic)
    
    def _get_obs(self,):
        return self.env_wrapper.obs_wrapper(self.simulator)

    def _get_reward(self,):
        return self.env_wrapper.reward_wrapper(self.simulator, self.game_status)

    def _get_done(self,):
        return self.env_wrapper.done_wrapper(self.game_status, self.step_cnt)
        
    def _get_info(self,):
        win_info = 'game going'
        # if self.game_status['n_alive_ally'] == 0 and self.game_status['n_alive_enemy'] == 0:
        #     win_info = 'tier'
        # elif self.game_status['n_alive_ally'] == 0:
        #     win_info = 'enemy win'
        # elif self.game_status['n_alive_enemy'] == 0:
        #     win_info = 'ally win'
        if self.game_status['n_alive_ally'] == 0:
            win_info = 'enemy win'
        elif self.game_status['is_detect_home']:
            win_info = 'ally win'

        info = {
            'n_alive_ally': self.game_status['n_alive_ally'],
            'n_alive_enemy': self.game_status['n_alive_enemy'],
            'win_info': win_info,
            'ally_damage': {
                str(i): self.game_status['ally_info'][i]['damage_val'] for i in range(1, self.n_ally+1)
            }
        }

        # infos = {str(i): info for i in range(1, self.n_ally+1)}
        return info

def merge_args(args1, args2):
    for key, value in args2.items():
        args1[key] = value
    return args1
