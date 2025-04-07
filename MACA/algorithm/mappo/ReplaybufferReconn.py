import numpy as np
import torch

class ReplaybufferReconn:
    def __init__(self, batchsize, episode_length, num_agent, obs_dim, state_dim):
        self.bs = batchsize
        self.episode_length = episode_length
        self.num_agent = num_agent
        self.obs_dim = obs_dim
        self.state_dim = state_dim

        self.episode_num = 0
        self.reset_buffer()

    def reset_buffer(self):
        self.buffer = {
            'obs_n': np.zeros((self.bs, self.episode_length, self.num_agent, self.obs_dim)),
            'state': np.zeros((self.bs, self.episode_length, self.state_dim)),
            'action_n':np.zeros((self.bs, self.episode_length, self.num_agent)),
            'action_log_n': np.zeros((self.bs, self.episode_length, self.num_agent)), 
            'value_n': np.zeros((self.bs, self.episode_length + 1, self.num_agent)),
            'reward_n': np.zeros((self.bs, self.episode_length, self.num_agent)),
            'done_n': np.zeros((self.bs, self.episode_length, self.num_agent)),
        }
        self.episode_num = 0

    def store_transition(self, episode_step, obs_n, state, action_n, action_log_n, value_n, reward_n, done_n):
        self.buffer['obs_n'][self.episode_num][episode_step] = obs_n
        self.buffer['state'][self.episode_num][episode_step] = state
        self.buffer['action_n'][self.episode_num][episode_step] = action_n
        self.buffer['action_log_n'][self.episode_num][episode_step] = action_log_n
        self.buffer['value_n'][self.episode_num][episode_step] = value_n
        self.buffer['reward_n'][self.episode_num][episode_step] = reward_n
        self.buffer['done_n'][self.episode_num][episode_step] = done_n

    def store_last_value(self, episode_step, value_n):
        self.buffer['value_n'][self.episode_num][episode_step] = value_n
        self.episode_num += 1

    def get_training_data(self):
        batch = {}
        for key in self.buffer.keys():
            batch[key] = torch.tensor(self.buffer[key], dtype=torch.float32)
        return batch
