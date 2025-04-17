import torch
import numpy as np
import argparse
from ReplaybufferCannon import ReplaybufferCannon
from ReplaybufferReconn import ReplaybufferReconn
from PolicyCannon import Mappo_Cannon
from PolicyReconn import Mappo_Reconn
import sys
import os
# from MACA.env.cannon_reconn_hierarical import CannonReconnHieraricalEnv

root_path = os.path.join(os.path.dirname(__file__), '../../../')
sys.path.append(root_path)
from MACA.env.radar_reconn_hierarical import RaderReconnHieraricalEnv
import time

class RunnerMACA:
    def __init__(self, args, env_name):
        self.args = args
        self.env_name = env_name
        self.number = args.number
        self.seed = args.seed

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.env = RaderReconnHieraricalEnv(None)
        self.args.N = self.env.n_ally
        self.args.N_Reconn = self.env.args.env.n_ally_reconn
        self.args.obs_dim = self.env.observation_spaces[0].shape[0]
        self.args.obs_dim_n = [self.args.obs_dim for _ in range(self.args.N)]
        self.args.state_dim = np.sum(self.args.obs_dim_n)
        self.args.action_dim_Reconn = self.env.action_spaces[0].shape[0]
        # self.args.action_dim_Cannon = [self.env.action_spaces[1][0].shape[0],
        #                                self.env.action_spaces[1][1]['attack_target'].n + 1]
        self.args.actor_input_dim = self.args.obs_dim
        # self.args.critic_input_dim = self.args.state_dim
        self.args.critic_input_dim = self.args.obs_dim
        self.args.turn_range = self.env.args.fighter.turn_range
        if args.add_agent_id:
            self.args.actor_input_dim_R = self.args.actor_input_dim + self.args.N_Reconn
            # self.args.actor_input_dim_C = self.args.actor_input_dim + self.args.N_Cannon
        else:
            self.args.actor_input_dim_R = self.args.actor_input_dim
            # self.args.actor_input_dim_C = self.args.actor_input_dim



        self.agent_Reconn = Mappo_Reconn(self.args)
        # self.agent_Cannon = Mappo_Cannon(self.args)
        self.replaybuffer_Reconn = ReplaybufferReconn(self.args.batch_size, self.args.episode_length, self.args.N_Reconn, self.args.obs_dim, self.args.state_dim)
        # self.replaybuffer_Cannon = ReplaybufferCannon(self.args.batch_size, self.args.episode_length, self.args.N_Cannon, self.args.obs_dim, self.args.state_dim)
        self.update_steps = 0

    def run(self):
        evaluate_times = 0
        reward_mean_record = []
        while self.update_steps < self.args.total_steps:
            episode_reward = self.run_episode(evaluate=False)
            # print(self.replaybuffer_Cannon.episode_num, self.replaybuffer_Reconn.episode_num)
            if self.replaybuffer_Reconn.episode_num >= self.args.batch_size:
                self.agent_Reconn.train(self.replaybuffer_Reconn, self.update_steps)
                self.replaybuffer_Reconn.reset_buffer()
                # self.agent_Cannon.train(self.replaybuffer_Cannon, self.update_steps)
                # self.replaybuffer_Cannon.reset_buffer()
                self.update_steps += 1
            if self.update_steps >= evaluate_times * self.args.evaluate_cycle:
                evaluate_reward = []
                win_nums = 0
                for _ in range(self.args.evaluate_nums):
                    episode_reward, win_info = self.run_episode(evaluate=True)
                    evaluate_reward.append(episode_reward)
                    if win_info == 'ally win':
                        win_nums += 1
                evaluate_times += 1
                reward_mean = sum(evaluate_reward) / self.args.evaluate_nums
                win_rate = win_nums / self.args.evaluate_nums
                print(f'Episode: {self.update_steps}, Reward: {reward_mean}, WinRate: {win_rate}')

                reward_mean_record.append(reward_mean)
                with open("./MACA/algorithm/ippo/result/log_{}.txt".format(self.number), "a") as f:
                    f.write(f'Episode: {self.update_steps}, Reward: {episode_reward}. WinRate: {win_rate}\n')
                
                # self.agent_Cannon.save_model(self.env_name, self.number, self.seed, self.update_steps)
                self.agent_Reconn.save_model(self.env_name, self.number, self.seed, self.update_steps)

        np.save("./MACA/algorithm/ippo/result/evaluate_reward_mean_record_{}.npy".format(self.number), reward_mean_record)

    
    def run_episode(self, evaluate=False):
        cnt = 0
        episode_reward = 0
        obs_n = self.env.reset()
        # obs_n = [obs for id, obs in obs_n.items()]
        # h_in_act_C = self.agent_Cannon.actor.init_hidden(self.args.N_Cannon)
        # h_in_cri_C = self.agent_Cannon.critic.init_hidden(self.args.N_Cannon)
        h_in_act_R = self.agent_Reconn.actor.init_hidden(self.args.N_Reconn)
        h_in_cri_R = self.agent_Reconn.critic.init_hidden(self.args.N_Reconn)

        for episode_step in range(self.args.episode_length):
            obs_n_Reconn = obs_n[:self.args.N_Reconn]
            # obs_n_Cannon = obs_n[self.args.N_Reconn:]
            state = np.array(obs_n).flatten()

            V_R, h_in_cri_R = self.agent_Reconn.get_value(obs_n_Reconn, h_in_cri_R) # [N_Reconn]
            # V_C, h_in_cri_C = self.agent_Cannon.get_value(obs_n_Cannon, h_in_cri_C)
            A_R, A_R_log, h_in_act_R = self.agent_Reconn.choose_action(obs_n_Reconn, h_in_act_R, evaluate)
            # A1_C, A2_C, A1_C_log, A2_C_log, h_in_act_C = self.agent_Cannon.choose_action(obs_n_Cannon, h_in_act_C, evaluate)
            A2_R = np.zeros(self.args.N_Reconn, dtype=int)

            # AA_C = [[A1_C[i], A2_C[i]] for i in range(A1_C.shape[0])]# [N_Cannon, 2]
            AA_R = [[A_R[i], A2_R[i]] for i in range(A_R.shape[0])]# [N_Reconn, 2]
            # AA = AA_R + AA_C
            AA = A_R

            next_obs_n, reward_n, done, info = self.env.step(AA)
            reward_n_Reconn = reward_n[:self.args.N_Reconn]
            # reward_n_Cannon = reward_n[self.args.N_Reconn:]
            episode_reward += sum(reward_n)
            done = done["__all__"]

            if not evaluate:
                self.replaybuffer_Reconn.store_transition(episode_step, obs_n_Reconn, state, A_R, A_R_log, V_R, reward_n_Reconn, done)
                # self.replaybuffer_Cannon.store_transition(episode_step, obs_n_Cannon, state, A1_C, A2_C, A1_C_log, A2_C_log, V_C, reward_n_Cannon, done)
            else:
                time.sleep(0.02)
                self.env.render(save_pic=True)
            
            obs_n = next_obs_n
            if done:
                break
        if not evaluate:
            # state = np.array(obs_n).flatten()
            V_R, _ = self.agent_Reconn.get_value(obs_n_Reconn, h_in_cri_R)
            # V_C, _ = self.agent_Cannon.get_value(obs_n_Cannon, h_in_cri_C)
            self.replaybuffer_Reconn.store_last_value(episode_step, V_R)
            # self.replaybuffer_Cannon.store_last_value(episode_step, V_C)
        win_info = info['win_info']
        return episode_reward, win_info

if __name__ == '__main__':
    parset = argparse.ArgumentParser()
    parset.add_argument('--total_steps', type=int, default=200)
    parset.add_argument('--batch_size', type=int, default=32)
    parset.add_argument('--episode_length', type=int, default=600)
    parset.add_argument('--gamma', type=float, default=0.99)
    parset.add_argument('--lamda', type=float, default=0.95)
    parset.add_argument('--K_epochs', type=int, default=16)
    parset.add_argument('--mini_batch_size', type=int, default=8)
    parset.add_argument('--use_adv_norm', type=bool, default=True)
    parset.add_argument('--evaluate_cycle', type=int, default=2)
    parset.add_argument('--hidden_dim', type=int, default=64)
    parset.add_argument('--lr', type=float, default=5e-4)
    parset.add_argument('--clip_param', type=float, default=0.2)
    parset.add_argument('--add_agent_id', type=bool, default=False)
    parset.add_argument('--evaluate_nums', type=int, default=10)
    parset.add_argument('--seed', type=int, default=0)
    parset.add_argument('--number', type=int, default=0)
    args = parset.parse_args()
    args.number = 2
    args.seed = 0
    runner = RunnerMACA(args, 'RaderReconnHieraricalEnv')
    runner.run()
        
