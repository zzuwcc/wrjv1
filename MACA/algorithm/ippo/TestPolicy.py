from PolicyCannon import Mappo_Cannon
from PolicyReconn import Mappo_Reconn
import time
import torch
import numpy as np
import argparse
import pandas as pd
import json

import sys
import os
sys.path.append("C:/Workspace/WRJ/MACA-2D")
from MACA.env.cannon_reconn_hierarical import CannonReconnHieraricalEnv
from MACA.render.gif_generator import gif_generate


def main(args):
    env = CannonReconnHieraricalEnv({'render':True})
    args.N = env.n_ally
    args.N_Reconn = env.args.env.n_ally_reconn
    args.N_Cannon = env.args.env.n_ally_cannon
    args.obs_dim = env.observation_spaces[0].shape[0]
    args.obs_dim_n = [args.obs_dim for _ in range(args.N)]
    args.state_dim = np.sum(args.obs_dim_n)
    args.action_dim_Reconn = env.action_spaces[0].shape[0]
    args.action_dim_Cannon = [env.action_spaces[1][0].shape[0],
                    env.action_spaces[1][1]['attack_target'].n + 1]
    args.actor_input_dim = args.obs_dim
    # args.critic_input_dim = args.state_dim
    args.critic_input_dim = args.obs_dim
    args.turn_range = env.args.fighter.turn_range

    if args.add_agent_id:
        args.actor_input_dim_R = args.actor_input_dim + args.N_Reconn
        args.actor_input_dim_C = args.actor_input_dim + args.N_Cannon
    else:
        args.actor_input_dim_R = args.actor_input_dim
        args.actor_input_dim_C = args.actor_input_dim

    env_name = "CannonReconnHieraricalEnv"
    number = args.number
    seed = args.seed
    evaluate_reward_mean_record = np.load('C:/Workspace/WRJ/MACA-2D/MACA/algorithm/ippo/result/evaluate_reward_mean_record_{}.npy'.format(number))
    step = np.argmax(evaluate_reward_mean_record) * args.evaluate_cycle
    # print('------------------------------------{}'.format(step))

    Reconn_policy = Mappo_Reconn(args)
    Reconn_policy.load_model(env_name, number, seed, step)
    Cannon_policy = Mappo_Cannon(args)
    Cannon_policy.load_model(env_name, number, seed, step)

    obs = env.reset()
    h_in_act_R = Reconn_policy.actor.init_hidden(args.N_Reconn)
    h_in_act_C = Cannon_policy.actor.init_hidden(args.N_Cannon)

    total_reward = 0
    while True:
        obs_n_Reconn = obs[:args.N_Reconn]
        obs_n_Cannon = obs[args.N_Reconn:]
        A_R, _, h_in_act_R = Reconn_policy.choose_action(obs_n_Reconn, h_in_act_R, evaluate=True)
        A1_C, A2_C, A1_C_log, A2_C_log, h_in_act_C = Cannon_policy.choose_action(obs_n_Cannon, h_in_act_C, evaluate=True)
        A2_R = np.zeros(args.N_Reconn, dtype=int)

        AA_C = [[A1_C[i], A2_C[i]] for i in range(A1_C.shape[0])]# [N_Cannon, 2]
        AA_R = [[A_R[i], A2_R[i]] for i in range(A_R.shape[0])]# [N_Reconn, 2]
        AA = AA_R + AA_C

        time.sleep(0.02)
        obs, reward, done, info = env.step(AA)
        env.render(save_pic=True)
        total_reward += sum(reward)
        done = done['__all__']
        if done:
            break
    if info['win_info'] == 'ally win':
        is_win = 1
    print(f'Total reward: {total_reward}, is_win: {is_win}')
    gif_generate('C:/Workspace/WRJ/MACA-2D/MACA/algorithm/ippo/result/render_{}.gif'.format(number))

if __name__ == '__main__':
    parset = argparse.ArgumentParser()
    parset.add_argument('--total_steps', type=int, default=100)
    parset.add_argument('--batch_size', type=int, default=32)
    parset.add_argument('--episode_length', type=int, default=600)
    parset.add_argument('--gamma', type=float, default=0.99)
    parset.add_argument('--lamda', type=float, default=0.95)
    parset.add_argument('--K_epochs', type=int, default=8)
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
    main(args)  
    


        

