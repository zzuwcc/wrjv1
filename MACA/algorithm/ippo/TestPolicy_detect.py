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
root_path = os.path.join(os.path.dirname(__file__), '../../../')
sys.path.append(root_path)
from MACA.env.radar_reconn_hierarical import RaderReconnHieraricalEnv
from MACA.render.gif_generator import gif_generate


def main(args, env_name:str):
    env = RaderReconnHieraricalEnv({'render':True}, args.map_name)
    args.N = env.n_ally
    args.N_Reconn = env.args.env.n_ally_reconn
    # args.N_Cannon = env.args.env.n_ally_cannon
    args.obs_dim = env.observation_spaces[0].shape[0]
    args.obs_dim_n = [args.obs_dim for _ in range(args.N)]
    args.state_dim = np.sum(args.obs_dim_n)
    args.action_dim_Reconn = env.action_spaces[0].shape[0]
    # args.action_dim_Cannon = [env.action_spaces[1][0].shape[0],
    #                 env.action_spaces[1][1]['attack_target'].n + 1]
    args.actor_input_dim = args.obs_dim
    # args.critic_input_dim = args.state_dim
    args.critic_input_dim = args.obs_dim
    args.turn_range = env.args.fighter.turn_range

    if args.add_agent_id:
        args.actor_input_dim_R = args.actor_input_dim + args.N_Reconn
        # args.actor_input_dim_C = args.actor_input_dim + args.N_Cannon
    else:
        args.actor_input_dim_R = args.actor_input_dim
        # args.actor_input_dim_C = args.actor_input_dim

    env_name = "RaderReconnHieraricalEnv"
    number = args.number
    seed = args.seed
    step = args.step 

    try: 
        evaluate_reward_mean_record = np.load(f"./MACA/algorithm/ippo/result/{env_name}/{args.map_name}/evaluate_reward_mean_record_{number}.npy")
        step = np.argmax(evaluate_reward_mean_record) * args.evaluate_cycle
        print('----------------the best poilcy is step: {}-----------------'.format(step))
    except FileNotFoundError as e:
        import traceback
        print(f"File not found: {e}")
        traceback.print_exc()
    except Exception as e:
        import traceback
        print(f"An error occurred: {e}")
        traceback.print_exc()

    Reconn_policy = Mappo_Reconn(args)
    Reconn_policy.load_model(env_name, number, seed, step, args.map_name)
    # Cannon_policy = Mappo_Cannon(args)
    # Cannon_policy.load_model(env_name, number, seed, step)

    obs = env.reset()
    h_in_act_R = Reconn_policy.actor.init_hidden(args.N_Reconn)
    # h_in_act_C = Cannon_policy.actor.init_hidden(args.N_Cannon)

    total_reward = 0
    while True:
        obs_n_Reconn = obs[:args.N_Reconn]
        obs_n_Cannon = obs[args.N_Reconn:]
        A_R, _, h_in_act_R = Reconn_policy.choose_action(obs_n_Reconn, h_in_act_R, evaluate=True)
        # A1_C, A2_C, A1_C_log, A2_C_log, h_in_act_C = Cannon_policy.choose_action(obs_n_Cannon, h_in_act_C, evaluate=True)
        A2_R = np.zeros(args.N_Reconn, dtype=int)

        # AA_C = [[A1_C[i], A2_C[i]] for i in range(A1_C.shape[0])]# [N_Cannon, 2]
        AA_R = [[A_R[i], A2_R[i]] for i in range(A_R.shape[0])]# [N_Reconn, 2]
        # AA = AA_R + AA_C
        AA = A_R

        time.sleep(0.02)
        obs, reward, done, info = env.step(AA)
        env.render(save_pic=True)
        total_reward += sum(reward)
        done = done['__all__']
        if done:
            break
    is_win = 0
    if info['win_info'] == 'ally win':
        is_win = 1
    print(f'Total reward: {total_reward}, is_win: {is_win}')
    
    gif_root_path = f"./MACA/algorithm/ippo/result/{env_name}/{args.map_name}"
    gif_generate(os.path.join(gif_root_path, 'render_{}.gif'.format(number)))

if __name__ == '__main__':
    parset = argparse.ArgumentParser()
    parset.add_argument('--total_steps', type=int, default=200)
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
    parset.add_argument('--number', type=int, default=15)
    parset.add_argument('--step', type=int, default=90)
    parset.add_argument('--map_name', type=str, default='zc_easy')
    args = parset.parse_args()
    main(args, 'RaderReconnHieraricalEnv')  
    


        

