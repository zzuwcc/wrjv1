from PolicyCannon import Mappo_Cannon
from PolicyReconn import Mappo_Reconn
from SubtaskSelect import SubtaskSelect
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
    args.critic_input_dim = args.state_dim
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

    evaluate_reward_mean_record = np.load('C:/Workspace/WRJ/MACA-2D/MACA/algorithm/subtask_mappo/result/evaluate_reward_mean_record_{}.npy'.format(number))
    step = np.argmax(evaluate_reward_mean_record) * args.evaluate_cycle
    # print('------------------------------------{}'.format(step))

    subtask_select = SubtaskSelect(args)
    subtask_select.load_model(env_name, number, seed, step)
    Reconn_policy = Mappo_Reconn(args, subtask_select=subtask_select)
    Reconn_policy.load_model(env_name, number, seed, step)
    Cannon_policy = Mappo_Cannon(args, subtask_select=subtask_select)
    Cannon_policy.load_model(env_name, number, seed, step)

    obs = env.reset()
    h_in_act_R = Reconn_policy.actor.init_hidden(args.N_Reconn)
    h_in_act_C = Cannon_policy.actor.init_hidden(args.N_Cannon)

    total_reward = 0
    is_win = 0

    # key: {name: WRJ1, step: 1, subtask: 1(围捕), action:{ori: 0.23, attack_target: 2}}
    skill_data = {}
    skill_dec = {0: '追击', 1: '围捕', 2: '集火'}
    step_cnt = 0

    while True:
        obs_n_Reconn = obs[:args.N_Reconn]
        obs_n_Cannon = obs[args.N_Reconn:]
        A_R, _, h_in_act_R = Reconn_policy.choose_action(obs_n_Reconn, h_in_act_R, evaluate=True)
        A1_C, A2_C, A1_C_log, A2_C_log, h_in_act_C = Cannon_policy.choose_action(obs_n_Cannon, h_in_act_C, evaluate=True)
        A2_R = np.zeros(args.N_Reconn, dtype=int)

        AA_C = [[A1_C[i], A2_C[i]] for i in range(A1_C.shape[0])]# [N_Cannon, 2]
        AA_R = [[A_R[i], A2_R[i]] for i in range(A_R.shape[0])]# [N_Reconn, 2]
        AA = AA_R + AA_C

        skill_Reconns = subtask_select.select_subtask(h_in_act_R)
        skill_Cannons = subtask_select.select_subtask(h_in_act_C)
        # save txt file
        # with open('./skill_assignment.txt', 'a') as f:
        #     for i in range(args.N_Reconn):
        #         max_index = torch.argmax(skill_Reconns[i])
        #         f.write(f'Reconn {i}: {skill_Reconns[i]}, {max_index}\n')
        #     for i in range(args.N_Cannon):
        #         max_index = torch.argmax(skill_Cannons[i])
        #         f.write(f'Cannon {i}: {skill_Cannons[i]}, {max_index}\n')
        #     f.write('------------------------------------\n')

        # xx = []
        # for i in range(args.N):
        #     if i < args.N_Reconn:
        #         xx.append(torch.argmax(skill_Reconns[i]).item())
        #     else:
        #         xx.append(torch.argmax(skill_Cannons[i - args.N_Reconn]).item())
        # skill_data.append(xx)

        step_data = {}
        for i in range(args.N):
            if i < args.N_Reconn:
                max_index = torch.argmax(skill_Reconns[i]).item()
            else:
                max_index = torch.argmax(skill_Cannons[i - args.N_Reconn]).item()
            step_data['WRJ{}'.format(i + 1)] = {'High-level Task': str(max_index) + '-' +  skill_dec[max_index], 'Low-level Task': {'Orientation': float(AA[i][0]), 'Attack target': int(AA[i][1])}}
        skill_data['Time Step {}'.format(step_cnt)] = step_data
        step_cnt += 1

        time.sleep(0.02)
        obs, reward, done, info = env.step(AA)
        env.render(save_pic=True)
        total_reward += sum(reward)
        done = done['__all__']
        if done:
            break

    # colnum = [f'Reconn{i + 1}' for i in range(args.N_Reconn)] + [f'Cannon{i + 1}' for i in range(args.N_Cannon)]
    # df = pd.DataFrame(skill_data, columns=colnum)
    # df.to_csv('./subtask_assignment.csv', index=False)

    with open("C:/Workspace/WRJ/MACA-2D/MACA/algorithm/subtask_mappo/result/subtask_assignment_detail_{}.json".format(number), 'w', encoding = 'utf-8') as f:
        json.dump(skill_data, f, indent=4, ensure_ascii=False)

    if info['win_info'] == 'ally win':
        is_win = 1
    print(f'Total reward: {total_reward}, is_win: {is_win}')
    gif_generate('C:/Workspace/WRJ/MACA-2D/MACA/algorithm/subtask_mappo/result/render_{}.gif'.format(number))

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
    parset.add_argument('--add_agent_id', type=bool, default=True)
    parset.add_argument('--K_subtask', type=int, default=3)
    parset.add_argument('--subtask_encoder_layers', type=int, default=2)
    parset.add_argument('--subtask_decoder_layers', type=int, default=2)
    parset.add_argument('--subtask_emb_dim', type=int, default=64)
    parset.add_argument('--is_detach', type=bool, default=True)
    parset.add_argument('--decoder_loss_weight', type=float, default=0.05)
    parset.add_argument('--evaluate_nums', type=int, default=10)
    parset.add_argument('--number', type=int, default=0)
    parset.add_argument('--seed', type=int, default=0)
    args = parset.parse_args()
    main(args)  
    


        

