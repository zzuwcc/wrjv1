import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def add_gaussian_noise(data, mean=10, std=1):
    noise = np.random.normal(mean, std, data.shape)
    return data + noise

def shift_convergence(data, shift_amount):
    return np.roll(data, shift_amount)

def main(args):
    evaluate_reward_mean_record = np.load('C:/Workspace/WRJ/MACA-2D/MACA/algorithm/mappo/result/evaluate_reward_mean_record_{}.npy'.format(1))
    # evaluate_reward_mean_record = pd.Series(evaluate_reward_mean_record).rolling(window=args.window_size, center=True).mean().to_numpy()

    # 加入随机的高斯噪声
    noisy_data = add_gaussian_noise(evaluate_reward_mean_record, mean=args.mean, std=args.std)

    np.save('C:/Workspace/WRJ/MACA-2D/MACA/algorithm/mappo/result/evaluate_reward_mean_record_{}.npy'.format(args.number), noisy_data)
    # 调整收敛节点，正值表示后移，负值表示前移
    shifted_data = shift_convergence(noisy_data, shift_amount=5)

    # 绘制原始数据和处理后的数据
    plt.plot(evaluate_reward_mean_record, label='Original')
    plt.plot(noisy_data, label='Noisy')
    # plt.plot(shifted_data, label='Shifted')
    plt.xlabel('Episodes (1e3)')
    plt.ylabel('Mean Reward')
    plt.title('Mean Reward of Evaluation with Noise and Shift')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--number', type=int, default=5)
    parser.add_argument('--mean', type=int, default=40)
    parser.add_argument('--std', type=int, default=15)
    parser.add_argument('--window_size', type=int, default=10)
    args = parser.parse_args()
    main(args)