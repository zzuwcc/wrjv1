import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd

def trans(x):
    return x * args.evaluate_cycle * args.K_epochs * (args.batch_size / args.mini_batch_size) / 1000

def plot_train_reward(args):
    evaluate_reward_mean_record = np.load('C:/Workspace/WRJ/MACA-2D/MACA/algorithm/ippo/result/evaluate_reward_mean_record_{}.npy'.format(args.number))
    evaluate_reward_mean_record = pd.Series(evaluate_reward_mean_record).rolling(window=args.window_size, center=True).mean()

    plt.plot(evaluate_reward_mean_record)
    plt.xlabel('Episodes (1e3)')
    plt.ylabel('Mean Reward')
    plt.title('Mean Reward of Evaluation')

    x_ticks = np.arange(0, len(evaluate_reward_mean_record), 10)
    x_labels = x_ticks * args.evaluate_cycle * args.K_epochs * (args.batch_size / args.mini_batch_size) / 1000
    plt.xticks(ticks=x_ticks, labels=x_labels)

    for i, reward in enumerate(evaluate_reward_mean_record):
        plt.annotate(f'({trans(i):.2f}, {reward:.0f})', (i, reward), textcoords="offset points", xytext=(0,10), ha='center', fontsize=4)

    plt.savefig('C:/Workspace/WRJ/MACA-2D/MACA/algorithm/ippo/result/DTDE_{}.png'.format(args.number), dpi=1200)
    plt.show()

    with open('C:/Workspace/WRJ/MACA-2D/MACA/algorithm/ippo/result/plot_info_{}.txt'.format(args.number), 'w') as f:
        f.write('Episode\tMean Reward\n')
        for episode, reward in enumerate(evaluate_reward_mean_record):
            f.write(f'{episode* args.evaluate_cycle * args.K_epochs * (args.batch_size / args.mini_batch_size) / 1000}\t{reward}\n')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--number', type=int, default=5)
    parser.add_argument('--K_epochs', type=int, default=16)
    parser.add_argument('--mini_batch_size', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--evaluate_cycle', type=int, default=2)
    parser.add_argument('--window_size', type=int, default=7)
    args = parser.parse_args()
    plot_train_reward(args)