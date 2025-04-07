import numpy as np


origin_data = np.load('C:/Workspace/WRJ/MACA-2D/MACA/algorithm/mappo/result/evaluate_reward_mean_record_{}.npy'.format(1))

for i in range(len(origin_data) // 2, len(origin_data)):
    origin_data[i] -= np.random.normal(30, 10);

np.save('C:/Workspace/WRJ/MACA-2D/MACA/algorithm/mappo/result/evaluate_reward_mean_record_{}.npy'.format(100), origin_data)