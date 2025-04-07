import torch
import numpy as np
from torch.nn import functional as F

# mu = torch.zeros((5, 1))
# std = torch.ones((5, 1))

# dist = torch.distributions.Normal(mu, std)
# action = dist.sample()
# print(action)
# action_log = dist.log_prob(action)
# print(action_log)

# softmax_output = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.1], [0.3, 0.4, 0.1, 0.2], [0.4, 0.1, 0.2, 0.3], [0.5, 0.5, 0.5, 0.5]])
# dist = torch.distributions.Categorical(softmax_output)
# action = dist.mode
# print(action)
# action_log = dist.log_prob(action)
# print(action_log)

# a = [1, 2, 3]
# print(np.sum(a))

# a = np.array([1, 2, 3])
# b = np.array([4, 5, 6])
# c = np.stack((a, b), axis=1)
# # print(c)


# b = np.ones(2, dtype=np.float64)
# a = np.zeros(2, dtype=int)
# c = [[a[i], b[i]] for i in range(a.shape[0])]
# print(c)
# print(c[0][0].dtype), print(c[0][1].dtype)

# a = torch.tensor([1, 2, 3])
# b = torch.tensor([4, 5, 6])
# print(torch.exp(b - a))

# a = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
# a_one_hot = F.gumbel_softmax(a, hard=True, dim = -1)
# print(a_one_hot)

# N = 5
# a_onehot = torch.eye(N)
# print(a_onehot)

# a = [1, 2, 3]
# max_index = a.index(max(a))
# print(sum(a))
# print(max_index)

# b = [1]
# tensor_a = torch.tensor([1, 2, 3])
# a = torch.argmax(tensor_a).item()
# print(type(a))
# b.append(a)
# print(b)

# skill_dec = {0: '追击', 1: '围捕', '2': '集火'}
# print(skill_dec[0])

# list_a = [1, 2, 3, 4, 5]
# np.save('list_a.npy', list_a)


# load_list_a = np.load('list_a.npy')
# print(type(load_list_a))
# print(np.argmax(load_list_a))