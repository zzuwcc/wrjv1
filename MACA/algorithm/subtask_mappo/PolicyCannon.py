import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SequentialSampler
from torch.nn import functional as F

class Actor(nn.Module):
    def __init__(self, args, subtask_select):
        self.args = args
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(args.actor_input_dim_C, args.hidden_dim)
        self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        self.subtask_select = subtask_select
        self.fc_mu_w = nn.Linear(args.subtask_emb_dim, args.hidden_dim * args.action_dim_Cannon[0])
        self.fc_mu_b = nn.Linear(args.subtask_emb_dim, args.action_dim_Cannon[0])
        self.fc_std_w = nn.Linear(args.subtask_emb_dim, args.hidden_dim * args.action_dim_Cannon[0])
        self.fc_std_b = nn.Linear(args.subtask_emb_dim, args.action_dim_Cannon[0])
        self.action2_w = nn.Linear(args.subtask_emb_dim, args.hidden_dim * args.action_dim_Cannon[1])
        self.action2_b = nn.Linear(args.subtask_emb_dim, args.action_dim_Cannon[1])
            

    def init_hidden(self, bs):
        return torch.zeros((bs, self.args.hidden_dim), device=self.fc1.weight.device)
    
    def forward(self, inputs, h_in):
        # inputs: [minibatch * n_agents, obs_dim]
        x = F.relu(self.fc1(inputs))
        h_out = self.rnn(x, h_in) # [minibatch * n_agents, hidden_dim]
        if self.args.is_detach:
            tau = h_out.detach()
        else:
            tau = h_out
        subtask_emb, subtask_onehot = self.subtask_select.subtask_encoder_decoder(tau)
        mu_w = self.fc_mu_w(subtask_emb).reshape(-1, self.args.hidden_dim, self.args.action_dim_Cannon[0])
        mu_b = self.fc_mu_b(subtask_emb).reshape(-1, self.args.action_dim_Cannon[0])
        std_w = self.fc_std_w(subtask_emb).reshape(-1, self.args.hidden_dim, self.args.action_dim_Cannon[0])
        std_b = self.fc_std_b(subtask_emb).reshape(-1, self.args.action_dim_Cannon[0])
        mu = self.args.turn_range * torch.tanh(torch.bmm(h_out.unsqueeze(1), mu_w).squeeze(1) + mu_b)
        std = F.softplus(torch.bmm(h_out.unsqueeze(1), std_w).squeeze(1) + std_b) + 1e-5
        action2_w = self.action2_w(subtask_emb).reshape(-1, self.args.hidden_dim, self.args.action_dim_Cannon[1])
        action2_b = self.action2_b(subtask_emb).reshape(-1, self.args.action_dim_Cannon[1])
        action2 = F.softmax(torch.bmm(h_out.unsqueeze(1), action2_w).squeeze(1) + action2_b, dim=-1)
        return mu, std, action2, h_out
    
class Critic(nn.Module):
    def __init__(self, args):
        self.args = args
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(args.critic_input_dim, args.hidden_dim)
        self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, 1)
    
    def init_hidden(self, bs):
        return torch.zeros((bs, self.args.hidden_dim), device=self.fc1.weight.device)
    
    def forward(self, inputs, h_in):
        # inputs: [minibatch * n_agents, state_dim]
        x = F.relu(self.fc1(inputs))
        h_out = self.rnn(x, h_in)
        v = self.fc3(h_out)
        return v, h_out

class Mappo_Cannon:
    def __init__(self, args, subtask_select):
        self.args = args
        self.subtask_select = subtask_select
        self.actor = Actor(args, subtask_select)
        self.critic = Critic(args)
        self.ac_params = list(self.actor.parameters()) + list(self.critic.parameters()) + list(self.subtask_select.subtask_encoder_decoder.parameters())
        self.ac_optimizer = torch.optim.Adam(self.ac_params, lr=args.lr, eps=1e-5)

    def choose_action(self, obs_n, h_in, evaluate):
        with torch.no_grad():
            # [n_agents, obs_dim]
            obs_n = torch.tensor(obs_n, dtype=torch.float32)
            if self.args.add_agent_id:
                obs_n = torch.cat([obs_n, torch.eye(self.args.N_Cannon)], dim=-1)
            mu, std, action2, h_out = self.actor(obs_n, h_in)
            if evaluate:
                action1 = mu
                dist2 = torch.distributions.Categorical(action2)
                action2 = dist2.mode
                return action1.squeeze(-1).numpy(), action2.numpy(), None, None, h_out # [n_agents]
            else:
                dist1 = torch.distributions.Normal(mu, std)
                action1 = dist1.sample()
                dist2 = torch.distributions.Categorical(action2)
                action2 = dist2.sample()
                action_log1 = dist1.log_prob(action1)
                action_log2 = dist2.log_prob(action2)
            return action1.squeeze(-1).numpy(), action2.numpy(), action_log1.squeeze(-1).numpy(), action_log2.squeeze(-1).numpy(), h_out
        
    def get_value(self, state, h_in):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).repeat(self.args.N_Cannon, 1)
            v_n, h_out = self.critic(state, h_in)
            return v_n.squeeze(-1).numpy(), h_out # [n_agents]
    
    def get_inputs(self, batch):
        actor_inputs = batch['obs_n']
        # [bs, episode_length, N_Cannon, obs_dim]
        if self.args.add_agent_id:
            actor_inputs = torch.cat([actor_inputs, torch.eye(self.args.N_Cannon).unsqueeze(0).unsqueeze(0).repeat(self.args.batch_size, self.args.episode_length, 1, 1)], dim=-1)
        critic_input = batch['state'].unsqueeze(2).repeat(1, 1, self.args.N_Cannon, 1)
        return actor_inputs, critic_input
    
    def train(self, replay_buffer, total_steps):
        batch = replay_buffer.get_training_data()

        # calculate the advantage using GAE
        adv = []
        gae = 0
        with torch.no_grad():  # adv and td_target have no gradient
            deltas = batch['reward_n'] + self.args.gamma * batch['value_n'][:, 1:] * (1 - batch['done_n']) - batch['value_n'][:, :-1]  # deltas.shape=(batch_size,episode_limit,N)
            for t in reversed(range(self.args.episode_length)):
                gae = deltas[:, t] + self.args.gamma * self.args.lamda * gae
                adv.insert(0, gae)
            adv = torch.stack(adv, dim=1)  # adv.shape(batch_size,episode_limit,N)
            v_target = adv + batch['value_n'][:, :-1]  # v_target.shape(batch_size,episode_limit,N)
            if self.args.use_adv_norm:  # Trick 1: advantage normalization
                adv = ((adv - adv.mean()) / (adv.std() + 1e-5))
        
        actor_inputs, critic_inputs = self.get_inputs(batch)

        # train actor
        # update steps: K_epochs * (batch_size / mini_batch_size)
        for _ in range(self.args.K_epochs):
            for index in BatchSampler(SequentialSampler(range(self.args.batch_size)), self.args.mini_batch_size, False):
                h_in_actor = self.actor.init_hidden(self.args.mini_batch_size * self.args.N_Cannon)
                h_in_critic = self.critic.init_hidden(self.args.mini_batch_size * self.args.N_Cannon)
                mu_now, std_now, action2_prob_now, value_now = [], [], [], []
                for t in range(self.args.episode_length):
                    mu, std, action2, h_in_actor = self.actor(actor_inputs[index, t].reshape(self.args.mini_batch_size * self.args.N_Cannon, -1), h_in_actor)
                    v, h_in_critic = self.critic(critic_inputs[index, t].reshape(self.args.mini_batch_size * self.args.N_Cannon, -1), h_in_critic)
                    mu_now.append(mu.reshape(self.args.mini_batch_size, self.args.N_Cannon, -1))
                    std_now.append(std.reshape(self.args.mini_batch_size, self.args.N_Cannon, -1))
                    action2_prob_now.append(action2.reshape(self.args.mini_batch_size, self.args.N_Cannon, -1))
                    value_now.append(v.reshape(self.args.mini_batch_size, self.args.N_Cannon))
                mu_now = torch.stack(mu_now, dim=1)
                std_now = torch.stack(std_now, dim=1)
                action2_prob_now = torch.stack(action2_prob_now, dim=1)
                value_now = torch.stack(value_now, dim=1)

                dist1 = torch.distributions.Normal(mu_now, std_now)
                dist2 = torch.distributions.Categorical(action2_prob_now)
                action_log1 = dist1.log_prob(batch['action1_n'][index].unsqueeze(-1)).squeeze(-1)
                action_log2 = dist2.log_prob(batch['action2_n'][index])
                ratio = torch.exp(action_log1 + action_log2 - batch['action1_log_n'][index].detach() - batch['action2_log_n'][index].detach())
                surr1 = ratio * adv[index]
                surr2 = torch.clamp(ratio, 1-self.args.clip_param, 1+self.args.clip_param) * adv[index]
                actor_loss = -torch.min(surr1, surr2)
                critic_loss = F.mse_loss(value_now, v_target[index])
                ac_loss = actor_loss.mean() + critic_loss.mean()
                self.ac_optimizer.zero_grad()
                ac_loss.backward()
                self.ac_optimizer.step()

    def save_model(self, env_name, number, seed, total_steps):
        torch.save(self.actor.state_dict(), "C:/Workspace/WRJ/MACA-2D/MACA/algorithm/subtask_mappo/model/Cannon_actor_env_{}_number_{}_seed_{}_step_{}k.pth".format(env_name, number, seed, int(total_steps)))

    def load_model(self, env_name, number, seed, step):
        self.actor.load_state_dict(torch.load("C:/Workspace/WRJ/MACA-2D/MACA/algorithm/subtask_mappo/model/Cannon_actor_env_{}_number_{}_seed_{}_step_{}k.pth".format(env_name, number, seed, step)))
