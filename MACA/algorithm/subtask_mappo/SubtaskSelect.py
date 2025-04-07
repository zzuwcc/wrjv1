import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SubtaskEncoderDecoder(nn.Module):
    def __init__(self, args):
        super(SubtaskEncoderDecoder, self).__init__()
        self.args = args
        if args.subtask_encoder_layers == 2:
            self.encoder = nn.Sequential(
                nn.Linear(args.hidden_dim, args.hidden_dim),
                nn.ReLU(),
                nn.Linear(args.hidden_dim, args.K_subtask)
            )
        else:
            self.encoder = nn.Linear(args.hidden_dim, args.K_subtask)
        if args.subtask_decoder_layers == 2:
            self.decoder = nn.Sequential(
                nn.Linear(args.K_subtask, args.hidden_dim),
                nn.ReLU(),
                nn.Linear(args.hidden_dim, args.subtask_emb_dim)
            )
        else:
            self.decoder = nn.Linear(args.K_subtask, args.subtask_emb_dim)
    def forward(self, inputs):
        # inputs: [minibatch * n_agents, hidden_dim]
        tau_emb = self.encoder(inputs)
        subtask_onehot = F.gumbel_softmax(tau_emb, hard=True, dim=-1)
        subtask_emb = self.decoder(subtask_onehot)
        return subtask_emb, subtask_onehot

class SubtaskSelect:
    def __init__(self, args):
        self.args = args
        self.subtask_encoder_decoder = SubtaskEncoderDecoder(args)
        self.optimizer = torch.optim.Adam(self.subtask_encoder_decoder.parameters(), lr=args.lr, eps=1e-5)

    def select_subtask(self, inputs):
        with torch.no_grad():
            _, subtask_onehot = self.subtask_encoder_decoder(inputs)
            return subtask_onehot
    
    def train(self):
        K_subtask_onehot = torch.eye(self.args.K_subtask).detach() # [K_subtask, K_subtask]
        # [K_subtask, subtask_emb_dim]
        K_subtask_emb = self.subtask_encoder_decoder.decoder(K_subtask_onehot)
        # 使得subtask_emb之间的距离尽可能大
        loss = torch.zeros(1)
        for i in range(self.args.K_subtask):
            for j in range(i+1, self.args.K_subtask):
                loss += torch.norm(K_subtask_emb[i] - K_subtask_emb[j])
        loss = self.args.decoder_loss_weight * loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def save_model(self, env_name, number, seed, total_steps):
        torch.save(self.subtask_encoder_decoder.state_dict(), "C:/Workspace/WRJ/MACA-2D/MACA/algorithm/subtask_mappo/model/SubtaskSelect_env_{}_number_{}_seed_{}_step_{}.pth".format(env_name, number, seed, int(total_steps)))

    def load_model(self, env_name, number, seed, step):
        self.subtask_encoder_decoder.load_state_dict(torch.load("C:/Workspace/WRJ/MACA-2D/MACA/algorithm/subtask_mappo/model/SubtaskSelect_env_{}_number_{}_seed_{}_step_{}.pth".format(env_name, number, seed, step)))


