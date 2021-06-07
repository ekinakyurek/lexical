import torch
from torch import nn
import torch.nn.functional as F
import pdb

class SimpleAttention(nn.Module):
    def __init__(self, n_features, n_hidden, key=False, copy=False, query=True, memory=False):
        super().__init__()
        self.key = key
        self.query = query
        self.memory = memory
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.copy = copy

        if self.copy: assert self.query

        if self.key:
            self.make_key = nn.Linear(n_features, n_hidden)
        if self.query:
            self.make_query = nn.Linear(n_features, (1+copy) * n_hidden)
        if self.memory:
            self.make_memory = nn.Linear(n_features, n_hidden)
        self.n_out = n_hidden

    def forward(self, features, hidden, mask=None):
        if self.key:
            key = self.make_key(features)
        else:
            key = features

        if self.memory:
            memory = self.make_memory(features)
        else:
            memory = features

        if self.query:
            query = self.make_query(hidden)
        else:
            query = hidden

        # attention
        #query = query.expand_as(key) # T X B X H
        if self.copy:
            query = query.view(1, -1, query.shape[-1] // 2, 2)
            key   = key.unsqueeze(-1)
            mask  = mask.unsqueeze(-1)
        elif len(query.shape) < 3:
            query = query.unsqueeze(0)

        scores = (key * query).sum(dim=2)

        if mask is not None:
            scores += mask * -99999

        if self.copy:
            scores, copy_scores = torch.chunk(scores,2,dim=-1)
            copy_distribution = F.softmax(copy_scores.squeeze(-1), dim=0)
            distribution = F.softmax(scores.squeeze(-1), dim=0)
        else:
            distribution = F.softmax(scores, dim=0)
            copy_distribution = distribution


        weighted = (memory * distribution.unsqueeze(2).expand_as(memory))
        summary = weighted.sum(dim=0, keepdim=True)

        # value
        return summary, distribution, copy_distribution
