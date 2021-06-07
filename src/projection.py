import torch
from torch import nn
import torch.nn.functional as F
import pdb
from torch.nn.parameter import Parameter
EPS = 1e-7

class SoftAlign(nn.Module):
    def __init__(self, proj, requires_grad=True):
        super().__init__()
        proj_tensor = torch.from_numpy(proj)
        proj_tensor = torch.log(torch.softmax(proj_tensor,dim=1) + EPS)
        self.proj = Parameter(proj_tensor)
        self.proj.requires_grad = requires_grad


    def forward(self, input):
          embedding = torch.softmax(self.proj,dim=1)
          return F.embedding(input, embedding)
