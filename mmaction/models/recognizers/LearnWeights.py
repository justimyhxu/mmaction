import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class LearnWeights(nn.Module):
    def __init__(self):
        super(LearnWeights, self).__init__()
        self.lmd1 = Parameter(torch.zeros(1)*1.0, requires_grad=True)
        self.lmd2 = Parameter(torch.zeros(1)*1.0, requires_grad=True)
    def forward(self, loss1, loss2):
        loss = torch.exp(-1.0*self.lmd1)*loss1 + self.lmd1 + torch.exp(-1.0*self.lmd2)*loss2 + self.lmd2
        return loss
