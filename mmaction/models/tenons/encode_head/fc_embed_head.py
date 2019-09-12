import torch
import torch.nn as nn
import torch.nn.functional as F
from ...registry import HEADS

@HEADS.register_module
class fc_embed_head(nn.Module):

    def __init__(self, input_dim=90, hid_dim=256, out_dim=256, init_std=0.001, zero_bn = True):
        super(fc_embed_head, self).__init__()
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.relu = nn.ReLU(inplace=True)
        self.init_std = init_std
        self.zero_bn = zero_bn
        self.fc1 = nn.Linear(input_dim, hid_dim)
        # self.bn = nn.Ba
        self.fc2 = nn.Linear(hid_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)

    def init_weights(self):
        nn.init.normal_(self.fc1.weight, 0, self.init_std)
        nn.init.constant_(self.fc1.bias, 0)

        nn.init.normal_(self.fc2.weight, 0, self.init_std)
        nn.init.constant_(self.fc2.bias, 0)

        if self.zero_bn:
            nn.init.zeros_(self.bn.weight)
            nn.init.zeros_(self.bn.bias)

    def forward(self, x):
        x = self.fc2(self.relu(self.fc1(x)))
        x = x.permute(0,2,1)
        x = self.bn(x)
        return x