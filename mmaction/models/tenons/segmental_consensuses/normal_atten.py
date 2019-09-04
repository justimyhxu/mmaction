import torch
import torch.nn as nn
import torch.nn.functional as F
from ...registry import SEGMENTAL_CONSENSUSES

@SEGMENTAL_CONSENSUSES.register_module
class normal_Atten(nn.Module):
    def __init__(self, time_scales, in_channels, time_dim, num_classes, mode = 'full_atten', init_std=0.01):
        super(normal_Atten, self).__init__()
        self.module_dict = nn.ModuleDict()
        self.time_scales = time_scales
        self.init_std = init_std
        self.mode = mode
        self.relu = nn.ReLU(inplace=True)
        for scale in time_scales:
            self.module_dict['conv1x1_{}'.format(scale)] = nn.Conv1d(num_classes, 512, scale)
        atten_dim = sum([time_dim-scale+1  for scale in time_scales])

        self.time_embed1 = nn.Linear(60, 512)
        self.time_embed2 = nn.Linear(512, num_classes)

        self.embed1 = nn.Linear(in_channels, atten_dim)
        self.embed2 = nn.Linear(in_channels, num_classes)
        self.embed3 = nn.Linear(num_classes, 1)
        self.embed4 = nn.Linear(num_classes, in_channels)
        self.embed5 = nn.Linear(in_channels, num_classes)

    def init_weights(self):
        for scale in self.time_scales:
            nn.init.normal_(self.module_dict['conv1x1_{}'.format(scale)].weight , 0, self.init_std)
            nn.init.constant_(self.module_dict['conv1x1_{}'.format(scale)].bias, 0)
        nn.init.normal_(self.embed1.weight, 0, self.init_std)
        nn.init.constant_(self.embed1.bias, 0)

        nn.init.normal_(self.embed2.weight, 0, self.init_std)
        nn.init.constant_(self.embed2.bias, 0)

        nn.init.normal_(self.embed3.weight, 0, self.init_std)
        nn.init.constant_(self.embed3.bias, 0)

        nn.init.normal_(self.embed4.weight, 0, self.init_std)
        nn.init.constant_(self.embed4.bias, 0)

        nn.init.normal_(self.embed5.weight, 0, self.init_std)
        nn.init.constant_(self.embed5.bias, 0)

        nn.init.normal_(self.time_embed1.weight, 0, self.init_std)
        nn.init.constant_(self.time_embed1.bias, 0)

        nn.init.normal_(self.time_embed2.weight, 0, self.init_std)
        nn.init.constant_(self.time_embed2.bias, 0)

    def forward(self, pos_encoding, feature_logit):
        if self.mode == 'avg':
            logit = feature_logit.mean(1)
        else:
            # [bx, num_segment, 60]
            # _c_feature_logit = feature_logit
            feature_logit = feature_logit.permute(0,2,1)
            multi_scale_feat = torch.cat([self.relu(self.module_dict['conv1x1_{}'.format(scale)](feature_logit))  for scale in self.time_scales], dim = 2)
            _c_multi_scale_feat = multi_scale_feat
            multi_feat_logit = multi_scale_feat.sum(dim=-1)
            atten_logit = self.embed1(multi_feat_logit)
            atten_weight = F.softmax(atten_logit, dim = 1)
            embed_feat = (_c_multi_scale_feat * atten_weight.unsqueeze(1)).sum(dim=-1)
            logit  = self.embed2(embed_feat)
        return logit , []










