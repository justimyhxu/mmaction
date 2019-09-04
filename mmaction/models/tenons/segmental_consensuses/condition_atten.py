import torch
import torch.nn as nn
import torch.nn.functional as F
from ...registry import SEGMENTAL_CONSENSUSES

@SEGMENTAL_CONSENSUSES.register_module
class Condition_Atten(nn.Module):
    def __init__(self, time_scales, in_channels, time_dim, num_classes, init_std=0.01):
        super(Condition_Atten, self).__init__()
        self.module_dict = nn.ModuleDict()
        self.time_scales = time_scales
        self.init_std = init_std
        self.relu = nn.ReLU(inplace=True)
        for scale in time_scales:
            self.module_dict['conv1x1_{}'.format(scale)] = nn.Conv1d(2*num_classes, 512, scale)
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
        # pos_encoding [bs, num_segment, 60]
        pos_encoding = self.time_embed2(self.relu(self.time_embed1(pos_encoding.float())))
        pos_encoding = torch.sigmoid(pos_encoding)

        # feature_logit [bs, num_segment, num_class]
        cat_feat = torch.cat([pos_encoding, feature_logit], dim=2)
        # swap channel
        cat_feat = cat_feat.permute(0,2,1)

        # bs, 512, time
        multi_scale_feat = torch.cat([self.relu(self.module_dict['conv1x1_{}'.format(scale)](cat_feat))  for scale in self.time_scales], dim = 2)

        _c_multi_scale_feat = multi_scale_feat
        # bs, 512, time
        # bs, 512
        multi_scale_feat = multi_scale_feat.sum(dim=-1)
        # bs, time
        atten_logit = self.embed1(multi_scale_feat)
        # bs, time
        atten_weight = F.softmax(atten_logit, dim=1)
        # aplly matrix multiply
        embed_feat = torch.bmm(_c_multi_scale_feat, atten_weight.unsqueeze(-1)).squeeze()
        # generate logit
        init_logit = self.embed2(embed_feat)

        # skip connection
        t_logit = torch.sigmoid(self.embed3(pos_encoding).squeeze().mean(dim=1, keepdim=True))
        feature_last_logit = feature_logit[:,-1,:]
        ps = t_logit * feature_last_logit + init_logit

        h_ps = self.relu(self.embed4(ps))
        final_logit = self.embed5(h_ps)

        return init_logit, final_logit








