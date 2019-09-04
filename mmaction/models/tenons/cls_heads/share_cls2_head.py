import torch
import torch.nn as nn
import torch.nn.functional as F
from ...registry import HEADS

@HEADS.register_module
class ShareCls2Head(nn.Module):
    """Simplest classification head"""

    def __init__(self,
                 with_avg_pool=True,
                 temporal_feature_size=1,
                 spatial_feature_size=7,
                 dropout_ratio=0.8,
                 in_channels=2048,
                 hid_channels=512,
                 noun_num_classes=101,
                 verb_num_classes=101,
		         init_std=0.01):

        super(ShareCls2Head, self).__init__()

        self.with_avg_pool = with_avg_pool
        self.dropout_ratio = dropout_ratio
        self.in_channels = in_channels
        self.dropout_ratio = dropout_ratio
        self.temporal_feature_size = temporal_feature_size
        self.spatial_feature_size = spatial_feature_size
        self.init_std = init_std

        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool3d((temporal_feature_size, spatial_feature_size, spatial_feature_size))
        self.share_fc1 = nn.Linear(in_channels, hid_channels)
        self.bn1 = nn.BatchNorm1d(hid_channels)
        self.dropout = nn.Dropout()
        self.share_fc2 = nn.Linear(hid_channels, hid_channels)
        self.bn2 = nn.BatchNorm1d(hid_channels)
        self._dropout = nn.Dropout()
        self.fc_noun_cls = nn.Linear(hid_channels, noun_num_classes)
        self.fc_verb_cls = nn.Linear(hid_channels, verb_num_classes)
        self.relu = nn.ReLU(inplace=True)
    def init_weights(self):
        nn.init.normal_(self.fc_noun_cls.weight, 0, self.init_std)
        nn.init.constant_(self.fc_noun_cls.bias, 0)

        nn.init.normal_(self.fc_verb_cls.weight, 0, self.init_std)
        nn.init.constant_(self.fc_verb_cls.bias, 0)

        nn.init.normal_(self.share_fc1.weight, 0, self.init_std)
        nn.init.constant_(self.share_fc1.bias, 0)

        nn.init.normal_(self.share_fc2.weight, 0, self.init_std)
        nn.init.constant_(self.share_fc2.bias, 0)

    def forward(self, x):
        if x.ndimension() == 4:
            x = x.unsqueeze(2)
        assert x.shape[1] == self.in_channels
        assert x.shape[2] == self.temporal_feature_size
        assert x.shape[3] == self.spatial_feature_size
        assert x.shape[4] == self.spatial_feature_size
        if self.with_avg_pool:
            x = self.avg_pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.bn1(self.share_fc1(x)))
        # x = self._dropout(x)
        x = self.relu(self.bn2(self.share_fc2(x)))

        noun_cls_score = self.fc_noun_cls(x)
        verb_cls_score = self.fc_verb_cls(x)
        return verb_cls_score, noun_cls_score

    def loss(self,
             cls_score,
             labels, name):
        return F.cross_entropy(cls_score, labels)
        #
        # return losses
