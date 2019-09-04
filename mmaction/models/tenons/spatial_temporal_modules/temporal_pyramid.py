import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import constant_init, kaiming_init
from ...registry import SPATIAL_TEMPORAL_MODULES


@SPATIAL_TEMPORAL_MODULES.register_module
class TemporalPyramidModule(nn.Module):
    def __init__(self, in_channels=1024):
        super(TemporalPyramidModule, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        self.conv2_1x1 = nn.Conv3d(in_channels, in_channels, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0),
                                   dilation=(1, 1, 1), bias=False)
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0),
                               dilation=(3, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(in_channels)

        self.conv3_1x1 = nn.Conv3d(in_channels, in_channels, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0),
                                   dilation=(1, 1, 1), bias=False)
        self.conv3 = nn.Conv3d(in_channels, in_channels, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0),
                               dilation=(2, 1, 1), bias=False)
        self.bn3 = nn.BatchNorm3d(in_channels)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                kaiming_init(m)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                constant_init(m, 0)

    def forward(self, input):
        branch2_up = F.interpolate(self.relu(self.bn2(self.conv2(input))), scale_factor=(2, 1, 1), mode='trilinear',
                                   align_corners=False)
        branch3_up = F.interpolate(self.conv2_1x1(self.relu(self.bn3(self.conv3(input)))) + branch2_up,
                                   scale_factor=(2, 1, 1), mode='trilinear', align_corners=False)
        return self.conv3_1x1(input) + branch3_up


'''
def main():
    input = torch.FloatTensor(16, 128, 8, 56,56).cuda()
    model = TemporalPyramidModule(128).cuda()
    out = model(input)

if __name__=='__main__':
   main()
'''


