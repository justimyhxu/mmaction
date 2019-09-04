import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_op(in_planes, out_planes, kernel_size, stride=1, padding=0, bias=False):
    return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)


class conv3x1x1_dcn_warp(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=1, padding=1, bias=False, offset_kernel_1x1=False):
        super(conv3x1x1_dcn_warp, self).__init__()
        if not offset_kernel_1x1:
            self.conv_offset = conv_op(in_planes, 4, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        else:
            self.conv_offset = conv_op(in_planes, 4, kernel_size=(3,1,1), stride=(1,1,1), padding=(1,0,0), bias=False)

        self.warp_conv = conv_op(in_planes*3, out_planes, kernel_size=(1,1,1))

    def grid_sample(self, input, grid):
        '''
        :param input: size(Bs, C, T, H, W)
        :param grid: size(Bs, 2, T, H, W)
        :return:
        '''
        Bs, C, T, H ,W = input.shape

        # input: size(Bs, T, C, H, W)
        input = input.permute(0, 2, 1, 3, 4)
        # grid:  size(Bs, T, H, W, 2)
        grid = grid.permute(0, 2, 3, 4, 1)

        # input: size(Bs*T, C, H, W)
        input = input.contiguous().view(-1, C, H, W)
        # grid: size(Bs*T, H, W, 2)
        grid = grid.contiguous().view(-1, H, W, 2)

        x_ = torch.arange(W).view(1, -1).expand(H, -1)
        y_ = torch.arange(H).view(-1, 1).expand(-1, W)
        grid_ = torch.stack([x_, y_], dim=2).float().cuda()
        grid_ = grid_.unsqueeze(0).expand(Bs*T, -1, -1, -1)
        grid = grid_ + grid
        grid_x = 2 * grid[:, :, :, 0]/ (W-1) -1
        grid_y = 2 * grid[:, :, :, 1]/ (H-1) -1
        sam_grid = torch.stack([grid_x, grid_y], dim=3)

        # sam_feat:(Bs*T, C, H, W)
        sam_feat = F.grid_sample(input, sam_grid, padding_mode='border')

        sam_feat = sam_feat.view(Bs, T, C, H, W)
        sam_feat = sam_feat.permute(0, 2, 1, 3, 4)
        return sam_feat

    def forward(self, input):
        # offset_size: [bs, 4, T, H, W]
        offset = self.conv_offset(input)

        forward_offset = offset[:,:2,...]
        backward_offset = offset[:, 2:,...]

        forward_feat = self.grid_sample(input, grid=forward_offset)
        backward_feat = self.grid_sample(input, grid=backward_offset)
        original_feat = input
        cat_feat = torch.cat([forward_feat, original_feat, backward_feat], dim=1)

        warp_feat = self.warp_conv(cat_feat)
        return warp_feat

def test():
    input = torch.randn(10, 512, 8, 14, 14)
    conv = conv3x1x1_dcn_warp(in_planes=512, out_planes=1024)
    cuda_conv = conv.cuda()
    input = input.cuda()
    output = cuda_conv(input)
    print('input_size: {}'.format(input.size()))
    print('output_size: {}'.format(output.size()))

if __name__ == '__main__':
    test()