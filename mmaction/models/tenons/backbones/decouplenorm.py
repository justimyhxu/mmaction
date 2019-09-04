import torch
import torch.nn as nn

from torch.nn import Parameter


__all__ = ['TemporalNorm', 'SpatialNorm']

class TemporalNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5,
                 momentum=0.1, affine=True,
                 track_running_stats=True
                 ):
        super(TemporalNorm, self).__init__()

        self.num_features = num_features
        self.eps = eps

        self.momentum = momentum
        self.affine = affine

        self.track_running_stats = track_running_stats

        #tensor_shape = (1, num_features, 1, 1, 1)
        tensor_shape = num_features

        if self.affine:
            self.weight = Parameter(torch.Tensor(tensor_shape))
            self.bias = Parameter(torch.Tensor(tensor_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        if self.track_running_stats:
            #     self.register_buffer('running_mean', torch.zeros(*tensor_shape))
            #     self.register_buffer('running_var', torch.ones(*tensor_shape))
            # else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
        self.reset_parameters()

    def forward(self, x):
        N, C, T, H, W = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous() # N T C H W
        x = x.view(-1, C, H, W)

        mean = x.mean(0, keepdim=True)
        var = x.var(0, keepdim=True)

        if self.running_mean is None or self.running_mean.size() != mean.size():
            self.running_mean = Parameter(mean.data)
            self.running_var = Parameter(var.data)

        if self.training and self.track_running_stats:
            self.running_mean.data = mean * self.momentum + \
                                     self.running_mean.data * (1 - self.momentum)
            self.running_var.data = var * self.momentum + \
                                    self.running_var.data * (1 - self.momentum)


        x = (x - self.running_mean) / (self.running_var + self.eps).sqrt()
        x = x.view(N, T, C, H, W).permute(0, 2, 1, 3, 4).contiguous() # N C T H W
        return x * self.weight.view(1,-1,1,1,1) + self.bias.view(1,-1,1,1,1)

    def reset_parameters(self):
        if self.track_running_stats:
            if self.running_mean is not None and self.running_var is not None:
                self.running_mean.zero_()
                self.running_var.fill_(1)
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def __repr__(self):
        return ('{name}({num_features}, eps={eps}, momentum={momentum},'
                ' affine={affine}, track_running_stats={track_running_stats})'
                .format(name=self.__class__.__name__, **self.__dict__))


class SpatialNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5,
                 momentum=0.1, affine=True,
                 track_running_stats=True
                 ):
        super(SpatialNorm, self).__init__()

        self.num_features = num_features
        self.eps = eps

        self.momentum = momentum
        self.affine = affine

        self.track_running_stats = track_running_stats

        #tensor_shape = (1, num_features, 1, 1, 1)
        tensor_shape = num_features

        if self.affine:
            self.weight = Parameter(torch.Tensor(tensor_shape))
            self.bias = Parameter(torch.Tensor(tensor_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        if self.track_running_stats:
            #     self.register_buffer('running_mean', torch.zeros(*tensor_shape))
            #     self.register_buffer('running_var', torch.ones(*tensor_shape))
            # else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
        self.reset_parameters()

    def forward(self, x):
        N, C, T, H, W = x.size()
        x = x.permute(1, 2, 3, 4, 0).contiguous() # C, T, H, W, N
        x = x.view(C, T, -1)

        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        if self.running_mean is None or self.running_mean.size() != mean.size():
            self.running_mean = Parameter(mean.data)
            self.running_var = Parameter(var.data)

        if self.training and self.track_running_stats:
            self.running_mean.data = mean * self.momentum + \
                                     self.running_mean.data * (1 - self.momentum)
            self.running_var.data = var * self.momentum + \
                                    self.running_var.data * (1 - self.momentum)


        x = (x - self.running_mean) / (self.running_var + self.eps).sqrt()
        x = x.view(C, T, H, W, N).permute(4, 0, 1, 2, 3).contiguous() # N C T H W
        return x * self.weight.view(1,-1,1,1,1) + self.bias.view(1,-1,1,1,1)

    def reset_parameters(self):
        if self.track_running_stats:
            if self.running_mean is not None and self.running_var is not None:
                self.running_mean.zero_()
                self.running_var.fill_(1)
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def __repr__(self):
        return ('{name}({num_features}, eps={eps}, momentum={momentum},'
                ' affine={affine}, track_running_stats={track_running_stats})'
                .format(name=self.__class__.__name__, **self.__dict__))

def main():
   norm = TemporalNorm(32).cuda()
   input = torch.FloatTensor(8, 32, 8, 56, 56).cuda()
   out = norm(input)
   print(out.shape)
   norm = SpatialNorm(32).cuda()
   out = norm(input)
   print(out.shape)
if __name__ == '__main__':
   main()
