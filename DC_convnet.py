import torch.nn as nn
import torch
from torch.autograd import Variable


class FixedLinear(nn.Linear):
    def __init__(self, in_size, out_size):
        super().__init__(in_size, out_size)
        self.in_size = in_size
        self.out_size = out_size

        del self.weight
        del self.bias

        self.old_weights = torch.load('subnet_weights1.pt')
        # self.weight = old_weights['classifier.weight']
        # self.bias = old_weights['classifier.bias']

    def forward(self, inp):
        self.weight = self.old_weights['classifier.weight'].cuda()
        self.bias = self.old_weights['classifier.bias'].cuda()
        result = super().forward(inp)

        # Avoid side effects in nn.Module
        del self.weight
        del self.bias

        return result


class FixedConv2d(nn.Conv2d):
    def __init__(self, in_channels,
                 out_channels,
                 kernel_size,
                 padding=0,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        super().__init__(in_channels,
                         out_channels,
                         kernel_size,
                         padding=padding,
                         stride=stride,
                         dilation=dilation,
                         groups=groups,
                         bias=bias)
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Remove the original weights and bias
        del self.weight
        # del self.bias

        self.old_weights = torch.load('subnet_weights1.pt')
        # self.weight = old_weights['features.0.weight']

    def forward(self, input):
        self.weight = self.old_weights['features.0.weight'].cuda()
        result = super().forward(input)

        # Avoid side effects in nn.Module
        del self.weight
        # del self.bias

        return result


class WeightedConv2d(nn.Conv2d):
    def __init__(self, in_channels,
                 out_channels,
                 kernel_size,
                 padding=0,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        super().__init__(in_channels,
                         out_channels,
                         kernel_size,
                         padding=padding,
                         stride=stride,
                         dilation=dilation,
                         groups=groups,
                         bias=bias)
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Remove the original weights and bias
        del self.weight
        # del self.bias

        # Add our new parameter
        self.W = nn.Parameter(torch.rand(1, 3))
        # 应该把old weights先加载进来？然后在forward中，根据实时的w，来计算实时的weights和bias？
        old_weights = torch.load('subnet_weights1.pt')
        self.feat = torch.stack([old_weights['features.1.weight'].reshape(-1),
                                 old_weights['features.4.weight'].reshape(-1),
                                 old_weights['features.7.weight'].reshape(-1)]).cuda()
        # print(self.feat.shape)

    def forward(self, input):
        # repopulate the original weights with our custom version
        # You can do any differentiable op here
        self.weight = torch.matmul(self.W, self.feat)[0].reshape(32, 32, 3, 3)
        # self.weight[1, 1] = 0  # Mask entry 1,1 in in weight to see some difference in the gradients
        # self.bias = self.M.squeeze(-1)
        result = super().forward(input)

        # Avoid side effects in nn.Module
        del self.weight
        # del self.bias

        return result


class DC_ConvNet(nn.Module):
    def __init__(self, channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, im_size=(32, 32)):
        super(DC_ConvNet, self).__init__()
        self.features, shape_feat = self._make_layers(channel, net_width, net_depth, net_norm, net_act, net_pooling,
                                                      im_size)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = FixedLinear(net_width, num_classes)

    def forward(self, x):
        out = self.features(x)
        # out = out.reshape(out.size(0), -1)
        out = self.global_pooling(out)
        out = self.classifier(out.view(out.size(0), -1))
        return out

    # def embed(self, x):
    #     out = self.features(x)
    #     out = out.reshape(out.size(0), -1)
    #     return out
    #
    # def embed_before_pool(self, x):
    #     out = nn.Sequential(*list(self.features.children())[:-1])(x)
    #     out = out.view(out.size(0), -1)
    #     return out

    def _get_activation(self, net_act):
        if net_act == 'sigmoid':
            return nn.Sigmoid()
        elif net_act == 'relu':
            return nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.01)
        else:
            exit('unknown activation function: %s' % net_act)

    def _get_pooling(self, net_pooling):
        if net_pooling == 'maxpooling':
            return nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        elif net_pooling == 'avgpooling':
            return nn.AvgPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'identity':
            return None
        else:
            exit('unknown net_pooling: %s' % net_pooling)

    def _get_normlayer(self, net_norm, shape_feat):
        # shape_feat = (c*h*w)
        if net_norm == 'batch':
            return nn.BatchNorm2d(shape_feat[0], affine=True)
        elif net_norm == 'layer':
            return nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == 'instance':
            return nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        elif net_norm == 'group':
            return nn.GroupNorm(4, shape_feat[0], affine=True)
        elif net_norm == 'identity':
            return None
        else:
            exit('unknown net_norm: %s' % net_norm)

    def _make_layers(self, channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size):
        layers = []
        layers += nn.Sequential(
            FixedConv2d(3, net_width, 3, padding=1, bias=False)
            # nn.Conv2d(3, net_width, 3, padding=1, bias=False),
            # nn.BatchNorm2d(C_curr)
        )

        in_channels = channel
        if im_size[0] == 28:
            im_size = (32, 32)
        shape_feat = [in_channels, im_size[0], im_size[1]]
        for d in range(net_depth):
            layers += [WeightedConv2d(in_channels, net_width, kernel_size=3,
                                      padding=3 if channel == 1 and d == 0 else 1, bias=False)]
            # layers += [nn.Conv2d(in_channels, net_width, kernel_size=3, padding=3 if channel == 1 and d == 0 else 1, bias=False)]
            shape_feat[0] = net_width
            if net_norm != 'identity':
                layers += [self._get_normlayer(net_norm, shape_feat)]
            layers += [self._get_activation(net_act)]
            in_channels = net_width
            if net_pooling != 'identity':
                layers += [self._get_pooling(net_pooling)]
                shape_feat[1] //= 2
                shape_feat[2] //= 2

        return nn.Sequential(*layers), shape_feat


if __name__ == '__main__':
    in_channels = 32
    out_channels = 32

    my_layer = WeightedConv2d(in_channels, out_channels, (3, 3), bias=False)
    first_layer = FixedConv2d(3, out_channels, (3, 3), bias=False)

    inp = torch.rand(64, 32, 32, 32)
    inp2 = torch.rand(64, 32, 32, 3)
    out = my_layer(inp2)
    print(out.shape)
    out.sum().backward()

    # print(my_layer.W.grad)
