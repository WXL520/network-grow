import torch.nn as nn
import torch
from torch.autograd import Variable


class DC_ConvNet(nn.Module):
    def __init__(self, channel, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, im_size=(32, 32)):
        super(DC_ConvNet, self).__init__()
        self.features, shape_feat = self._make_layers(channel, net_width, net_depth, net_norm, net_act, net_pooling,
                                                      im_size)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(net_width, num_classes)

    def forward(self, x):
        out = self.features(x)
        # out = out.reshape(out.size(0), -1)
        out = self.global_pooling(out)
        out = self.classifier(out.view(out.size(0), -1))
        return out

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
            nn.Conv2d(3, net_width, 3, padding=1, bias=False),
            # nn.BatchNorm2d(C_curr)
        )

        in_channels = channel
        if im_size[0] == 28:
            im_size = (32, 32)
        shape_feat = [in_channels, im_size[0], im_size[1]]
        for d in range(net_depth):
            layers += [nn.Conv2d(in_channels, net_width, kernel_size=3, padding=3 if channel == 1 and d == 0 else 1, bias=False)]
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
