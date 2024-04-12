import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dimension import *
import math
import copy
import torch.autograd.variable as V


class minEntropyLoss(nn.Module):
    def __init__(self, weight=0.1, interval=8):
        super(minEntropyLoss, self).__init__()
        self.weight = weight
        self.criterion = nn.CrossEntropyLoss()
        self.interval = interval

    def forward(self, input, target, alpha=0, beta=0, epoch=1, first_cell=False):
        self.update(epoch)
        # loss1 = F.cross_entropy(input, target)
        loss1 = self.criterion(input, target)
        Alpha_normal = F.softmax(streng_func(alpha[0]).cuda(), dim=-1)  # streng_func有啥用啊？？
        normal_entLoss = torch.sum(torch.mul(Alpha_normal, torch.log(Alpha_normal)).cuda())

        loss2 = -normal_entLoss * 3  #

        w = 0.2  # 要调试一下？？？
        nor_dis_loss = 0
        nor_ent_loss = 0
        beta1 = beta[0]
        if first_cell:
            Beta = streng_func2(beta1[0])
            nor_dis_loss += w * torch.pow((torch.sum(Beta[Beta > 0]) - 2), 2)
            Beta = F.softmax(Beta.cuda(), dim=-1)
            nor_ent_loss += torch.sum(torch.mul(Beta, torch.log(Beta)).cuda())
            loss3 = (nor_dis_loss - nor_ent_loss)
        else:
            for i in range(2):
                Beta = streng_func2(beta1[i])
                nor_dis_loss += w * torch.pow((torch.sum(Beta[Beta > 0]) - 2), 2)
                Beta = F.softmax(Beta.cuda(), dim=-1)
                nor_ent_loss += torch.sum(torch.mul(Beta, torch.log(Beta)).cuda())
            loss3 = (nor_dis_loss - nor_ent_loss)
        loss3 = loss3 * 3
        # print(f'loss1: {loss1}')
        # print(f'loss2: {loss2}')
        # print(f'loss3: {loss3}')
        # print(f'nor_dis_loss: {nor_dis_loss}')
        # 这几个权重的变化趋势，也打印出来看看？？loss1，loss2，loss3也打印出来看看
        loss = loss1 + self.weight * self.weight1 * (self.weight2 * loss2 + 4 * loss3)  # 这个4是怎么回事？系数设为定值？
        # print(f'weighted loss2: {self.weight * self.weight1 * self.weight2 * loss2}')
        # print(f'weighted loss3: {self.weight * self.weight1 * 4 * loss3}')
        return loss, self.weight * self.weight1 * self.weight2 * loss2, self.weight * self.weight1 * 4 * loss3

    def update(self, epoch):
        self.weight1 = linear(epoch)
        self.weight2 = log_(epoch)


def streng_func(t):
    x = 2 * t
    mask1 = (x < -1).float().cuda()  # <-1的位置置1，其余置0
    mask2 = (x >= -1).float().cuda() + (x < 1).float().cuda() - 1  # >-1和<1的位置置1，其余置0
    mask3 = (x >= 1).float().cuda()  # >1的位置置1，其余置0
    x1 = torch.mul(mask1, torch.pow(x, 3))
    x2 = torch.mul(mask2, x)
    x3 = torch.mul(mask3, x)
    return x1 + x2 + x3  # 仅仅把<-1的位置替换为3次方，为啥啊？
    # return t


def streng_func2(t):
    x = t
    mask1 = (x < 1).float().cuda()  # <1的位置置1，>1的位置置0
    mask2 = (x >= 1).float().cuda()  # 相反
    x1 = torch.mul(mask1, x).cuda()  # >1的位置替换为0
    x2 = torch.mul(mask2, 1).cuda()
    return x1 + x2  # 仅仅是把>1的值替换为1？图啥啊？？？
    # return t
