import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype


class MixedOp(nn.Module):

    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))


class Cell(nn.Module):

    def __init__(self, steps, C_prev_prev, C_prev, C):
        super(Cell, self).__init__()
        # self.reduction = reduction

        # if reduction_prev:
        #   self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        # else:
        #   self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)  # 每次先加个ReLUConvBN模块好烦人啊
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        # self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 1
                op = MixedOp(C, stride)
                self._ops.append(op)

    def forward(self, s0, s1, weights):
        # 输入应该改成一个seq？因为一个cell可能有很多个输入节点
        s0 = self.preprocess0(s0)  # 这里默认每个cell只有两个输入节点，在我的场景，可能有多个吧？
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):  # step应该为2
            # 对每个内部节点，将所有前驱节点的输出加权
            s = sum(self._ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        # return torch.cat(states[-self._multiplier:], dim=1)
        return states[-2], states[-1]  # 输出改成最后一个节点？


class Network(nn.Module):

    def __init__(self, C, num_classes, layers, criterion, steps=2, multiplier=4, stem_multiplier=3):
        super(Network, self).__init__()
        # 当前的network，是不是应该继承之前的arch参数和weight参数？并且weight参数只要subnet的部分？
        # 那还不如直接传入base arch和对应的weight参数？subnet weights怎么获得？
        # 暂时先不考虑weights的继承？？？
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        # self._multiplier = multiplier

        C_curr = stem_multiplier * C  # stem_multiplier是啥意思啊？
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),  # input channel是3，output channel为什么是3*16
            nn.BatchNorm2d(C_curr)
        )  # 把这里改成单层convnet的body吧？作为s0和s1

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C  # 当前dim为什么是16啊？prev dim为什么是16*3啊？
        self.cell = Cell(steps, C_prev_prev, C_prev, C_curr)  # cell内部都是dim=16？

        # self.cells = nn.ModuleList()  # module就行了，没有list
        # reduction_prev = False
        # for i in range(layers):  # 每个grow step应该是只有一个cell，所以这里没有循环？
        #   if i in [layers//3, 2*layers//3]:  # 我应该没这个问题，两个节点狗策
        #     C_curr *= 2
        #     reduction = True
        #   else:
        #     reduction = False
        #   cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
        #   reduction_prev = reduction
        #   self.cells += [cell]
        #   C_prev_prev, C_prev = C_prev, multiplier*C_curr  # multiplier=4是啥意思啊？？？应该是cat4个内部节点作为输出导致的

        self.global_pooling = nn.AdaptiveAvgPool2d(1)  # 这里应该用convnet中的reshape代替吗？
        self.classifier = nn.Linear(C, num_classes)

        self._initialize_alphas()

    # def new(self):
    #   model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
    #   for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
    #       x.data.copy_(y.data)
    #   return model_new

    def forward(self, input):
        s0 = s1 = self.stem(input)  # convnet的第一个layer的输出
        # for i, cell in enumerate(self.cells):
        #     if cell.reduction:
        #         weights = F.softmax(self.alphas_reduce, dim=-1)
        #     else:
        #         weights = F.softmax(self.alphas_normal, dim=-1)
        #     s0, s1 = s1, cell(s0, s1, weights)
        weights = F.softmax(self.alphas_normal, dim=-1)
        s0, s1 = self.cell(s0, s1, weights)  # 这里要改改，我希望cell有两个输出，两个节点分别输出自己的表示，而不是输出它们的平均值

        out = self.global_pooling(s1)  # 最终的输出是第二个节点的输出？或者是最后两个节点的平均值？还是搞成可选的？
        logits = self.classifier(out.view(out.size(0), -1))  # 输出层改成convnet形式？
        return logits

    def _loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target)

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(PRIMITIVES)

        self.alphas_normal = Variable(1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)
        self._arch_parameters = [
            self.alphas_normal,
        ]

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):

        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(range(i + 2),  # input edges中，weights最大的两个
                               key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[
                        :2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene

        gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())

        genotype = Genotype(normal=gene_normal)
        # concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        # genotype = Genotype(normal=gene_normal, normal_concat=concat)  # normal_concat是不是不需要了？？？
        return genotype
