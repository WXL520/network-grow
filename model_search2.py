"""
1.cell的输出是concat还是最后一个节点？目前是设置为最后一个节点，
但是加入新的cell时，新cell的输入可以是任意前驱节点，这导致加入cell前后的不一致？
"""

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

        self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)  # 每次先加个ReLUConvBN模块好烦人啊
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self.preprocess2 = ReLUConvBN(C, C, 1, 1, 0, affine=False)
        self.preprocess3 = ReLUConvBN(C, C, 1, 1, 0, affine=False)
        self._steps = steps

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        for i in range(self._steps):
            for j in range(4 + i):  # 这里的2改成2*(len(genotypes)+1)
                stride = 1
                op = MixedOp(C, stride)
                self._ops.append(op)

    def forward(self, states, weights):
        # 输入应该改成一个seq？因为一个cell可能有很多个输入节点
        s0 = self.preprocess0(states[0])  # 这里默认每个cell只有两个输入节点，在我的场景，可能有多个吧？
        s1 = self.preprocess1(states[1])
        s2 = self.preprocess2(states[2])
        s3 = self.preprocess3(states[3])

        states = [s0, s1, s2, s3]
        offset = 0
        for i in range(self._steps):
            # 对每个内部节点，将所有前驱节点的输出加权
            s = sum(self._ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        # return torch.cat(states[-self._multiplier:], dim=1)
        return states[-2], states[-1]  # 输出改成最后一个节点？


class Learned_Cell(nn.Module):
    def __init__(self, genotype, C_prev_prev, C_prev, C):
        super(Learned_Cell, self).__init__()

        self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)
        op_names, indices = zip(*genotype.normal)
        self._compile(C, op_names, indices)

    def _compile(self, C, op_names, indices):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2  # 我想抠出来每个step的输出，怎么搞？

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 1
            op = OPS[name](C, stride, True)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1, drop_prob=0.0):
        # learned cell的输入，也是两个吗？应该是个state list吧？？？
        # 因为一个cell的输入可以是前面所有的内部节点？要改改。。。暂时先不改吧。。
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)
            s = h1 + h2
            states += [s]
        # return torch.cat([states[i] for i in self._concat], dim=1)
        return states[-2], states[-1]  # 改为输出cell内的每个节点？？？


class Network(nn.Module):

    def __init__(self, genotype, C, num_classes, layers, criterion, steps=2, multiplier=4, stem_multiplier=3):
        super(Network, self).__init__()
        # 当前的network，是不是应该继承之前的arch参数和weight参数？并且weight参数只要subnet的部分？
        # 那还不如直接传入base arch和对应的weight参数？subnet weights怎么获得？
        # 暂时先不考虑weights的继承？？？
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps

        C_curr = stem_multiplier * C  # stem_multiplier是啥意思啊？要不要先固定为16简化问题？
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),  # input channel是3，output channel为什么是3*16
            nn.BatchNorm2d(C_curr)
        )  # 把这里改成单层convnet的body吧？作为s0和s1
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C  # 当前dim为什么是16啊？prev dim为什么是16*3啊？估计是根据经验设置的

        # 在这里构建genotype对应的arch，只要把stem作为前驱节点，就要有个pre操作来保证channel一致，挺烦人
        # 应该用一个class，来输出每个固定cell的所有节点表示？
        # 输入的genotype应该改为genotypes吧？可以有多个learned cell
        self.pre_cell = Learned_Cell(genotype, C_prev_prev, C_prev, C_curr)

        self.cell = Cell(steps, C_prev_prev, C_prev, C_curr)  # 这里steps应该改成2*(len(genotypes)+1)???

        self.global_pooling = nn.AdaptiveAvgPool2d(1)  # 这里应该用convnet中的reshape代替吗？
        self.classifier = nn.Linear(C, num_classes)

        self._initialize_alphas()

    # def new(self):
    #   model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
    #   for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
    #       x.data.copy_(y.data)
    #   return model_new

    # def pre_arch(self, arch_param, ):
    #     # 从input到所有固定内部节点的信息流
    #     s0 = s1 = self.stem(input)  # 根据这个得到s2和s3？
    #
    #     # op_names, indices = zip(*genotype.normal)
    #     # concat = genotype.normal_concat

    def forward(self, input):
        s0 = s1 = self.stem(input)  # convnet的第一个layer的输出
        states = [s0, s1]
        s2, s3 = self.pre_cell(s0, s1)
        states.append(s2)
        states.append(s3)

        # s0,s1改成除input之外的cell外部的所有节点？这几个节点是不搜索的
        # 这里要得出所有固定内部节点的表示，s0,s1,s2,...,那这里就应该先从input走到cell之前，怎么搞。。
        weights = F.softmax(self.alphas_normal, dim=-1)
        s0_curr, s1_curr = self.cell(states, weights)  # 这里要改改，我希望cell有两个输出，两个节点分别输出自己的表示，而不是输出它们的平均值

        out = self.global_pooling(s1_curr)  # 最终的输出是第二个节点的输出？或者是最后两个节点的平均值？还是搞成可选的？
        logits = self.classifier(out.view(out.size(0), -1))  # 输出层改成convnet形式？
        return logits

    def _loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target)

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + 2 + i))  # 应该把2替换成learned_cell的个数
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
            n = 4  # 应该改成4吧？
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(range(i + 4),  # 2改成4
                               key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
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
