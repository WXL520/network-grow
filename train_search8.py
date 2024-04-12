"""
PC-Darts版本
"""
import os
import sys
import time
import glob
import re
import numpy as np
import torch
import utils
# import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
# import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model_search8 import Network, SubNetwork
from architect8 import Architect
from genotypes import Genotype, PRIMITIVES
from MinEntropyLoss import minEntropyLoss, streng_func, streng_func2

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')  # 改成更大？？？
parser.add_argument('--subnet_batch_size', type=int, default=64, help='subnet batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
# parser.add_argument('--aux_weight', type=float, default=0.1, help='')
# parser.add_argument('--interval', type=float, default=8, help='')
parser.add_argument('--subnet_learning_rate', type=float, default=0.025, help='subnet init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--subnet_epochs', type=int, default=20, help='num of training epochs for subnet')  # 设为20会导致无法产生one-hot，two-hot分布
parser.add_argument('--init_channels', type=int, default=32, help='num of init channels')
parser.add_argument('--subnet_init_channels', type=int, default=32, help='num of init channels for subnet')
parser.add_argument('--layers', type=int, default=1, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_lr_alpha', type=float, default=3e-4, help='')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')

# parser.add_argument('--temperature', type=float, default=1.0,
#                     help='initial softmax temperature')
# parser.add_argument('--temperature_min', type=float, default=0.005,
#                     help='minimal softmax temperature')
parser.add_argument('--max_grow_steps', type=int, default=1, help='max grow steps from a small arch')
args = parser.parse_args()

# args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
# utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

# log_format = '%(asctime)s %(message)s'
# logging.basicConfig(stream=sys.stdout, level=logging.INFO,
#     format=log_format, datefmt='%m/%d %I:%M:%S %p')
# fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
# fh.setFormatter(logging.Formatter(log_format))
# logging.getLogger().addHandler(fh)


CIFAR_CLASSES = 10


def main():
    # if not torch.cuda.is_available():
    #     # logging.info('no gpu device available')
    #     print('no gpu device available')
    #     sys.exit(1)

    np.random.seed(args.seed)
    # torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    print(f'args: {args}')
    # logging.info('gpu device = %d' % args.gpu)
    # logging.info("args = %s", args)

    # genotypes = []
    genotypes = [Genotype(normal=[('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 2), ('sep_conv_3x3', 1)]),
                 Genotype(normal=[('sep_conv_3x3', 2), ('dil_conv_3x3', 3), ('sep_conv_5x5', 4), ('sep_conv_3x3', 3)]),
                 Genotype(normal=[('sep_conv_5x5', 4), ('skip_connect', 5), ('sep_conv_5x5', 6), ('sep_conv_5x5', 3)]),
                 Genotype(normal=[('sep_conv_3x3', 6), ('sep_conv_3x3', 7), ('dil_conv_5x5', 8), ('sep_conv_5x5', 7)]),
                 Genotype(normal=[('sep_conv_3x3', 9), ('sep_conv_5x5', 7), ('dil_conv_3x3', 9), ('dil_conv_3x3', 10)])]
    # genotypes = [Genotype(normal=[('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2), ('sep_conv_3x3', 1)]),
    #              Genotype(normal=[('sep_conv_3x3', 2), ('sep_conv_5x5', 3), ('sep_conv_5x5', 4), ('sep_conv_5x5', 3)]),
    #              Genotype(normal=[('sep_conv_3x3', 4), ('dil_conv_5x5', 5), ('dil_conv_3x3', 6), ('sep_conv_5x5', 3)])]
    # genotypes = [Genotype(normal=[('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2), ('sep_conv_3x3', 1)]),
    #              Genotype(normal=[('sep_conv_3x3', 2), ('sep_conv_5x5', 3), ('sep_conv_5x5', 4), ('sep_conv_5x5', 3)]),
    #              Genotype(normal=[('sep_conv_5x5', 4), ('sep_conv_3x3', 5), ('dil_conv_3x3', 6), ('sep_conv_5x5', 3)]),
    #              Genotype(normal=[('sep_conv_3x3', 6), ('dil_conv_5x5', 7), ('sep_conv_3x3', 8), ('sep_conv_5x5', 7)])]
    # genotypes = [Genotype(normal=[('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2), ('sep_conv_3x3', 1)]),
    #              Genotype(normal=[('sep_conv_3x3', 2), ('dil_conv_3x3', 3), ('sep_conv_5x5', 4), ('sep_conv_3x3', 3)]),
    #              Genotype(normal=[('sep_conv_5x5', 4), ('dil_conv_5x5', 5), ('sep_conv_5x5', 6), ('sep_conv_5x5', 3)]),
    #              Genotype(normal=[('sep_conv_3x3', 6), ('sep_conv_3x3', 7), ('sep_conv_3x3', 8), ('sep_conv_5x5', 7)])]
    # genotypes = [Genotype(normal=[('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2), ('sep_conv_3x3', 1)])]

    t0 = time.time()
    first_subnet_acc = train_searched_model(genotypes)
    print(f'subnet time: {time.time() - t0}')
    print(f'first subnet acc: {first_subnet_acc}')
    for i in range(args.max_grow_steps):
        # 在每个grow step，都将上一个step得到的genotype作为输入，用于构建固定arch
        print(f'grow step {i}')
        t1 = time.time()
        genotypes = search_curr_genotype(genotypes)
        print(f'search time: {time.time() - t1}')
        print(f'searched genotypes: {genotypes}')

        # 新得到的genotypes好不好，需要构建model并训练来进行验证，如果好就继续grow？如果不好呢，返回上个step重新搜？
        t3 = time.time()
        subnet_acc = train_searched_model(genotypes, inheritance=True)  # 试试不retrain的结果
        print(f'final subnet time: {time.time() - t3}')
        print(f'current subnet acc: {subnet_acc}')


def search_curr_genotype(genotypes):
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    model = Network(genotypes, args.init_channels, CIFAR_CLASSES, args.layers, criterion)
    init_state_dict = model.state_dict()

    # inherit weights from last subnet
    if len(genotypes) > 0:
        pretrained_model_dict = torch.load('subnet_weights11.pt')
        curr_model_dict = model.state_dict()  # 这里包不包括架构参数啊？？？
        # 只继承learnable参数
        # reuse_dict = {k: v for k, v in pretrained_model_dict.items()
        #               if k in curr_model_dict.keys() and 'running' not in k and 'tracked' not in k}
        reuse_dict = {k: v for k, v in pretrained_model_dict.items()
                      if k in curr_model_dict.keys()}
        cls_weights = {k[7:]: v for k, v in pretrained_model_dict.items() if 'classifier' in k}
        reuse_dict.update(cls_weights)
        print(f'reuse: {len(reuse_dict)}')

        # 对每个op，将subnet中最后出现的该op的参数复制给supernet中该op的新参数
        op_list = []
        for genotype in genotypes:
            op_names, indices = zip(*genotype.normal)
            op_list.append(op_names)
        for i in range(len(op_list)):
            for j in range(len(op_list[i])):
                if op_list[i][j] == 'sep_conv_3x3':
                    sep3_key_prefix = f'pre_cells.{i}._ops.{j}'
                elif op_list[i][j] == 'sep_conv_5x5':
                    sep5_key_prefix = f'pre_cells.{i}._ops.{j}'
                elif op_list[i][j] == 'dil_conv_3x3':
                    dil3_key_prefix = f'pre_cells.{i}._ops.{j}'
                elif op_list[i][j] == 'dil_conv_5x5':
                    dil5_key_prefix = f'pre_cells.{i}._ops.{j}'

        cell_weights = {}
        for k, v in curr_model_dict.items():
            if k[:4] == 'cell': #and 'running' not in k and 'tracked' not in k:
                op_key = re.findall('\d+', k)[1]
                op = PRIMITIVES[int(op_key)]
                if op == 'sep_conv_3x3':
                    sub_k = sep3_key_prefix + k[18:]  # 这个12后面也要改改
                    cell_weights[k] = pretrained_model_dict[sub_k]
                elif op == 'sep_conv_5x5':
                    sub_k = sep5_key_prefix + k[18:]
                    cell_weights[k] = pretrained_model_dict[sub_k]
                elif op == 'dil_conv_3x3':
                    sub_k = dil3_key_prefix + k[18:]
                    cell_weights[k] = pretrained_model_dict[sub_k]
                elif op == 'dil_conv_5x5':
                    sub_k = dil5_key_prefix + k[18:]
                    cell_weights[k] = pretrained_model_dict[sub_k]

        # # 先试试仅对new param进行shrink-perturb初始化
        # for k, v in cell_weights.items():
        #     rand_v = torch.randn_like(v, dtype=torch.float)  # 是不是得比较下origin param和rand param的幅度？
        #     # print(v)
        #     # print('='*100)
        #     # print(rand_v)
        #     new_v = 0.4 * v + 0.1 * rand_v
        #     cell_weights[k] = new_v

        reuse_dict.update(cell_weights)

        # 试试对所有param进行shrink-perturb初始化，randn应该是不行的，应该重新初始化arch，并保存未训练的state dict
        for k, v in reuse_dict.items():
            rand_v = init_state_dict[k]
            new_v = 0.4 * v.cuda() + 0.1 * rand_v.cuda()
            reuse_dict[k] = new_v

        print(f'new reuse: {len(reuse_dict)}')
        print(f'curr dict: {len(curr_model_dict)}')
        for k in curr_model_dict.keys():
            if k not in reuse_dict:
                print(k)

        curr_model_dict.update(reuse_dict)
        model.load_state_dict(curr_model_dict)
        print("loading finished!")

    model = model.cuda()
    # print('=' * 100)
    # for k, v in model.state_dict().items():
    #     if 'running' not in k and 'tracked' not in k:
    #         print(k, v.shape)
    # logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(  # 这是weights的optimizer吧
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=0)

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=0)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    architect = Architect(model, args)  # 架构参数优化器

    tau_max = 3
    tau_min = 0.1
    his_val_acc = -1
    for epoch in range(1, args.epochs + 1):
        lr = scheduler.get_lr()[0]
        model.set_tau(tau_max - (tau_max - tau_min) * epoch / (args.epochs - 1))  # epochs改成250？
        # logging.info('epoch %d lr %e', epoch, lr)

        genotype = model.genotype()
        # # logging.info('genotype = %s', genotype)
        print('genotype = %s', genotype)
        print(F.softmax(model.alphas_normal, dim=-1))

        # training
        train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, epoch, genotypes)
        print(f'train_acc: {train_acc}')

        valid_acc, valid_obj = infer(valid_queue, model, criterion, genotypes)
        print(f'valid_acc: {valid_acc}')

        if valid_acc > his_val_acc:
            his_val_acc = valid_acc
            torch.save(model.state_dict(), 'supernet_weights11.pt')
            new_geno = genotype

        scheduler.step()
    # torch.save(model.state_dict(), 'supernet_weights1.pt')
    # utils.save(model, os.path.join(args.save, 'weights.pt'))  # 没有early stop吗？保存最后一个model和genotype？？？

    # return genotypes + [genotype]
    return genotypes + [new_geno]


def train_searched_model(genotypes, inheritance=False):
    model = SubNetwork(args.subnet_init_channels, CIFAR_CLASSES, args.layers, genotypes)
    # 如果不继承BN layer的不可学习参数，会导致初始性能很差？？？
    if inheritance:
        # if len(genotypes) > 2:
        pretrained_supernet_weights = torch.load('supernet_weights11.pt')
        curr_subnet_weights = model.state_dict()
        # new_subnet_precell_weights = {k: v for k, v in pretrained_supernet_weights.items() if
        #                               k in curr_subnet_weights.keys() and 'running' not in k and 'tracked' not in k}
        new_subnet_precell_weights = {k: v for k, v in pretrained_supernet_weights.items() if
                                      k in curr_subnet_weights.keys()}
        last_genotype = genotypes[-1]
        op_names, indices = zip(*last_genotype.normal)
        new_pretrained_weights = {}
        for i, (name, index) in enumerate(zip(op_names, indices)):  # 在supernet中找到derived op的weights，并改名
            num1 = index - 2 * (len(genotypes) - 2) if len(genotypes) > 2 else index
            offset = 4 if len(genotypes) > 1 else 2  # # 这里的4是不是应该改改？2是steps
            num1 += offset if i >= 2 else 0
            num2 = PRIMITIVES.index(name)
            new_num1 = len(genotypes) - 1
            new_num2 = i
            for k, v in pretrained_supernet_weights.items():  # cell._ops.8._ops.7.op.1.weight
                # 给supernet的weights改名字，从cell改为pre-cell，序号也得改
                if f'cell._ops.{num1}._ops.{num2}.' in k:
                    # print(f'cell._ops.{num1}._ops.{num2}.')
                    # start = re.search('op\.', k).span()[0]  # 确实存在不含'op.'的字符串。。
                    new_k = f'pre_cells.{new_num1}._ops.{new_num2}.' + k[19:]  # 19后期要用变量替换下
                    new_pretrained_weights[new_k] = v
                elif 'classifier' in k:
                    new_k = 'subnet_' + k
                    new_pretrained_weights[new_k] = v
        new_subnet_cell_weights = {k: v for k, v in new_pretrained_weights.items() if
                                   k in curr_subnet_weights.keys()}

        # print('Total : {}, update: {}'.format(len(new_pretrained_weights), len(new_subnet_cell_weights)))
        print(f'new_subnet_precell_weights length: {len(new_subnet_precell_weights)}')
        print(f'new_subnet_cell_weights length: {len(new_subnet_cell_weights)}')
        print(f'curr_subnet_weights length: {len(curr_subnet_weights)}')
        new_subnet_precell_weights.update(new_subnet_cell_weights)

        # # 试试对所有param进行shrink-perturb初始化
        for k, v in new_subnet_precell_weights.items():
            rand_v = curr_subnet_weights[k]
            new_v = 0.4 * v.cuda() + 0.1 * rand_v.cuda()
            new_subnet_precell_weights[k] = new_v

        curr_subnet_weights.update(new_subnet_precell_weights)
        model.load_state_dict(curr_subnet_weights)
        print("loading supernet weights finished!")

    model = model.cuda()
    # print('*' * 100)
    # for k, v in model.state_dict().items():
    #     if 'running' not in k and 'tracked' not in k:
    #         print(k, v.shape)
    # logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.subnet_learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.subnet_batch_size, shuffle=True, pin_memory=True, num_workers=0)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.subnet_batch_size, shuffle=False, pin_memory=True, num_workers=0)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.subnet_epochs))

    best_val_acc = 0.
    for epoch in range(args.subnet_epochs):
        scheduler.step()
        # model.drop_path_prob = args.drop_path_prob * epoch / args.subnet_epochs

        valid_acc, valid_obj = subnet_infer(valid_queue, model, criterion)
        print(f'valid acc: {valid_acc}')

        if valid_acc > best_val_acc:
            best_val_acc = valid_acc
            torch.save(model.state_dict(), 'subnet_weights11.pt')

        train_acc, train_obj = subnet_train(train_queue, model, criterion, optimizer)

    return best_val_acc


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, epoch, genotypes):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    for step, (input, target) in enumerate(train_queue):
        model.train()
        n = input.size(0)

        # input = Variable(input, requires_grad=False).cuda()
        # target = Variable(target, requires_grad=False).cuda(non_blocking=True)
        input = input.cuda()
        target = target.cuda()

        # get a random minibatch from the search queue with replacement
        input_search, target_search = next(iter(valid_queue))
        # input_search = Variable(input_search, requires_grad=False).cuda()
        # target_search = Variable(target_search, requires_grad=False).cuda(non_blocking=True)
        input_search = input_search.cuda()
        target_search = target_search.cuda()

        # if epoch == args.interval:  # warm up 8个epoch？
        #     architect.set()
        # architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled, epoch=epoch)
        if epoch >= 0:  # 没有warm up性能更好？？
            architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled, epoch=epoch)

        optimizer.zero_grad()
        logits = model(input)
        # if genotypes:
        #     first_cell = False
        # else:
        #     first_cell = True
        # loss, loss2, loss3 = criterion(logits, target, alpha=model._arch_parameters, beta=model._beta_parameters,
        #                                epoch=epoch, first_cell=first_cell)
        loss = criterion(logits, target)  # first_cell=??
        loss.backward()
        # nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)  # 导致两次运行结果不同???
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        # if step % args.report_freq == 0:
        #     # logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
        #     print('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion, genotypes):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        # input = Variable(input, volatile=True).cuda()
        # target = Variable(target, volatile=True).cuda(non_blocking=True)
        input = input.cuda()
        target = target.cuda()

        # if tensor.dtype is torch.long:  # 这段代码备用。。。
        #     # these all are just num_batches_tracked of the batchnorm
        #     return tensor

        logits = model(input)
        # if genotypes:
        #     first_cell = False
        # else:
        #     first_cell = True
        # loss = criterion(logits, target, first_cell=first_cell)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        # objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        # if step % args.report_freq == 0:
        #     # logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
        #     print('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def subnet_train(train_queue, model, criterion, optimizer):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()

    for step, (input, target) in enumerate(train_queue):
        # input = Variable(input).cuda()
        # target = Variable(target).cuda(async=True)
        input = input.cuda()
        target = target.cuda()

        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)
        # if args.auxiliary:
        #     loss_aux = criterion(logits_aux, target)
        #     loss += args.auxiliary_weight * loss_aux
        loss.backward()
        # nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        # if step % args.report_freq == 0:
        #     logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def subnet_infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        # input = Variable(input, volatile=True).cuda()
        # target = Variable(target, volatile=True).cuda(async=True)
        input = input.cuda()
        target = target.cuda()

        logits = model(input)
        loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        # if step % args.report_freq == 0:
        #     logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


######################################################################
# Exponential annealing for softmax temperature
######################################################################
class Temp_Scheduler(object):
    def __init__(self, total_epochs, curr_temp, base_temp, temp_min=0.05, last_epoch=-1):
        self.total_epochs = total_epochs
        self.curr_temp = curr_temp
        self.base_temp = base_temp
        self.temp_min = temp_min
        self.last_epoch = last_epoch
        self.step(last_epoch + 1)

    def step(self, epoch=None):
        return self.decay_whole_process()

    def decay_whole_process(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        # self.curr_temp = (1 - self.last_epoch / self.total_epochs) * (self.base_temp - self.temp_min) + self.temp_min
        # if self.curr_temp < self.temp_min:
        #     self.curr_temp = self.temp_min

        self.curr_temp = max(self.base_temp * 0.90 ** self.last_epoch, self.temp_min)

        return self.curr_temp


if __name__ == '__main__':
    main()
