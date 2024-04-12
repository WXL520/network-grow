"""
PC-Darts版本
"""
import os
import sys
import time
import glob
import numpy as np
import torch
import utils
# import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model_search5 import Network, SubNetwork
from architect import Architect
from genotypes import Genotype

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--subnet_batch_size', type=int, default=96, help='subnet batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.0, help='min learning rate')
parser.add_argument('--subnet_learning_rate', type=float, default=0.025, help='subnet init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--subnet_epochs', type=int, default=100, help='num of training epochs for subnet')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--subnet_init_channels', type=int, default=16, help='num of init channels for subnet')
parser.add_argument('--layers', type=int, default=1, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=4, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=6e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')

# parser.add_argument('--temperature', type=float, default=5.0,
#                     help='initial softmax temperature')
# parser.add_argument('--temperature_min', type=float, default=0.001,
#                     help='minimal softmax temperature')
parser.add_argument('--max_grow_steps', type=int, default=10, help='max grow steps from a small arch')
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
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    print(f'args: {args}')
    # logging.info('gpu device = %d' % args.gpu)
    # logging.info("args = %s", args)

    # 关于dataset和loss定义的代码，应该可以在不同的grow step共用吧？optimizer，schedule和model相关，不能共用
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

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

    genotypes = []
    # genotypes = [Genotype(normal=[('dil_conv_5x5', 1), ('max_pool_3x3', 0), ('avg_pool_3x3', 2), ('dil_conv_5x5', 0)]),
    #              Genotype(normal=[('sep_conv_5x5', 3), ('sep_conv_5x5', 2), ('skip_connect', 3), ('max_pool_3x3', 4)]),
    #              Genotype(normal=[('dil_conv_3x3', 3), ('sep_conv_3x3', 2), ('dil_conv_3x3', 2), ('sep_conv_3x3', 6)]),
    #              Genotype(normal=[('dil_conv_3x3', 4), ('dil_conv_5x5', 6), ('dil_conv_5x5', 8), ('avg_pool_3x3', 4)]),
    #              Genotype(normal=[('sep_conv_5x5', 7), ('sep_conv_5x5', 8), ('dil_conv_3x3', 10), ('sep_conv_3x3', 8)])]

    for i in range(args.max_grow_steps):
        # if i == 0:
        #     subnet_acc = train_searched_model(genotypes)
        #     print(f'first subnet acc: {subnet_acc}')

        # 在每个grow step，都将上一个step得到的genotype作为输入，用于构建固定arch
        print(f'grow step {i}')
        genotypes = search_curr_genotype(genotypes, train_queue, valid_queue, criterion)
        print(f'searched genotypes: {genotypes}')

        # 新得到的genotypes好不好，需要构建model并训练来进行验证，如果好就继续grow？如果不好呢，返回上个step重新搜？
        subnet_acc = train_searched_model(genotypes)
        print(f'current subnet acc: {subnet_acc}')


def search_curr_genotype(genotypes, train_queue, valid_queue, criterion):
    model = Network(genotypes, args.init_channels, CIFAR_CLASSES, args.layers, criterion)
    if genotypes:
        pretrained_model_dict = torch.load('last_subnet_weights_2.pt')
        curr_model_dict = model.state_dict()
        new_dict = {k: v for k, v in pretrained_model_dict.items() if k in curr_model_dict.keys()}
        curr_model_dict.update(new_dict)
        print('Total : {}, update: {}'.format(len(pretrained_model_dict), len(new_dict)))
        print(f'pretrained: {pretrained_model_dict.keys()}')
        model.load_state_dict(curr_model_dict)
        print("loading finished!")
    model = model.cuda()
    print('=' * 100)
    for k, v in model.state_dict().items():
        print(k, v.shape)
    # logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(  # 这是weights的optimizer吧？
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    # temp_scheduler = Temp_Scheduler(args.epochs, model._temp, args.temperature, temp_min=args.temperature_min)

    architect = Architect(model, args)  # 架构参数优化器

    for epoch in range(args.epochs):
        lr = scheduler.get_lr()[0]
        # model._temp = temp_scheduler.step()
        # print(f'temperature: {model._temp}')
        # logging.info('epoch %d lr %e', epoch, lr)

        genotype = model.genotype()
        # # logging.info('genotype = %s', genotype)
        print('genotype = %s', genotype)

        # print(F.softmax(model.alphas_normal, dim=-1))

        # training
        train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, epoch)
        # logging.info('train_acc %f', train_acc)
        print(f'train_acc: {train_acc}')

        # validation
        if args.epochs - epoch <= 1:
            valid_acc, valid_obj = infer(valid_queue, model, criterion)
            # logging.info('valid_acc %f', valid_acc)
            print(f'valid_acc: {valid_acc}')

        scheduler.step()

        # utils.save(model, os.path.join(args.save, 'weights.pt'))  # 没有early stop吗？保存最后一个model和genotype？？？

    return genotypes + [genotype]


def train_searched_model(genotypes):
    model = SubNetwork(args.subnet_init_channels, CIFAR_CLASSES, args.layers, genotypes)
    model = model.cuda()
    print('*' * 100)
    for k, v in model.state_dict().items():
        print(k, v.shape)
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
        # logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
        # model.drop_path_prob = args.drop_path_prob * epoch / args.subnet_epochs

        train_acc, train_obj = subnet_train(train_queue, model, criterion, optimizer)
        # logging.info('subnet_train_acc %f', train_acc)

        valid_acc, valid_obj = subnet_infer(valid_queue, model, criterion)
        # logging.info('subnet_valid_acc %f', valid_acc)

        if valid_acc > best_val_acc:
            best_val_acc = valid_acc
            # torch.save(model.state_dict(), 'last_subnet_weights_2.pt')

        # utils.save(model, os.path.join(args.save, 'weights.pt'))

    return best_val_acc


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, epoch):
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

        # if epoch >= 15:
        architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)

        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)  # 导致两次运行结果不同
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        # if step % args.report_freq == 0:
        #     # logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
        #     print('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        # input = Variable(input, volatile=True).cuda()
        # target = Variable(target, volatile=True).cuda(non_blocking=True)
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
        nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
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
