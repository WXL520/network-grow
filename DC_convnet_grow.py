from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as dset
import argparse

from DC_convnet import DC_ConvNet
# from DC_convnet_model import DC_ConvNet
import utils

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--subnet_batch_size', type=int, default=64, help='subnet batch size')
parser.add_argument('--subnet_epochs', type=int, default=100, help='num of training epochs for subnet')
parser.add_argument('--subnet_learning_rate', type=float, default=0.025, help='subnet init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--w_lr', type=float, default=3e-3, help='w learning rate')
parser.add_argument('--w_weight_decay', type=float, default=1e-3, help='weight decay for w')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
args = parser.parse_args()


def main():
    np.random.seed(args.seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)
    # torch.cuda.manual_seed(args.seed)

    model = DC_ConvNet(32, 10, 32, 5, 'relu', 'identity', 'maxpooling')

    for k, v in model.state_dict().items():
        print(k, v.shape)
    old_weights = torch.load('subnet_weights1.pt')
    for k, v in old_weights.items():
        print(k, v.shape)
    curr_weights = model.state_dict()
    new_weights1 = {k: v for k, v in old_weights.items() if k in curr_weights.keys()}
    print(new_weights1.keys())

    old_body_weights = torch.stack([old_weights['features.1.weight'].reshape(-1),
                                    old_weights['features.4.weight'].reshape(-1),
                                    old_weights['features.7.weight'].reshape(-1)])  # 要展平吗
    w = torch.FloatTensor([[ 1., 0,  0],
                           [ 0, 1,  0.],
                           [0., 0., 1],
                           [0,  1.,  0],
                           [ 0.,  1., 0]])
    w = torch.FloatTensor([[ 0.1234, -0.1700,  0.2963],
                           [ 1.0777, -0.0489,  0.2692],
                           [0.7509, 0.3500, 0.0230],
                           [-0.2398,  0.6306,  0.2153],
                           [ 0.5272,  0.3910, -0.0689]])
    new_body_weights = torch.matmul(w, old_body_weights)
    new_weights2 = {'features.1.weight': new_body_weights[0].reshape(32, 32, 3, 3),
                    'features.4.weight': new_body_weights[1].reshape(32, 32, 3, 3),
                    'features.7.weight': new_body_weights[2].reshape(32, 32, 3, 3),
                    'features.10.weight': new_body_weights[3].reshape(32, 32, 3, 3),
                    'features.13.weight': new_body_weights[4].reshape(32, 32, 3, 3)}
    print(new_weights2.keys())
    curr_weights.update(new_weights1)
    curr_weights.update(new_weights2)
    model.load_state_dict(curr_weights)

    # model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    # criterion = criterion.cuda()

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.subnet_learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    # optimizer = torch.optim.Adam(model.parameters(),
    #                              lr=args.w_lr, betas=(0.5, 0.999),
    #                              weight_decay=args.weight_decay)

    train_transform, valid_transform = data_transforms_cifar10(args)
    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.subnet_batch_size, shuffle=True, pin_memory=True, num_workers=0)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.subnet_batch_size, shuffle=False, pin_memory=True, num_workers=0)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.subnet_epochs))

    # # for w learning
    # best_val_acc = 0.
    # for epoch in range(args.subnet_epochs):
    #     valid_acc, valid_obj = subnet_infer(valid_queue, model, criterion)
    #     print(f'valid acc: {valid_acc}')
    #
    #     if valid_acc > best_val_acc:
    #         best_val_acc = valid_acc
    #         # torch.save(model.state_dict(), 'subnet_weights2.pt')
    #     for k, v in model.named_parameters():
    #         print(k, v)
    #
    #     train_acc, train_obj = subnet_train(train_queue, model, criterion, optimizer)

    best_val_acc = 0.
    for epoch in range(args.subnet_epochs):
        scheduler.step()
        valid_acc, valid_obj = subnet_infer(valid_queue, model, criterion)
        print(f'valid acc: {valid_acc}')

        if valid_acc > best_val_acc:
            best_val_acc = valid_acc
            # torch.save(model.state_dict(), 'subnet_weights2.pt')

        train_acc, train_obj = subnet_train(train_queue, model, criterion, optimizer)

    return best_val_acc


def subnet_train(train_queue, model, criterion, optimizer):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()

    for step, (input, target) in enumerate(train_queue):
        # input = input.cuda()
        # target = target.cuda()

        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)
        # print(f'train loss: {loss}')
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

    return top1.avg, objs.avg


def subnet_infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        # input = input.cuda()
        # target = target.cuda()

        logits = model(input)
        loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

    return top1.avg, objs.avg


# dataset
def data_transforms_cifar10(args):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_transform, valid_transform


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


if __name__ == '__main__':
    main()
