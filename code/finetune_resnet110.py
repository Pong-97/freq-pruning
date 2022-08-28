# -*- coding: utf-8 -*-  
from __future__ import print_function
import os
import argparse
import shutil
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.datasets import MNIST, CIFAR10, ImageNet, ImageFolder
from torch.autograd import Variable
import matplotlib.pyplot as plt # plt 用于显示图片
from models import *
from models.vgg_16_bn import vgg_16_bn
from models.resnet_cifar import resnet_56,resnet_110
# import torchvision.models as models


import pdb



# Training settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar10)')
parser.add_argument('--sparsity-regularization', '-sr', dest='sr', action='store_true',
                    help='train with channel sparsity regularization')
parser.add_argument('--s', type=float, default=0.0001,
                    help='scale sparse rate (default: 0.0001)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test_batch_size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=600, metavar='N',
                    help='number of epochs to train (default: 160)')  #160 for train
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', type=float, default=0.08, metavar='LR',
                    help='learning rate (default: 0.1)')  #default: 0.1
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default="", type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=150, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--arch', default='vgg', type=str, 
                    help='architecture to use')
parser.add_argument('--depth', default=16, type=int,
                    help='depth of the neural network')
parser.add_argument('--seg_num', default=3, type=int)


parser.add_argument('--refine', default="/home2/pengyifan/pyf/freq-lite/logs/resnet110/405060/model_best50.pth.tar", type=str, metavar='PATH',
                    help='path to the pruned model to be fine tuned')
#vgg19 cifar10: '/home2/pengyifan/pyf/freq/logs/vgg19cifar10/pruned.pth.tar'
#vgg19 cifar100: "/home2/pengyifan/pyf/freq/logs/vgg19cifar100/pruned.pth.tar"
# /home2/pengyifan/pyf/freq/logs/temp/pruned6.pth.tar
parser.add_argument('--save', default='/home2/pengyifan/pyf/freq-lite/logs/resnet110/405060/finetuned/p47' , type=str, metavar='PATH',
                    help='path to save prune model (default: current directory)')
#vgg19 cifar10: '/home2/pengyifan/pyf/freq/logs/vgg19cifar10'    
#vgg19 cifar100: "/home2/pengyifan/pyf/freq/logs/vgg19cifar100"      

# CUDA_VISIBLE_DEVICES=1 nohup python -u finetune_resnet110.py > /home2/pengyifan/pyf/freq-lite/logs/resnet110/405060/finetuned/p47/1.log 2>&1 &

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


SEG_NUM = args.seg_num  #3, 4, 5, 8, 16
SEG_IDEX = 1
compress_rate=[0.1]+[0.60]*35+[0.0]*2+[0.6]*6+[0.4]*3+[0.1]+[0.4]+[0.1]+[0.4]+[0.1]+[0.4]+[0.1]+[0.4]

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if not os.path.exists(args.save):
    os.makedirs(args.save)


if args.refine:
    print("=> loaded checkpoint '{}'".format(args.refine))
    checkpoint = torch.load(args.refine)
    cfg = checkpoint['cfg'] 
    # pdb.set_trace()
    # model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth, cfg=checkpoint['cfg'])
    model = resnet_110(cfg=cfg)
    # print(checkpoint['state_dict'])
    model.cuda()
    model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['state_dict'])
    print(model)
    num_parameters = sum([param.nelement() for param in model.parameters()])
    print("num_parameters: ", num_parameters)
    print(compress_rate)
    print("=> save path '{}'".format(args.save))

else:
    model = resnet_110(compress_rate=compress_rate)
    # input = torch.randn(1, 3, 32, 32)
    # flops, params = profile(model, inputs=(input, ) )  #  profile（模型，输入数据）
    # print("FLOPS: ", flops, "\nPARAMS:", params)
    # pdb.set_trace()
    model.cuda()
    model = torch.nn.DataParallel(model)
    print(model)
    num_parameters = sum([param.nelement() for param in model.parameters()])
    print("num_parameters: ", num_parameters)
    print(compress_rate)

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.epochs*0.5, args.epochs*0.75], gamma=0.1)

def transform_convert(img_tensor, transform):
    """
    param img_tensor: tensor
    param transforms: torchvision.transforms
    """
    if 'Normalize' in str(transform):
        normal_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transform.transforms))
        mean = torch.tensor(normal_transform[0].mean, dtype=img_tensor.dtype, device=img_tensor.device)
        std = torch.tensor(normal_transform[0].std, dtype=img_tensor.dtype, device=img_tensor.device)
        img_tensor.mul_(std[:,None,None]).add_(mean[:,None,None])
        
    img_tensor = img_tensor.transpose(0,2).transpose(0,1)  # C x H x W  ---> H x W x C
    
    if 'ToTensor' in str(transform) or img_tensor.max() < 1:
        img_tensor = img_tensor.detach().numpy()*255
    
    if isinstance(img_tensor, torch.Tensor):
    	img_tensor = img_tensor.numpy()
    
    if img_tensor.shape[2] == 3:
        img = Image.fromarray(img_tensor.astype('uint8')).convert('RGB')
    elif img_tensor.shape[2] == 1:
        img = Image.fromarray(img_tensor.astype('uint8')).squeeze()
    else:
        raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(img_tensor.shape[2]))
        
    return img

trans_forimagenet = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225]),
                                        ])

kwargs = {'num_workers': 16, 'pin_memory': True} if args.cuda else {}
if args.dataset == 'cifar10':
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10("/repository/linhang/data/cifar-10/", train=True, download=True,
                    transform=transforms.Compose([
                        transforms.Pad(4),
                        transforms.RandomCrop(32),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                    ])),
        batch_size=128, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10("/repository/linhang/data/cifar-10/", train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                    ])),
        batch_size=128, shuffle=True, **kwargs)

if args.dataset == 'cifar100':
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('/repository/linhang/data/cifar-100/', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.Pad(4),
                        transforms.RandomCrop(32),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                    ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('/repository/linhang/data/cifar-100/', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                    ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

if args.dataset == 'imagenet':
    print("数据集为ILSVRC2012")
    data_tmp = imagenet.Data()
    train_loader = data_tmp.loader_train
    test_loader = data_tmp.loader_test


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        # std = getStd()
        # loss = F.cross_entropy(output, target) + SCALE*(1/std)
        loss = F.cross_entropy(output, target)
        pred = output.data.max(1, keepdim=True)[1]
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            # print('\nTrain Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}\t'.format(
            #     epoch, batch_idx * len(data), len(train_loader.dataset),
            #     100. * batch_idx / len(train_loader), loss.item()))
            print('\nTrain Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}\t'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    
    for data, target in test_loader:
        # pdb.set_trace()
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
        output = model(data)
        test_loss += F.cross_entropy(output, target).item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%), learning rate: {}\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset), optimizer.param_groups[0]['lr']))
    return float(correct) / float(len(test_loader.dataset))

def save_checkpoint(state, is_best, filepath):
    torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
    if is_best:
        print("is best and save!")
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, 'model_best.pth.tar'))

test()
best_prec1 = 0.
for epoch in range(args.start_epoch, args.epochs):
    # if epoch in [args.epochs*0.5, args.epochs*0.75]:
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] *= 0.1

    train(epoch)
    scheduler.step()
    prec1 = test()
    is_best = prec1 > best_prec1
    best_prec1 = max(prec1, best_prec1)
    # save_checkpoint({
    #     'epoch': epoch + 1,
    #     'state_dict': model.state_dict(),
    #     'best_prec1': best_prec1,
    #     'optimizer': optimizer.state_dict(),
    # }, is_best, filepath=args.save)

    save_checkpoint({

        'state_dict': model.state_dict(),
        'compress_rate': compress_rate,
        'cfg': cfg,
        'num_parameters': num_parameters

    }, is_best, filepath=args.save)

print("Best accuracy: "+str(best_prec1))
# print(optimizer.state_dict())
