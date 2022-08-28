import os
import argparse
import numpy as np
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image

from data_process import CIFAR10_DATA_PROCESS, CIFAR100_DATA_PROCESS, MNIST_DATA_PROCESS
from models import *
from models.vgg_16_bn import vgg_16_bn
from models.resnet_cifar import resnet_56, resnet_110
from models.resnet_imagenet import resnet_50

import pdb

# Prune settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
parser.add_argument('--seed', type=int, default=2, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar10)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--depth', type=int, default=16,
                    help='depth of the vgg')
parser.add_argument('--percent', type=float, default=0.85,
                    help='scale sparse rate (default: 0.5)')
parser.add_argument('--model', default="/home2/pengyifan/pyf/freq/pretrained_model/vgg_16_bn.pt.pt", type=str, metavar='PATH',
                    help='path to the model (default: none)')
#vgg16 cifar10 :"/home2/pengyifan/pyf/network-slimming-master/logs/vgg16cifar10/vgg_16_bn.pt.pt"
#vgg19 cifar10 :"/home2/pengyifan/pyf/network-slimming-master/logs/vgg19cifar10/model_best.pth.tar"
#vgg19 cifar100 :"/home2/pengyifan/pyf/network-slimming-master/logs/VGG19CIFAR100/model_best.pth.tar"
#resnet164 cifar100 :"/home2/pengyifan/pyf/network-slimming-master/logs/resnet164CIFAR100/model_best.pth.tar"
#resnet164 cifar10 :"/home2/pengyifan/pyf/network-slimming-master/logs/resnet164CIFAR10/model_best.pth.tar"
#vgg19 mnist :"/home2/pengyifan/pyf/network-slimming-master/logs/vgg19MNIST/model_best.pth.tar"
#cnn mnist :"/home2/pengyifan/pyf/network-slimming-master/logs/cnnMNIST/model_best.pth.tar"
#resnet50 imagenet :"/home2/pengyifan/pyf/network-slimming-master/logs/Resnet50ImageNet/resnet50-19c8e357.pth"
#resnet34 imagenet :"/home2/pengyifan/pyf/network-slimming-master/logs/Resnet34Imagenet/resnet34-333f7ec4.pth"
parser.add_argument('--save', default="/home2/pengyifan/pyf/freq-lite/logs/temp", type=str, metavar='PATH',
                    help='path to save pruned model (default: none)')
#vgg16 cifar10 :'/home2/pengyifan/pyf/freq/logs/vgg16cifar10/'
#vgg19 cifar10: "/home2/pengyifan/pyf/freq/logs/vgg19cifar10/perfect_models/"
#vgg19 cifar100: '/home2/pengyifan/pyf/freq/logs/vgg19cifar100/'


ALPHA = 0.5
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

def processed_forward(model_path, keep_ratio, dataset='cifar10', seg_num=8, layer_idx=0, model_arch='vgg16'):
    
    model, model_head, model_tail = prepare_model(model_path, layer_idx, model_arch=model_arch)

    # for tail
    # print("Aphla: ", ALPHA)
    kwargs = {'num_workers':16, 'pin_memory': True} if args.cuda else {}
    if dataset == 'cifar10':
        loader_tail = DataLoader(CIFAR10_DATA_PROCESS("/repository/linhang/data/cifar-10/", 
                                            train=False, 
                                            transform=transforms.Compose([
                                                transforms.Pad(4),
                                                transforms.RandomCrop(32),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                                ]),
                                            seg_num=seg_num, seg_idex=8,
                                            forward=True, tail=True),
                                batch_size=args.batch_size, 
                                shuffle=True, **kwargs)
    elif dataset == 'cifar100':
        loader_pre = DataLoader(CIFAR100_DATA_PROCESS('/repository/linhang/data/cifar-100/', 
                                            train=False, 
                                            transform=transforms.Compose([
                                                transforms.Pad(4),
                                                transforms.RandomCrop(32),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                                ]),
                                            seg_num=seg_num, seg_idex=1,
                                            forward=True),
                                batch_size=args.batch_size, 
                                shuffle=True, **kwargs) 

    inputs, targets = next(iter(loader_tail))
    inputs = inputs.cuda()
    targets = targets.cuda()

    # for layer in model_tail.modules():
    #     if isinstance(layer, nn.BatchNorm2d):
    #         layer.weight.requires_grad = False
    model_tail.train()
    model_tail.zero_grad()
    outputs = model_tail.forward(inputs)
    loss = F.nll_loss(outputs, targets)
    # loss = F.cross_entropy(outputs, targets)
    loss.backward()

    idx_tail = 0
    conv_grads_abs_tail = []
    for layer in model_tail.modules():
        if isinstance(layer, nn.Conv2d):
            if idx_tail == layer_idx:
                sum = torch.squeeze(torch.sum(layer.weight.grad.abs().clone(),dim=[1,2,3]))
                conv_grads_abs_tail.append(sum)   
            idx_tail = idx_tail + 1     

    all_grads_tail = torch.cat([torch.flatten(x) for x in conv_grads_abs_tail]) #变化越大的保留
    # pdb.set_trace()
    # for head
    kwargs = {'num_workers':16, 'pin_memory': True} if args.cuda else {}
    if dataset == 'cifar10':
        loader_head = DataLoader(CIFAR10_DATA_PROCESS("/repository/linhang/data/cifar-10/", 
                                            train=False, 
                                            transform=transforms.Compose([
                                                transforms.Pad(4),
                                                transforms.RandomCrop(32),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                                ]),
                                            seg_num=seg_num, seg_idex=1,
                                            forward=True, head=True),
                                batch_size=args.batch_size, 
                                shuffle=True, **kwargs)
    elif dataset == 'cifar100':
        loader_pre = DataLoader(CIFAR100_DATA_PROCESS('/repository/linhang/data/cifar-100/', 
                                            train=False, 
                                            transform=transforms.Compose([
                                                transforms.Pad(4),
                                                transforms.RandomCrop(32),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                                ]),
                                            seg_num=seg_num, seg_idex=1,
                                            forward=True),
                                batch_size=args.batch_size, 
                                shuffle=True, **kwargs) 

    inputs, targets = next(iter(loader_head))
    inputs = inputs.cuda()
    targets = targets.cuda()

    # for layer in model_head.modules():
    #     if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
    #         layer.weight.requires_grad = False
    model_head.train()
    model_head.zero_grad()
    outputs = model_head.forward(inputs)
    loss = F.nll_loss(outputs, targets)
    # loss = F.cross_entropy(outputs, targets)
    loss.backward()

    idx_head = 0
    conv_grads_abs_head = []
    for layer in model_head.modules():
        if isinstance(layer, nn.Conv2d):
            if idx_head == layer_idx:
                sum = torch.squeeze(torch.sum(layer.weight.grad.abs().clone(),dim=[1,2,3]))
                conv_grads_abs_head.append(sum)    
            idx_head = idx_head + 1                      
    all_grads_head = torch.cat([torch.flatten(x) for x in conv_grads_abs_head]) #变化越大的保留
    # bn_grads_abs = bn_grads_abs_head + bn_grads_abs_tail
    all_grads = (1-ALPHA)*all_grads_tail - ALPHA*(all_grads_head)
    # 对all_grads_tail响应越大，说明越适配低频；对all_grads_head响应越大，说明越适配高频；
    # 如果要保留梯度大的，all_grads_tail前面的系数需要为正，all_grads_head前面的系数需要为负。

    num_params_to_keep = int(len(all_grads) * keep_ratio)
    threshold, _ = torch.topk(all_grads, num_params_to_keep, sorted=True) #for 去尾
    reject_score = threshold[-1] 

    pruned = 0
    cfg = []
    cfg_mask = []

    idx = 0
    for k,[m0, m1, m2] in enumerate(zip(model.modules(), model_tail.modules(), model_head.modules())):
        if isinstance(m0, nn.Conv2d):
            if idx == layer_idx:
                grad_copy = (1-ALPHA)*torch.squeeze(torch.sum(m1.weight.grad.abs().clone(),dim=[1,2,3])) - ALPHA*torch.squeeze(torch.sum(m2.weight.grad.abs().clone(),dim=[1,2,3]))
                mask = grad_copy.ge(reject_score).float().cuda()
                cfg.append(int(torch.sum(mask)))
                cfg_mask.append(mask.clone())
                print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.format(k, mask.shape[0], int(torch.sum(mask))))
            else:
                mask = torch.ones(m0.weight.shape[0]).float().cuda()
                cfg.append(int(torch.sum(mask)))
                cfg_mask.append(mask.clone())                
            idx = idx + 1 
        elif isinstance(m0, nn.MaxPool2d):
            cfg.append('M') 

    # layeridx = 0
    # for k, m in enumerate(model.modules()):
    #     if isinstance(m, nn.BatchNorm2d):
    #         m.weight.data.mul_(cfg_mask[layeridx])
    #         m.bias.data.mul_(cfg_mask[layeridx])
    #         layeridx += 1
    
    # print("actually pruning rate: ", float(pruned)/len(all_grads))
    if model_arch == 'vgg16':
        newmodel = vgg_16_bn(cfg=cfg, compress_rate=[0,0,0,0,0,0,0,0,0,0,0,0,0])
    elif model_arch == 'resnet56':
        newmodel = resnet_56(cfg=cfg)
    if args.cuda:
        newmodel.cuda()
        newmodel = nn.DataParallel(newmodel) 
    newmodel = transfer_value(cfg_mask, model, newmodel)


    return cfg, newmodel


def log(model, cfg, seed):
    num_parameters = sum([param.nelement() for param in model.parameters()])
    print("number of parameters: ", num_parameters)
    savepath = os.path.join(args.save, "prune{}.txt".format(seed))
    with open(savepath, "w") as fp:
        fp.write("Configuration: \n"+str(cfg)+"\n")
        fp.write("Number of parameters: \n"+str(num_parameters)+"\n")
        # fp.write("Test accuracy: \n"+str(acc))


def transfer_value(cfg_mask, model, newmodel):
    layer_id_in_cfg = 0
    start_mask = torch.ones(3)
    end_mask = cfg_mask[layer_id_in_cfg]
    linear_flag = 0
    for [m0, m1] in zip(model.modules(), newmodel.modules()):
        if isinstance(m0, nn.BatchNorm2d):
            # pdb.set_trace()
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            if idx1.size == 1:
                idx1 = np.resize(idx1,(1,))
            m1.weight.data = m0.weight.data[idx1.tolist()].clone()
            m1.bias.data = m0.bias.data[idx1.tolist()].clone()
            m1.running_mean = m0.running_mean[idx1.tolist()].clone()
            m1.running_var = m0.running_var[idx1.tolist()].clone()
            layer_id_in_cfg += 1
            start_mask = end_mask.clone()
            if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                end_mask = cfg_mask[layer_id_in_cfg]
        elif isinstance(m0, nn.Conv2d):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            # print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
            w1 = w1[idx1.tolist(), :, :, :].clone()
            m1.weight.data = w1.clone()
        # elif isinstance(m0, nn.Linear):
        #     if(linear_flag == 1):
        #         m1.weight.data = m0.weight.data[:, :].clone()
        #         m1.bias.data = m0.bias.data.clone()
        #         continue
        #     idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
        #     # pdb.set_trace()
        #     if idx0.size == 1:
        #         idx0 = np.resize(idx0, (1,))
        #     m1.weight.data = m0.weight.data[:, idx0].clone()
        #     m1.bias.data = m0.bias.data.clone()
        #     linear_flag += 1

    for layer in model.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            layer.weight.requires_grad = True
    for layer in newmodel.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            layer.weight.requires_grad = True

    return newmodel


def test(model):
    kwargs = {'num_workers': 16, 'pin_memory': True} if args.cuda else {}
    if args.dataset == 'cifar10':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10("/repository/linhang/data/cifar-10/", train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
    elif args.dataset == 'cifar100':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100("/repository/linhang/data/cifar-100/", train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
    else:
        raise ValueError("No valid dataset is given.")
    model.eval()
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    print('\nTest set: Accuracy: {}/{} ({:.1f}%)\n'.format(
        correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    return correct / float(len(test_loader.dataset))


def prepare_model(model_path='', layer_idx=0, model_arch='vgg16'):
    if model_arch=='vgg16':
        if layer_idx == 0:
            model = vgg_16_bn([0,0,0,0,0,0,0,0,0,0,0,0,0])
            model_head = vgg_16_bn([0,0,0,0,0,0,0,0,0,0,0,0,0])
            model_tail = vgg_16_bn([0,0,0,0,0,0,0,0,0,0,0,0,0])
            ori_model_path = '/home2/pengyifan/pyf/hypergraph_cluster/log/pretrained_model/vgg_16_bn.pt.pt'
            if os.path.isfile(ori_model_path):
                print("=> loading checkpoint '{}'".format(ori_model_path))
                checkpoint = torch.load(ori_model_path, map_location='cpu')

                best_prec1 = checkpoint['best_prec1']
                model.load_state_dict(checkpoint['state_dict'], strict=True)
                model_head.load_state_dict(checkpoint['state_dict'], strict=True)
                model_tail.load_state_dict(checkpoint['state_dict'], strict=True)
                print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}".format(ori_model_path, checkpoint['epoch'], best_prec1))
            if args.cuda:
                model.cuda()
                model = nn.DataParallel(model) 
                model_head.cuda()
                model_head = nn.DataParallel(model_head) 
                model_tail.cuda()
                model_tail = nn.DataParallel(model_tail) 
        else:
            checkpoint = torch.load(model_path, map_location='cpu')
            model_cfg = checkpoint['cfg']
            best_prec1 = checkpoint['best_prec1']
            model = vgg_16_bn(cfg=model_cfg,compress_rate=[0,0,0,0,0,0,0,0,0,0,0,0,0])
            model_head = vgg_16_bn(cfg=model_cfg,compress_rate=[0,0,0,0,0,0,0,0,0,0,0,0,0])
            model_tail = vgg_16_bn(cfg=model_cfg,compress_rate=[0,0,0,0,0,0,0,0,0,0,0,0,0])
            model.cuda()
            model = nn.DataParallel(model) 
            model_head.cuda()
            model_head = nn.DataParallel(model_head) 
            model_tail.cuda()
            model_tail = nn.DataParallel(model_tail) 
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            model_head.load_state_dict(checkpoint['state_dict'], strict=True)
            model_tail.load_state_dict(checkpoint['state_dict'], strict=True)
            print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}".format(model_path, checkpoint['epoch'], best_prec1))

    elif model_arch=='resnet56':
        if layer_idx == 0:
            model = resnet_56()
            model_head = resnet_56()
            model_tail = resnet_56()
            ori_model_path = "/home2/pengyifan/pyf/freq/pretrained_model/cifar_resnet56/model_best.pth.tar"
            if os.path.isfile(ori_model_path):
                print("=> loading checkpoint '{}'".format(ori_model_path))
                checkpoint = torch.load(ori_model_path, map_location='cpu')
                # for k, v in checkpoint['state_dict'].items():
                #     if 'linear' in k:
                #         k = k.replace('linear', 'fc')
                # pdb.set_trace()
                fc_w = {'module.fc.weight': checkpoint['state_dict']['module.linear.weight']}
                fc_b = {'module.fc.bias': checkpoint['state_dict']['module.linear.bias']}
                checkpoint['state_dict'].update(fc_w)
                checkpoint['state_dict'].update(fc_b)
                checkpoint['state_dict'].pop('module.linear.weight')
                checkpoint['state_dict'].pop('module.linear.bias')

                model.cuda()
                model = nn.DataParallel(model) 
                model_head.cuda()
                model_head = nn.DataParallel(model_head) 
                model_tail.cuda()
                model_tail = nn.DataParallel(model_tail) 
                # best_prec1 = checkpoint['best_prec1']
                # model.load_state_dict({k.replace('module.',''):v for k,v in checkpoint['state_dict'].items()})
                model.load_state_dict(checkpoint['state_dict'], strict=True) #TODO:'strict= ?'
                model_head.load_state_dict(checkpoint['state_dict'], strict=True)
                model_tail.load_state_dict(checkpoint['state_dict'], strict=True)
                print("=> loaded checkpoint '{}'  ".format(ori_model_path))
        
        else:
            checkpoint = torch.load(model_path, map_location='cpu')
            model_cfg = checkpoint['cfg']
            best_prec1 = checkpoint['best_prec1']
            model = resnet_56(cfg=model_cfg)
            model_head = resnet_56(cfg=model_cfg)
            model_tail = resnet_56(cfg=model_cfg)
            model.cuda()
            model = nn.DataParallel(model) 
            model_head.cuda()
            model_head = nn.DataParallel(model_head) 
            model_tail.cuda()
            model_tail = nn.DataParallel(model_tail) 
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            model_head.load_state_dict(checkpoint['state_dict'], strict=True)
            model_tail.load_state_dict(checkpoint['state_dict'], strict=True)
            print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}".format(model_path, checkpoint['epoch'], best_prec1))

    elif model_arch=='resnet110':
        if layer_idx == 0:
            model = resnet_110()
            model_head = resnet_110()
            model_tail = resnet_110()
            ori_model_path = "/home2/pengyifan/pyf/hypergraph_cluster/log/pretrained_model/resnet_110.pt"
            if os.path.isfile(ori_model_path):
                print("=> loading checkpoint '{}'".format(ori_model_path))
                checkpoint = torch.load(ori_model_path, map_location='cpu')
                model.cuda()
                model = nn.DataParallel(model) 
                model_head.cuda()
                model_head = nn.DataParallel(model_head) 
                model_tail.cuda()
                model_tail = nn.DataParallel(model_tail) 
                model.load_state_dict(checkpoint, strict=True) #TODO:'strict= ?'
                model_head.load_state_dict(checkpoint, strict=True)
                model_tail.load_state_dict(checkpoint, strict=True)
                print("=> loaded checkpoint '{}'  ".format(ori_model_path))
        
        else:
            checkpoint = torch.load(model_path, map_location='cpu')
            model_cfg = checkpoint['cfg']
            best_prec1 = checkpoint['best_prec1']
            model = resnet_110(cfg=model_cfg)
            model_head = resnet_110(cfg=model_cfg)
            model_tail = resnet_110(cfg=model_cfg)
            model.cuda()
            model = nn.DataParallel(model) 
            model_head.cuda()
            model_head = nn.DataParallel(model_head) 
            model_tail.cuda()
            model_tail = nn.DataParallel(model_tail) 
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            model_head.load_state_dict(checkpoint['state_dict'], strict=True)
            model_tail.load_state_dict(checkpoint['state_dict'], strict=True)
            print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}".format(model_path, checkpoint['epoch'], best_prec1))

    return model, model_head, model_tail


# for seedhahaha in range(100):

#     args = parser.parse_args()
#     args.cuda = not args.no_cuda and torch.cuda.is_available()

#     if args.cuda:
#         torch.cuda.manual_seed(seedhahaha)
#     print("\n#################################### seed={} ####################################".format(seedhahaha))

#     if args.percent:
#         keep_ratio = 1-args.percent
#         print("pruning rate: ", args.percent)

#     if not os.path.exists(args.save):
#         os.makedirs(args.save)

#     model, model_head, model_tail = prepare_model()

#     reject_score, cfg, cfg_mask = processed_forward(model_path, keep_ratio, args.dataset, layer_idx= )

#     print('Pre-processing Successful!')

#     # Make real prune
#     print(cfg)

#     newmodel = vgg_16_bn(cfg=cfg, compress_rate=[0,0,0,0,0,0,0,0,0,0,0,0,0])
#     if args.cuda:
#         newmodel.cuda()
#         newmodel = nn.DataParallel(newmodel) 

#     log(newmodel, cfg, seedhahaha)

#     newmodel = transfer_value(cfg_mask, model, newmodel)
#     # pdb.set_trace()
#     # test(model)   
#     torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()}, os.path.join(args.save, 'pruned{}.pth.tar'.format(seedhahaha)))
#     # print("=> saved the newmodel at '{}'".format(args.save))

#     print("\nSmall test.")
#     inputs = torch.rand((2, 3, 32, 32)).cuda()
#     model = newmodel.cuda().train()
#     output = model(inputs)
#     print(output.shape)