import os
import argparse
import numpy as np
import shutil
import torch
import torch.nn as nn
from torch.nn import modules
import torch.nn.functional as F
from torch.nn.modules import module
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from models.resnet_cifar import ResBasicBlock
from data_process import CIFAR10_DATA_PROCESS, CIFAR100_DATA_PROCESS, MNIST_DATA_PROCESS
from models import *
from models.resnet_cifar import resnet_110

import pdb

# Prune settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
parser.add_argument('--seed', type=int, default=2, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--dataset', type=str, default='imagenet',
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
parser.add_argument('--save', default="/home2/pengyifan/pyf/freq-lite/logs/temp", type=str, metavar='PATH',
                    help='path to save pruned model (default: none)')


ALPHA = 0.5
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


def test_model_enable(model):
    inputs = torch.rand((1, 3, 32, 32)).cuda()
    model = model.cuda().train()
    output = model(inputs)
    print("test, output's shape: ", output.shape)

def processed_forward(model_path, keep_ratio, dataset='cifar10', seg_num=8, block_idx=0, model_arch='resnet110', ):
    # pdb.set_trace()
    if block_idx != 0:
        model, model_head, model_tail = prepare_model(model_path, block_idx, model_arch=model_arch)

        # for tail
        # print("Aphla: ", ALPHA)
        print("process for tail.")
        kwargs = {'num_workers':0, 'pin_memory': True} if args.cuda else {}
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

        inputs, targets = next(iter(loader_tail))
        inputs = inputs.cuda()
        targets = targets.cuda()

        # for layer in model_tail.modules():
        #     if isinstance(layer, nn.BatchNorm2d):
        #         layer.weight.requires_grad = False
        model_tail.train()
        model_tail.zero_grad()
        outputs = model_tail.forward(inputs)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()

        blockidx = 0
        conv1_grads_abs_tail = []
        conv2_grads_abs_tail = []
        for layer in model_tail.modules():
            if isinstance(layer, ResBasicBlock):
                if blockidx == block_idx:
                    for k1, layer1 in enumerate(layer.modules()):
                        if isinstance(layer1, nn.Conv2d):
                            # pdb.set_trace()
                            if k1==1:
                                sum = torch.squeeze(torch.sum(layer1.weight.grad.abs().clone(),dim=[1,2,3]))
                                conv1_grads_abs_tail.append(sum)
                            if k1==4:
                                sum = torch.squeeze(torch.sum(layer1.weight.grad.abs().clone(),dim=[1,2,3]))
                                conv2_grads_abs_tail.append(sum)
                blockidx = blockidx + 1
        
        # for head
        print("process for head.")
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

        inputs, targets = next(iter(loader_head))
        inputs = inputs.cuda()
        targets = targets.cuda()

        # for layer in model_head.modules():
        #     if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        #         layer.weight.requires_grad = False
        model_head.train()
        model_head.zero_grad()
        outputs = model_head.forward(inputs)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()

        blockidx=0
        conv1_grads_abs_head = []
        conv2_grads_abs_head = []
        for layer in model_head.modules():
            if isinstance(layer, ResBasicBlock):
                if blockidx == block_idx:
                    for k1, layer1 in enumerate(layer.modules()):
                        if isinstance(layer1, nn.Conv2d):
                            # pdb.set_trace()
                            if k1==1:
                                # print(k1, " ", layer1)
                                sum = torch.squeeze(torch.sum(layer1.weight.grad.abs().clone(),dim=[1,2,3]))
                                conv1_grads_abs_head.append(sum)
                            elif k1==4:
                                sum = torch.squeeze(torch.sum(layer1.weight.grad.abs().clone(),dim=[1,2,3]))
                                conv2_grads_abs_head.append(sum)                        

                blockidx = blockidx + 1
                
        conv1_grads = (1-ALPHA)*conv1_grads_abs_tail[0] - ALPHA*(conv1_grads_abs_head[0])
        conv2_grads = (1-ALPHA)*conv2_grads_abs_tail[0] - ALPHA*(conv2_grads_abs_head[0])

        conv1_num_params_to_keep = int(len(conv1_grads) * keep_ratio[0])
        conv2_num_params_to_keep = int(len(conv2_grads) * keep_ratio[1])

        threshold1, _ = torch.topk(conv1_grads, conv1_num_params_to_keep, sorted=True)
        threshold2, _ = torch.topk(conv2_grads, conv2_num_params_to_keep, sorted=True)
        reject_score1 = threshold1[-1] 
        reject_score2 = threshold2[-1] 

        cfg1 = 0
        cfg1_mask = []
        mask = conv1_grads.ge(reject_score1).float().cuda()
        cfg1 = int(torch.sum(mask))
        cfg1_mask.append(mask.clone())

        cfg2 = 0
        cfg2_mask = []
        mask = conv2_grads.ge(reject_score2).float().cuda()
        cfg2 = int(torch.sum(mask))
        cfg2_mask.append(mask.clone())

        cfg = []
        cfg_mask=[]
        for k, m in enumerate(model.modules()):
            if isinstance(m, nn.BatchNorm2d):  
                # pdb.set_trace()
                cfg.append(m.weight.shape[0])
                cfg_mask.append(torch.ones(m.weight.shape[0]))

        cfg[2*block_idx+1] = cfg1
        cfg[2*block_idx+2] = cfg2

        cfg_mask[2*block_idx+1] = cfg1_mask[0]
        cfg_mask[2*block_idx+2] = cfg2_mask[0]

        newmodel = resnet_110(cfg=cfg)
        # pdb.set_trace()
        if args.cuda:
            newmodel.cuda()
            newmodel = nn.DataParallel(newmodel) 
        newmodel = transfer_value(cfg_mask, model, newmodel)
        # pdb.set_trace()
        test_model_enable(newmodel)
    else:
        model, model_head, model_tail = prepare_model(model_path, block_idx, model_arch=model_arch)

        # for tail
        # print("Aphla: ", ALPHA)
        print("process for tail.")
        kwargs = {'num_workers':0, 'pin_memory': True} if args.cuda else {}
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

        inputs, targets = next(iter(loader_tail))
        inputs = inputs.cuda()
        targets = targets.cuda()

        # for layer in model_tail.modules():
        #     if isinstance(layer, nn.BatchNorm2d):
        #         layer.weight.requires_grad = False
        model_tail.train()
        model_tail.zero_grad()
        outputs = model_tail.forward(inputs)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()

        conv0_grads_abs_tail = []
        for layer in model_tail.modules():
            if isinstance(layer, nn.Conv2d):
                sum = torch.squeeze(torch.sum(layer.weight.grad.abs().clone(),dim=[1,2,3]))
                conv0_grads_abs_tail.append(sum)
                break

        blockidx = 0
        conv1_grads_abs_tail = []
        conv2_grads_abs_tail = []
        for layer in model_tail.modules():
            if isinstance(layer, ResBasicBlock):
                if blockidx == block_idx:
                    for k1, layer1 in enumerate(layer.modules()):
                        if isinstance(layer1, nn.Conv2d):
                            # pdb.set_trace()
                            if k1==1:
                                sum = torch.squeeze(torch.sum(layer1.weight.grad.abs().clone(),dim=[1,2,3]))
                                conv1_grads_abs_tail.append(sum)
                            if k1==4:
                                sum = torch.squeeze(torch.sum(layer1.weight.grad.abs().clone(),dim=[1,2,3]))
                                conv2_grads_abs_tail.append(sum)
                blockidx = blockidx + 1
        
        # for head
        print("process for head.")
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

        inputs, targets = next(iter(loader_head))
        inputs = inputs.cuda()
        targets = targets.cuda()

        # for layer in model_head.modules():
        #     if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        #         layer.weight.requires_grad = False
        model_head.train()
        model_head.zero_grad()
        outputs = model_head.forward(inputs)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()

        conv0_grads_abs_head = []
        for layer in model_head.modules():
            if isinstance(layer, nn.Conv2d):
                sum = torch.squeeze(torch.sum(layer.weight.grad.abs().clone(),dim=[1,2,3]))
                conv0_grads_abs_head.append(sum)
                break

        blockidx=0
        conv1_grads_abs_head = []
        conv2_grads_abs_head = []
        for layer in model_head.modules():
            if isinstance(layer, ResBasicBlock):
                if blockidx == block_idx:
                    for k1, layer1 in enumerate(layer.modules()):
                        if isinstance(layer1, nn.Conv2d):
                            # pdb.set_trace()
                            if k1==1:
                                # print(k1, " ", layer1)
                                sum = torch.squeeze(torch.sum(layer1.weight.grad.abs().clone(),dim=[1,2,3]))
                                conv1_grads_abs_head.append(sum)
                            elif k1==4:
                                sum = torch.squeeze(torch.sum(layer1.weight.grad.abs().clone(),dim=[1,2,3]))
                                conv2_grads_abs_head.append(sum)                        
                blockidx = blockidx + 1

        conv0_grads = (1-ALPHA)*conv0_grads_abs_tail[0] - ALPHA*(conv0_grads_abs_head[0])        
        conv1_grads = (1-ALPHA)*conv1_grads_abs_tail[0] - ALPHA*(conv1_grads_abs_head[0])
        conv2_grads = (1-ALPHA)*conv2_grads_abs_tail[0] - ALPHA*(conv2_grads_abs_head[0])

        conv0_num_params_to_keep = int(len(conv0_grads) * keep_ratio[0])
        conv1_num_params_to_keep = int(len(conv1_grads) * keep_ratio[1])
        conv2_num_params_to_keep = int(len(conv2_grads) * keep_ratio[2])

        threshold0, _ = torch.topk(conv0_grads, conv0_num_params_to_keep, sorted=True)
        threshold1, _ = torch.topk(conv1_grads, conv1_num_params_to_keep, sorted=True)
        threshold2, _ = torch.topk(conv2_grads, conv2_num_params_to_keep, sorted=True)
        reject_score0 = threshold0[-1] 
        reject_score1 = threshold1[-1] 
        reject_score2 = threshold2[-1] 

        cfg0 = 0
        cfg0_mask = []
        mask = conv0_grads.ge(reject_score0).float().cuda()
        cfg0 = int(torch.sum(mask))
        cfg0_mask.append(mask.clone())

        cfg1 = 0
        cfg1_mask = []
        mask = conv1_grads.ge(reject_score1).float().cuda()
        cfg1 = int(torch.sum(mask))
        cfg1_mask.append(mask.clone())

        cfg2 = 0
        cfg2_mask = []
        mask = conv2_grads.ge(reject_score2).float().cuda()
        cfg2 = int(torch.sum(mask))
        cfg2_mask.append(mask.clone())

        cfg = []
        cfg_mask=[]
        for k, m in enumerate(model.modules()):
            if isinstance(m, nn.BatchNorm2d):  
                # pdb.set_trace()
                cfg.append(m.weight.shape[0])
                cfg_mask.append(torch.ones(m.weight.shape[0]))
        # pdb.set_trace()
        cfg[0] = cfg0
        cfg[1] = cfg1
        cfg[2] = cfg2

        cfg_mask[0] = cfg0_mask[0]
        cfg_mask[1] = cfg1_mask[0]
        cfg_mask[2] = cfg2_mask[0]

        newmodel = resnet_110(cfg=cfg)
        # pdb.set_trace()
        if args.cuda:
            newmodel.cuda()
            newmodel = nn.DataParallel(newmodel) 
        newmodel = transfer_value(cfg_mask, model, newmodel)
        # pdb.set_trace()
        test_model_enable(newmodel)

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
    for k, [m0, m1] in enumerate(zip(model.modules(), newmodel.modules())):
        if isinstance(m0, nn.BatchNorm2d):
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
            # pdb.set_trace()
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
        #     idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
        #     if idx0.size == 1:
        #         idx0 = np.resize(idx0, (1,))
        #     m1.weight.data = m0.weight.data[:, idx0].clone()
        #     m1.bias.data = m0.bias.data.clone()

    for layer in model.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            layer.weight.requires_grad = True
    for layer in newmodel.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            layer.weight.requires_grad = True
    print("transfer value finished.")

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


def prepare_model(model_path='', block_idx=0, model_arch='resnet50'):
    if block_idx == 0:
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
            model.load_state_dict(checkpoint['state_dict'], strict=True) #TODO:'strict= ?'
            model_head.load_state_dict(checkpoint['state_dict'], strict=True)
            model_tail.load_state_dict(checkpoint['state_dict'], strict=True)
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
