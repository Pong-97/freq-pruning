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
from models.resnet import Bottleneck

from imagenet_data_process import IMAGENET_DATA_PROCESS
from models.resnet_imagenet import ResBottleneck
from data_process import CIFAR10_DATA_PROCESS, CIFAR100_DATA_PROCESS, MNIST_DATA_PROCESS
from models import *
from models.resnet_imagenet import resnet_50

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


def test_model_enable(model):
    inputs = torch.rand((1, 3, 224, 224)).cuda()
    model = model.cuda().train()
    output = model(inputs)
    print("test, output's shape: ", output.shape)

def processed_forward(model_path, keep_ratio, dataset='imagenet', seg_num=8, block_idx=0, model_arch='resnet50', ):

    model, model_head, model_tail = prepare_model(model_path, block_idx, model_arch=model_arch)

    # for tail
    # print("Aphla: ", ALPHA)
    print("process for tail.")
    kwargs = {'num_workers':0, 'pin_memory': True} if args.cuda else {}
    if args.dataset == 'imagenet':
        val_dataset_path = '/repository2/linhang/data/ImageNet/ILSVRC2012_img_val/'
        val_dataset = IMAGENET_DATA_PROCESS(
            root=val_dataset_path,
            transform=transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225])]),
            seg_num=seg_num, seg_idex=8 ,
            forward=True, tail=True                          
            )

        loader_tail = torch.utils.data.DataLoader(val_dataset,
                                                batch_size=1,
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
    conv3_grads_abs_tail = []
    for layer in model_tail.modules():
        if isinstance(layer, ResBottleneck):
            if blockidx == block_idx:
                for k1, layer1 in enumerate(layer.modules()):
                    if isinstance(layer1, nn.Conv2d):
                        if k1==1:
                            sum = torch.squeeze(torch.sum(layer1.weight.grad.abs().clone(),dim=[1,2,3]))
                            conv1_grads_abs_tail.append(sum)
                        if k1==4:
                            sum = torch.squeeze(torch.sum(layer1.weight.grad.abs().clone(),dim=[1,2,3]))
                            conv2_grads_abs_tail.append(sum)
                        if k1==7:
                            sum = torch.squeeze(torch.sum(layer1.weight.grad.abs().clone(),dim=[1,2,3]))
                            conv3_grads_abs_tail.append(sum)
            blockidx = blockidx + 1
    
    # for head
    print("process for head.")
    kwargs = {'num_workers':16, 'pin_memory': True} if args.cuda else {}
    if args.dataset == 'imagenet':
        val_dataset_path = '/repository2/linhang/data/ImageNet/ILSVRC2012_img_val/'
        val_dataset = IMAGENET_DATA_PROCESS(
            root=val_dataset_path,
            transform=transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225])]),
            seg_num=seg_num, seg_idex=1 ,
            forward=True, head=True                          
            )

        loader_head = torch.utils.data.DataLoader(val_dataset,
                                                batch_size=1,
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
    conv3_grads_abs_head = []
    for layer in model_head.modules():
        if isinstance(layer, ResBottleneck):
            if blockidx == block_idx:
                for k1, layer1 in enumerate(layer.modules()):
                    if isinstance(layer1, nn.Conv2d):
                        if k1==1:
                            # print(k1, " ", layer1)
                            sum = torch.squeeze(torch.sum(layer1.weight.grad.abs().clone(),dim=[1,2,3]))
                            conv1_grads_abs_head.append(sum)
                        elif k1==4:
                            sum = torch.squeeze(torch.sum(layer1.weight.grad.abs().clone(),dim=[1,2,3]))
                            conv2_grads_abs_head.append(sum)                        
                        elif k1==7:
                            sum = torch.squeeze(torch.sum(layer1.weight.grad.abs().clone(),dim=[1,2,3]))
                            conv3_grads_abs_head.append(sum)
            blockidx = blockidx + 1
            
    conv1_grads = (1-ALPHA)*conv1_grads_abs_tail[0] - ALPHA*(conv1_grads_abs_head[0])
    conv2_grads = (1-ALPHA)*conv2_grads_abs_tail[0] - ALPHA*(conv2_grads_abs_head[0])
    conv3_grads = (1-ALPHA)*conv3_grads_abs_tail[0] - ALPHA*(conv3_grads_abs_head[0])

    conv1_num_params_to_keep = int(len(conv1_grads) * keep_ratio[0])
    conv2_num_params_to_keep = int(len(conv2_grads) * keep_ratio[1])
    conv3_num_params_to_keep = int(len(conv3_grads) * keep_ratio[2])

    threshold1, _ = torch.topk(conv1_grads, conv1_num_params_to_keep, sorted=True)
    threshold2, _ = torch.topk(conv2_grads, conv2_num_params_to_keep, sorted=True)
    threshold3, _ = torch.topk(conv3_grads, conv3_num_params_to_keep, sorted=True)
    reject_score1 = threshold1[-1] 
    reject_score2 = threshold2[-1] 
    reject_score3 = threshold3[-1] 

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

    cfg3 = 0
    cfg3_mask = []
    mask = conv3_grads.ge(reject_score3).float().cuda()
    cfg3 = int(torch.sum(mask))
    cfg3_mask.append(mask.clone())

    cfg = []
    cfg_mask=[]
    for k, m in enumerate(model.modules()):
        if isinstance(m, nn.BatchNorm2d) and (k not in [19, 55, 102, 171]) and k>3:  
            # pdb.set_trace()
            cfg.append(m.weight.shape[0])
            cfg_mask.append(torch.ones(m.weight.shape[0]))

    cfg[3*block_idx] = cfg1
    cfg[3*block_idx+1] = cfg2
    cfg[3*block_idx+2] = cfg3

    cfg_mask[3*block_idx] = cfg1_mask[0]
    cfg_mask[3*block_idx+1] = cfg2_mask[0]
    cfg_mask[3*block_idx+2] = cfg3_mask[0]

    newmodel = resnet_50(cfg=cfg)
    # pdb.set_trace()
    if args.cuda:
        newmodel.cuda()
        newmodel = nn.DataParallel(newmodel) 
    newmodel = transfer_value(cfg_mask, model, newmodel, block_idx)
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


def transfer_value(cfg_mask, model, newmodel, block_idx):

    layer_id_in_cfg = 0
    start_mask = torch.ones(int(64))
    end_mask = cfg_mask[layer_id_in_cfg]
    for k, [m0, m1] in enumerate(zip(model.modules(), newmodel.modules())):
        # print("\nk: ", k)
        # print(m0)
        # print(m1)
        if k in [18, 54, 101, 170]:
            if (k==18 and block_idx==0) or (k==54 and block_idx==3) or (k==101 and block_idx==7) or (k==170 and block_idx==13):
                continue
            else:
                w1 = m0.weight.data[:m1.weight.data.shape[0], :, :, :].clone()
                w1 = w1[:, :m1.weight.data.shape[1], :, :].clone()
                m1.weight.data = w1.clone()
        elif k in [19, 55, 102, 171]:
            if (k==19 and block_idx==0) or (k==55 and block_idx==3) or (k==102 and block_idx==7) or (k==171 and block_idx==13):
                continue
            else:
                m1.weight.data = m0.weight.data[:m1.weight.data.shape[0]].clone()
                m1.bias.data = m0.bias.data[:m1.weight.data.shape[0]].clone()
                m1.running_mean = m0.running_mean[:m1.weight.data.shape[0]].clone()
                m1.running_var = m0.running_var[:m1.weight.data.shape[0]].clone()                
        elif isinstance(m0, nn.BatchNorm2d) and k>3:
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
        elif isinstance(m0, nn.Conv2d) and k>2:
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


def prepare_model(model_path='', layer_idx=0, model_arch='resnet50'):

    if model_arch=='resnet50':
        if layer_idx == 0:
            model = resnet_50()
            model_head = resnet_50()
            model_tail = resnet_50()
            ori_model_path = "/home2/pengyifan/pyf/hypergraph_cluster/log/pretrained_model/resnet50-19c8e357.pth"
            if os.path.isfile(ori_model_path):
                print("=> loading checkpoint '{}'".format(ori_model_path))
                checkpoint = torch.load(ori_model_path, map_location='cpu')
                # pdb.set_trace()
                model.load_state_dict(checkpoint, strict=True) #TODO:'strict= ?'
                model_head.load_state_dict(checkpoint, strict=True)
                model_tail.load_state_dict(checkpoint, strict=True)
                model.cuda()
                model = nn.DataParallel(model) 
                model_head.cuda()
                model_head = nn.DataParallel(model_head) 
                model_tail.cuda()
                model_tail = nn.DataParallel(model_tail) 
                print("=> loaded checkpoint '{}'  ".format(ori_model_path))
        
        else:
            checkpoint = torch.load(model_path, map_location='cpu')
            model_cfg = checkpoint['cfg']
            best_prec1 = checkpoint['best_prec1']
            model = resnet_50(cfg=model_cfg)
            model_head = resnet_50(cfg=model_cfg)
            model_tail = resnet_50(cfg=model_cfg)
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
