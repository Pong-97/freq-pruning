import os
import argparse
import numpy as np
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

from tqdm import tqdm
from models.vgg_16_bn import vgg_16_bn
from models.resnet_imagenet import resnet_50
from freq_process_imagenet import processed_forward
from normal_imagenet import Data


from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image

# from models.resnet import Bottleneck
# from imagenet import Data

import pdb

# import warnings
# warnings.filterwarnings("ignore")


# Prune settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
parser.add_argument('--seed', type=int, default=2, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--dataset', type=str, default='imagenet',
                    help='training dataset (default: cifar10)')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--depth', type=int, default=16,
                    help='depth of the vgg')
parser.add_argument('--epochs', type=int, default=120,
                    help='depth of the vgg')
parser.add_argument('--percent', type=float, default=0.5,
                    help='scale sparse rate (default: 0.5)')
# parser.add_argument('--model', default="./checkpoint/cluster/checkpoint1.pth.tar", type=str, metavar='PATH',
#                     help='path to the model (default: none)')

parser.add_argument('--student_epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 160)')  #160 for train

# parser.add_argument('--refine', default="/home2/pengyifan/pyf/hypergraph_cluster/log/temp/0", type=str, metavar='PATH',
#                     help='path to the pruned model to be fine tuned')
parser.add_argument('--save', default="/home2/pengyifan/pyf/freq-lite/logs/temp" , type=str, metavar='PATH',
                    help='path to save prune model (default: current directory)') 


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


def test_model_enable(model):
    inputs = torch.rand((1, 3, 224, 224)).cuda()
    model = model.cuda().train()
    output = model(inputs)
    print("test, output's shape: ", output.shape)


def transfer_value(model,new_model,masks):

    layer_id_in_cfg = 0
    start_mask = torch.ones(3)
    end_mask = masks[layer_id_in_cfg]
    for k, [m0, m1] in enumerate(zip(model.modules(), new_model.modules())):
        # print("\nk: ", k)
        # print(m0)
        # print(m1)
        if k in [19, 55, 102, 171, 18, 54, 101, 170]:
            continue
        if isinstance(m0, nn.BatchNorm2d):
            # if(layer_id_in_cfg <= layer_id):
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))   #np.asarray(end_mask.cpu().numpy()) == 1时的索引
            # pdb.set_trace()
            if idx1.size == 1:
                idx1 = np.resize(idx1,(1,))
            # print("before m1: ", m1.weight.data.shape)
            m1.weight.data = m0.weight.data[idx1.tolist()].clone()
            m1.bias.data = m0.bias.data[idx1.tolist()].clone()
            m1.running_mean = m0.running_mean[idx1.tolist()].clone()
            m1.running_var = m0.running_var[idx1.tolist()].clone()
            # print("after m1: ", m1.weight.data.shape)
            # print("\n")
            layer_id_in_cfg += 1
            start_mask = end_mask.clone()
            if layer_id_in_cfg < len(masks):  # do not change in Final FC
                end_mask = masks[layer_id_in_cfg]
        elif isinstance(m0, nn.Conv2d):
            # if(layer_id_in_cfg <= layer_id):
            # if m1.weight.data.shape[0]==64:
            # pdb.set_trace()
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            # print("layer id: ",layer_id_in_cfg)
            # print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            # pdb.set_trace()
            w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()   #m.weight.data.size()   (out,in,size[0],size[1])
            w1 = w1[idx1.tolist(), :, :, :].clone()
            # print("before m1: ", m1.weight.data.shape)
            # print("after m1: ", w1.shape)
            # pdb.set_trace() 
            m1.weight.data = w1.clone()
        elif isinstance(m0, nn.Linear):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            # pdb.set_trace()
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            m1.weight.data = m0.weight.data[:, idx0].clone()
            m1.bias.data = m0.bias.data.clone()
    # pdb.set_trace()
    print("transfer value finished.")
    # inputs = torch.rand((1, 3, 32, 32)).cuda()
    # new_model = new_model.cuda().train()
    # output = new_model(inputs)
    # print(output.shape)
    return new_model


def test(model, test_loader, optimizer=None):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
        output = model(data)
        # test_loss += F.cross_entropy(output, target, size_average=False).item() # sum up batch loss
        test_loss += F.cross_entropy(output, target, reduction='sum').item()
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        # break

    test_loss /= len(test_loader.dataset)

    if optimizer != None:
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%), learning rate: {}'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset), optimizer.param_groups[0]['lr']))
    else:
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))            
    return test_loss, correct.item() / float(len(test_loader.dataset))
    

def save_checkpoint(state, is_best, filepath, block_idx):
    torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
    if is_best:
        print("is best and save!")
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, 'model_best.pth.tar'))
        shutil.copyfile(os.path.join(filepath, 'model_best.pth.tar'), os.path.join(filepath, 'model_best{}.pth.tar'.format(block_idx)))



def distillation(y, labels, teacher_scores, temp, alpha):
    return nn.KLDivLoss()(F.log_softmax(y / temp, dim=1), F.softmax(teacher_scores / temp, dim=1)) * (temp * temp * 2.0 * alpha) + F.cross_entropy(y, labels) * (1. - alpha)


def train_student_kd(model, teacher_model, train_loader, optimizer, epoch):
    model.train()
    trained_samples = 0
    # pdb.set_trace()
    # for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)

        # teacher_output = teacher_model(data)
        # teacher_output = teacher_output.detach() 
        # loss = distillation(output, target, teacher_output, temp=5.0, alpha=0.7) #default: temp=5.0, alpha=0.7

        loss = F.cross_entropy(output, target)

        loss.backward()
        optimizer.step()

        trained_samples += len(data)
        progress = (batch_idx + 1) / len(train_loader) * 100
        # break
    

def student_kd_main(student_kd_model, cfg, teacher_model, train_loader, test_loader, block_idx):

    # print(student_kd_model)
    num_parameters = sum([param.nelement() for param in student_kd_model.parameters()])
    print("num_parameters: ", num_parameters)
    print(cfg)
    print("=> save model at'{}'".format(args.save))

    epochs = args.student_epochs
    torch.manual_seed(7)
    best_prec1 = 0.

    # print(student_kd_model)
    optimizer_student = optim.SGD(student_kd_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler_student = optim.lr_scheduler.MultiStepLR(optimizer_student, milestones=[int(args.student_epochs*0.5), int(args.student_epochs*0.75)], gamma=0.1)

    student_history = []
    # loss, acc = test(student_kd_model, test_loader, optimizer=optimizer_student)
    # print("micro-fintune, current accurancy: ", acc)
    for epoch in range(1, epochs + 1):
        train_student_kd(student_kd_model, teacher_model, train_loader, optimizer_student, epochs)
        print("\nepoch: ", epoch, "learning rate: ", optimizer_student.param_groups[0]['lr'])
        scheduler_student.step()
        loss, acc = test(student_kd_model, test_loader, optimizer=optimizer_student)
        student_history.append((loss, acc))

        is_best = acc > best_prec1
        best_prec1 = max(acc, best_prec1)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': student_kd_model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer_student.state_dict(),
            'cfg': cfg,
            'block_idx': block_idx,

        }, is_best, filepath=args.save, block_idx=block_idx)

    print("Best accuracy: "+str(best_prec1), '\n')
    return student_kd_model, student_history


def finetune_train(epoch, model, train_loader, optimizer):
    model.train()
    # for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
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
        if batch_idx % 500 == 0:
            # print('\nTrain Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}\t'.format(
            #     epoch, batch_idx * len(data), len(train_loader.dataset),
            #     100. * batch_idx / len(train_loader), loss.item()))
            print('\nTrain Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}\t'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def main():
    if args.cuda:
        torch.cuda.manual_seed(7)

    if not os.path.exists(args.save):
        os.makedirs(args.save) 

    if args.dataset == 'imagenet':
        print("数据集为ILSVRC2012")
        data_tmp = Data()
        train_loader = data_tmp.loader_train
        test_loader = data_tmp.loader_test
 
    teacher_model = resnet_50()
    # teacher_model.cuda()
    # teacher_model = torch.nn.DataParallel(teacher_model)
    # teacher_model_path = ""
    # print("=> loading teacher model checkpoint '{}'".format(teacher_model_path))
    # checkpoint = torch.load(teacher_model_path, map_location='cpu')
    # teacher_model.load_state_dict(checkpoint['state_dict'], strict=True)
    # teacher_model.cuda()
    # teacher_model = torch.nn.DataParallel(teacher_model)
    # loss, acc = test(teacher_model, test_loader)
    # print("teacher model accurancy: ", acc)


    tar_cfg = []
    cfg =  [64, 64, 256, 64, 64, 256, 64, 64, 256, 128, 128, 512, 128, 128, 512, 128, 128, 512, 128, 128, 512, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 512, 512, 2048, 512, 512, 2048, 512, 512, 2048]  
    # cfg = [item for sub_list in cfg for item in sub_list]  
    # compress_rate = [0.25, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45]
    # compress_rate = [0.5, 0.5, 0.5]*16
    compress_rate = [0.5, 0.5, 0.296875, 0.5, 0.5, 0.296875, 0.5, 0.5, 0.296875, 0.5, 0.5, 0.296875, 0.5, 0.5, 0.296875, 0.5, 0.5, 0.296875, 0.5, 0.5, 0.296875, 0.5, 0.5, 0.296875, 0.5, 0.5, 0.296875, 0.5, 0.5, 0.296875, 0.5, 0.5, 0.296875, 0.5, 0.5, 0.296875, 0.5, 0.5, 0.296875, 0.5, 0.5, 0.0, 0.5, 0.5, 0.0, 0.5, 0.5, 0.0]
    # [32, 32, 180, 32, 32, 180, 32, 32, 180, 64, 64, 360, 64, 64, 360, 64, 64, 360, 64, 64, 360, 128, 128, 720, 128, 128, 720, 128, 128, 720, 128, 128, 720, 128, 128, 720, 128, 128, 720, 256, 256, 2048, 256, 256, 2048, 256, 256, 2048]
    for i in range(len(cfg)):
        tar_cfg.append(int(cfg[i]*(1-compress_rate[i])))
    print(tar_cfg)
    ################################  not refine  ################################  
    for i in range(16):
        print("##############################################################################")
        keep_rate = [1-compress_rate[3*i], 1-compress_rate[3*i+1], 1-compress_rate[3*i+2]]
        cfg, newmodel = processed_forward(os.path.join(args.save, 'model_best.pth.tar'), keep_rate, block_idx=i, model_arch='resnet50')         
        student_kd_model = newmodel
        print("block index: ", i+1, "/16", )
        student_kd_model, student_kd_history = student_kd_main(student_kd_model, cfg, teacher_model, train_loader, test_loader, block_idx=i)
        # pdb.set_trace()
    ##############################################################################

    # # ################################  refine  ################################  
    # checkpoint = torch.load(os.path.join(args.save, 'model_best.pth.tar'), map_location='cpu')
    # finished_cfg = checkpoint['cfg']
    # train_block_idx = checkpoint['block_idx'] + 1 #TODO:要不要加1?
    # # pdb.set_trace()
    # for i in range(3*(train_block_idx-1)):
    #     cfg[i] = finished_cfg[i]
    # for i in range(train_block_idx-1, 16):
    #     keep_rate = [1-compress_rate[3*i], 1-compress_rate[3*i+1], 1-compress_rate[3*i+2]]
    #     cfg, newmodel = processed_forward(os.path.join(args.save, 'model_best.pth.tar'), keep_rate, block_idx=i, model_arch='resnet50')         
    #     student_kd_model = newmodel
    #     print("block index: ", i+1)
    #     student_kd_model, student_kd_history = student_kd_main(student_kd_model, cfg, teacher_model, train_loader, test_loader, block_idx=i)
    # # ##########################################################################  

    print("=> test, loading student model checkpoint '{}'".format(os.path.join(args.save, 'model_best.pth.tar')))
    checkpoint = torch.load(os.path.join(args.save, 'model_best.pth.tar'), map_location='cpu')
    print(checkpoint['cfg'])
    test_model = resnet_50(cfg=checkpoint['cfg'])
    test_model.cuda()
    test_model = torch.nn.DataParallel(test_model)
    test_model.load_state_dict(checkpoint['state_dict'], strict=True)
    loss, acc = test(test_model, test_loader)
    print("test model accurancy: ", acc, 'path:', os.path.join(args.save, 'model_best.pth.tar'))
    # inputs = torch.randn(1,3,32,32).cuda()
    # flops, params = profile(newmodel, inputs=(inputs, ) )  #  profile（模型，输入数据）
    # print("FLOPS: ", flops, "\nPARAMS:", params)

    print("###################  Finetune  ###################")
    finetune_model = test_model
    finetune_model_cfg = checkpoint['cfg']
    fintune_optimizer = optim.SGD(finetune_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    fintune_scheduler = optim.lr_scheduler.MultiStepLR(fintune_optimizer, milestones=[args.epochs*0.5, args.epochs*0.75], gamma=0.1)

    best_prec1 = 0.
    for epoch in range(args.epochs):
        finetune_train(epoch, finetune_model, train_loader, fintune_optimizer)
        fintune_scheduler.step()
        loss, acc = test(finetune_model, test_loader, fintune_optimizer)
        is_best = acc > best_prec1
        best_prec1 = max(acc, best_prec1)


        save_checkpoint({
            'state_dict': finetune_model.state_dict(),
            'cfg': finetune_model_cfg,
            'epoch': epoch + 1,
            'best_prec1': best_prec1,
            'optimizer': fintune_optimizer.state_dict(),
        }, is_best, filepath=args.save, block_idx='finetune{}'.format(epoch+1))

if __name__ == '__main__':
    main()