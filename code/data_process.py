# -*- coding: utf-8 -*-  
from PIL import Image
import os
import os.path
import hashlib
import numpy as np
import pickle

import cv2
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, utils
from torchvision.datasets import MNIST, CIFAR10, ImageNet, ImageFolder

import pdb


class CIFAR10_DATA_PROCESS(Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(
            self,
            root,
            train=True,
            transform=None,
            target_transform=None,
            download=False,
            seg_num=None,
            seg_idex=None,
            forward=False,
            head=False,
            tail=False
            ):

        super(CIFAR10_DATA_PROCESS, self).__init__()
        self.root = root
        self.train = train  # training set or test set
        self.transform = transform


        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self.seg_num = seg_num
        self.seg_idex = seg_idex

        self.forward = forward

        self.head = head
        self.tail = tail


        self._load_meta()

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            data = pickle.load(infile)
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        # pdb.set_trace()
        img, target = self.data[index], self.targets[index]
        # (Image.fromarray(np.uint8(img))).save('/home2/pengyifan/pyf/freq/temp/before.png') #test

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image

        # img = Image.fromarray(img)

        if self.transform is not None:
            if (self.seg_num is not None) and (self.seg_idex is not None):
                h,w = img.shape[:2]  #改代码是通过cv2.imread改的，(32, 32, 3)，不知道是否匹配
                # print(img.shape)
                bsf = np.ones((h,w,3))    

                if self.seg_num==5:
                    radius = [0,3,6,9,12] 
                    if self.seg_idex==self.seg_num:
                        R = 16 
                        r = 12 
                    else:
                        R = radius[self.seg_idex]
                        r = R-3

                if self.seg_num==4:
                    radius = [0,4,8,12] 
                    if self.seg_idex==self.seg_num:
                        R = 16 
                        r = 12 
                    else:
                        R = radius[self.seg_idex]
                        r = R-4

                if self.seg_num==3:
                    radius = [0,5,10] 
                    if self.seg_idex==self.seg_num:
                        R = 16 
                        r = 10 
                    else:
                        R = radius[self.seg_idex]
                        r = R-5

                if self.seg_num==8:
                    radius = [0, 2, 4, 6, 8, 10, 12, 14] 
                    if self.seg_idex==self.seg_num:
                        R = 16 
                        r = 14 
                    else:
                        R = radius[self.seg_idex]
                        r = R-2

                if self.seg_num==16:
                    radius = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15] 
                    if self.seg_idex==self.seg_num:
                        R = 16 
                        r = 15
                    else:
                        R = radius[self.seg_idex]
                        r = R-1
                
                if self.seg_num==100: #test
                    bsf[3,4,0] = 1

                if self.seg_num!=100:
                    for x in range(w):
                        for y in range(h):
                            # if self.seg_idex==self.seg_num:
                            #     if ((x-(w-1)/2)**2 + (y-(h-1)/2)**2) >= (r**2):
                            #         bsf[y,x,:] = 0 #Band stop filter
                            # else:
                            #     if ((x-(w-1)/2)**2 + (y-(h-1)/2)**2) <= (R**2) and ((x-(w-1)/2)**2 + (y-(h-1)/2)**2) >= (r**2):
                            #         bsf[y,x,:] = 0 #Band stop filter
                            if self.forward==False:
                                if self.seg_idex==self.seg_num:
                                    if ((x-(w-1)/2)**2 + (y-(h-1)/2)**2) >= (r**2):
                                        bsf[y,x,:] = 0 #Band stop filter
                                else:
                                    if ((x-(w-1)/2)**2 + (y-(h-1)/2)**2) <= (R**2) and ((x-(w-1)/2)**2 + (y-(h-1)/2)**2) >= (r**2):
                                        bsf[y,x,:] = 0 #Band stop filter
                            else:
                                if self.head and not self.tail:
                                    if not (((x-(w-1)/2)**2 + (y-(h-1)/2)**2) <= (R**2) and ((x-(w-1)/2)**2 + (y-(h-1)/2)**2) >= (r**2)):
                                        bsf[y,x,:] = 0 
                                elif self.tail and not self.head:
                                    if not (((x-(w-1)/2)**2 + (y-(h-1)/2)**2) >= (r**2)):
                                        bsf[y,x,:] = 0 
                                else:
                                    print("error!")
            
                # pdb.set_trace()
                freq = np.fft.fft2(img,axes=(0,1))
                freq = np.fft.fftshift(freq)
                # (Image.fromarray(np.uint8(bsf*255))).save('/home2/pengyifan/pyf/freq/temp/bsf.png') #test
                # (Image.fromarray(np.uint8(freq))).save('/home2/pengyifan/pyf/freq/temp/freq.png') #test
                # pdb.set_trace()
                f = freq * bsf
                # (Image.fromarray(np.uint8(f[:,:,0]))).save('/home2/pengyifan/pyf/freq/temp/f1.png') #test
                # (Image.fromarray(np.uint8(f[:,:,1]))).save('/home2/pengyifan/pyf/freq/temp/f2.png') #test
                # (Image.fromarray(np.uint8(f[:,:,2]))).save('/home2/pengyifan/pyf/freq/temp/f3.png') #test
                img = np.abs(np.fft.ifft2(np.fft.ifftshift(f),axes=(0,1)))
                img = np.clip(img,0,255)
                img = img.astype('uint8')     
                # (Image.fromarray(np.uint8(img))).save('/home2/pengyifan/pyf/freq/temp/after.png') #test
                # pdb.set_trace()
                img = Image.fromarray(img)
                img = self.transform(img)
            else:
                img = self.transform(img)
            

        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")


def check_integrity(fpath, md5=None):
    if not os.path.isfile(fpath):
        return False
    if md5 is None:
        return True
    return check_md5(fpath, md5)

def calculate_md5(fpath, chunk_size=1024 * 1024):
    md5 = hashlib.md5()
    with open(fpath, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            md5.update(chunk)
    return md5.hexdigest()


def check_md5(fpath, md5, **kwargs):
    return md5 == calculate_md5(fpath, **kwargs)


class CIFAR100_DATA_PROCESS(CIFAR10_DATA_PROCESS):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }


class MNIST_DATA_PROCESS(MNIST):
    
    def __init__(
            self,
            root,
            train=True,
            transform=None,
            target_transform=None,
            download=False,
            seg_num=5,
            seg_idex=1
    ):
        super(MNIST, self).__init__(root)
        self.train = train  # training set or test set

        self.data, self.targets = torch.load("/home2/pengyifan/datasets/MNIST/MNIST/processed/test.pt")

        self.seg_num = seg_num
        self.seg_idex = seg_idex

        self.transform = transform

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        R = 0
        r = 0

    
        

        if self.transform is not None:
            
            h,w = img.shape[:2]  #改代码是通过cv2.imread改的，(32, 32, 3)，不知道是否匹配
            # h = img.shape[0]
            # w = img.shape[1]
            # print(img.shape)
            bsf = np.ones((h,w))    

            if self.seg_num==3:
                radius = [0,6,12] 
                if self.seg_idex==self.seg_num:
                    R = 14 
                    r = 12 
                else:
                    R = radius[self.seg_idex]
                    r = R-6

            if self.seg_num==4:
                radius = [0,4,8,12] 
                if self.seg_idex==self.seg_num:
                    R = 14 
                    r = 12 
                else:
                    R = radius[self.seg_idex]
                    r = R-4

            if self.seg_num==7:
                radius = [0,2,4,6,8,10,12] 
                if self.seg_idex==self.seg_num:
                    R = 14 
                    r = 12 
                else:
                    R = radius[self.seg_idex]
                    r = R-2


            if self.seg_num==14:
                radius = [0,1,2,3,4,5,6,7,8,9,10,11,12,13] 
                if self.seg_idex==self.seg_num:
                    R = 14 
                    r = 13
                else:
                    R = radius[self.seg_idex]
                    r = R-1


            for x in range(w):
                for y in range(h):
                    # if self.seg_idex==self.seg_num:
                    #     if ((x-(w-1)/2)**2 + (y-(h-1)/2)**2) >= (r**2):
                    #         bsf[y,x,:] = 0 #Band stop filter
                    # else:
                    #     if ((x-(w-1)/2)**2 + (y-(h-1)/2)**2) <= (R**2) and ((x-(w-1)/2)**2 + (y-(h-1)/2)**2) >= (r**2):
                    #         bsf[y,x,:] = 0 #Band stop filter
                    if ((x-(w-1)/2)**2 + (y-(h-1)/2)**2) <= (R**2) and ((x-(w-1)/2)**2 + (y-(h-1)/2)**2) >= (r**2):
                        bsf[y,x] = 0 #Band stop filter
            
            # pdb.set_trace()
            freq = np.fft.fft2(img,axes=(0,1))
            freq = np.fft.fftshift(freq)
            f = freq * bsf
            img = np.abs(np.fft.ifft2(f,axes=(0,1)))
            img = np.clip(img,0,255)
            img = img.astype('uint8')     

            img = Image.fromarray(img)
            img = self.transform(img)

        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return img, target








