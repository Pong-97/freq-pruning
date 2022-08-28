import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader


class Data:
    def __init__(self, is_evaluate=False):
        pin_memory = True
        # if args.gpu is not None:   #如果使用gpu的话，pin_memory为True
        #     pin_memory = True
            
        scale_size = 224
        train_data_dir = "/repository2/linhang/data/ImageNet/ILSVRC2012_img_train/"
        val_data_dir = "/repository2/linhang/data/ImageNet/ILSVRC2012_img_val/"
        # traindir = os.path.join(train_data_dir, 'train')
        # valdir = os.path.join(val_data_dir, 'val')
        traindir = train_data_dir
        valdir = val_data_dir

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if not is_evaluate:
            trainset = datasets.ImageFolder(
                traindir,
                transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.Resize(scale_size),
                    transforms.ToTensor(),
                    normalize,
                ]))

            self.loader_train = DataLoader(
                trainset,
                batch_size=256, #512
                shuffle=True,
                num_workers=32,
                pin_memory=pin_memory)

        testset = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.Resize(scale_size),
                transforms.ToTensor(),
                normalize,
            ]))

        self.loader_test = DataLoader(
            testset,
            batch_size=32, #128
            shuffle=False,
            num_workers=32,
            pin_memory=True)
