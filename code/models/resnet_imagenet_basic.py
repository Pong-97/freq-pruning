import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision.models import resnet18
import numpy as np
import pdb

from thop import profile


norm_mean, norm_var = 0.0, 1.0
# cfg = [[64], [64, 64, 64], [256, 64, 64]*2, [256, 128, 128], [512, 128, 128]*3, [512, 256, 256], [1024, 256, 256]*5, [1024, 512, 512], [2048, 512, 512]*2]
# defaultresnet50cfg = [item for sub_list in cfg for item in sub_list] 
defaultresnet50cfg = [64, 64, 256, 64, 64, 256, 64, 64, 256, 128, 128, 512, 128, 128, 512, 128, 128, 512, 128, 128, 512, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 512, 512, 2048, 512, 512, 2048, 512, 512, 2048]  
defaultresnet18cfg = [64, 64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512]
defaultresnet34cfg = [64, 64, 64, 64, 64, 64, 128, 128, 128, 128, 128, 128, 128, 128, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, ]
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class ResBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, compress_rate=[0.], tmp_name=None, not_first_block=0):
        super(ResBasicBlock, self).__init__()
        keep_rate1 = 1-compress_rate[0]
        self.conv1 = nn.Conv2d(inplanes, int(planes*keep_rate1), kernel_size=3, bias=False, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(int(planes*keep_rate1))
        self.relu = nn.ReLU(inplace=True)

        keep_rate2 = 1-compress_rate[1]
        self.conv2 = nn.Conv2d(int(planes*keep_rate1), int(planes*keep_rate2), kernel_size=3, stride=1, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(int(planes*keep_rate2))
        self.downsample = downsample
        self.stride = stride
        if not_first_block == 1:
            self.downsample = nn.Sequential()
            if inplanes != int(planes*keep_rate2):
                if inplanes < int(planes*keep_rate2):
                    gap = int(planes*keep_rate2) - inplanes
                    self.downsample = LambdaLayer(
                        lambda x: F.pad(x[:, :, :, :], (0, 0, 0, 0, gap//2, gap-gap//2), "constant", 0)
                        )

                elif inplanes > int(planes*keep_rate2):
                    gap_scale = inplanes // int(planes*keep_rate2)
                    after_slice = int(np.ceil(inplanes / (gap_scale+1)))
                    gap = int(planes*keep_rate2) - after_slice
                    self.downsample = LambdaLayer(
                        lambda x: F.pad(x[:, ::(gap_scale+1), :, :], (0, 0, 0, 0, gap//2, gap-gap//2), "constant", 0)
                        )   

    def forward(self, x):
        # pdb.set_trace()
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, compress_rate=[0.], tmp_name=None, not_first_block=0):
        super(ResBottleneck, self).__init__()

        keep_rate1 = 1-compress_rate[0]
        self.conv1 = nn.Conv2d(inplanes, int(planes*keep_rate1), kernel_size=1, bias=False)
        self.conv1.compress_rate = compress_rate[0]
        self.conv1.tmp_name = tmp_name
        self.bn1 = nn.BatchNorm2d(int(planes*keep_rate1))
        self.relu1 = nn.ReLU(inplace=True)

        keep_rate2 = 1-compress_rate[1]
        self.conv2 = nn.Conv2d(int(planes*keep_rate1), int(planes*keep_rate2), kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(int(planes*keep_rate2))
        self.conv2.compress_rate = compress_rate[1]
        self.conv2.tmp_name = tmp_name
        self.relu2 = nn.ReLU(inplace=True)

        keep_rate3 = 1-compress_rate[2]
        self.conv3 = nn.Conv2d(int(planes*keep_rate2), int(planes*keep_rate3 * self.expansion), kernel_size=1, bias=False)
        self.conv3.compress_rate = compress_rate[2]
        self.conv3.tmp_name = tmp_name
        self.bn3 = nn.BatchNorm2d(int(planes*keep_rate3 * self.expansion))
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = downsample
        if not_first_block == 1:
            self.downsample = nn.Sequential()
            if inplanes != int(planes*keep_rate3 * self.expansion):
                if inplanes < int(planes*keep_rate3 * self.expansion):
                    gap = int(planes*keep_rate3 * self.expansion) - inplanes
                    self.downsample = LambdaLayer(
                        lambda x: F.pad(x[:, :, :, :], (0, 0, 0, 0, gap//2, gap-gap//2), "constant", 0)
                        )

                elif inplanes > int(planes*keep_rate3 * self.expansion):
                    gap_scale = inplanes // int(planes*keep_rate3 * self.expansion)
                    after_slice = int(np.ceil(inplanes / (gap_scale+1)))
                    gap = int(planes*keep_rate3 * self.expansion) - after_slice
                    self.downsample = LambdaLayer(
                        lambda x: F.pad(x[:, ::(gap_scale+1), :, :], (0, 0, 0, 0, gap//2, gap-gap//2), "constant", 0)
                        )                      

        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        # residual = self.downsample(x)

        out += residual
        out = self.relu3(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, num_layers, num_blocks, cfg, num_classes=1000):
        super(ResNet, self).__init__()
        if cfg ==None:
            if num_layers == 18:
                cfg =defaultresnet18cfg
                compress_rate = [0.0]*16

            elif num_layers == 34:
                cfg =defaultresnet34cfg
                compress_rate = [0.0]*32

        
        if num_layers == 18:
            compress_rate = [0.0]*16
            for i in range(len(compress_rate)):
                compress_rate[i] = (defaultresnet18cfg[i]-cfg[i])/defaultresnet18cfg[i]

        if num_layers == 34:
            compress_rate = [0.0]*32
            for i in range(len(compress_rate)):
                compress_rate[i] = (defaultresnet34cfg[i]-cfg[i])/defaultresnet34cfg[i]

        self.compress_rate = compress_rate
        self.num_layers = num_layers
        # pdb.set_trace()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, compress_rate=compress_rate[0 : 2*num_blocks[0]], tmp_name='layer1')
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, compress_rate=compress_rate[2*num_blocks[0] : 2*num_blocks[0]+2*num_blocks[1]], tmp_name='layer2')
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, compress_rate=compress_rate[2*num_blocks[0]+2*num_blocks[1] : 2*num_blocks[0]+2*num_blocks[1]+2*num_blocks[2]], tmp_name='layer3')
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, compress_rate=compress_rate[2*num_blocks[0]+2*num_blocks[1]+2*num_blocks[2] : ], tmp_name='layer4')
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(int(512 * block.expansion * (1-compress_rate[-1])), num_classes)

        self.initialize()


    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, planes, blocks, stride, compress_rate, tmp_name):
        downsample = None
        keep_rate_shortcut = 1-compress_rate[2]
        # pdb.set_trace()
        if stride != 1 or self.inplanes != int(planes * block.expansion * keep_rate_shortcut):
            conv_short = nn.Conv2d(self.inplanes, int(planes * block.expansion * keep_rate_shortcut),
                                   kernel_size=1, stride=stride, bias=False)
            conv_short.compress_rate = compress_rate[2]
            conv_short.tmp_name = tmp_name + '_shortcut'
            downsample = nn.Sequential(
                conv_short,
                nn.BatchNorm2d(int(planes * block.expansion * keep_rate_shortcut)),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, compress_rate=compress_rate[0:2],
                            tmp_name=tmp_name + '_block' + str(1)))
        keep_rate_block1 = 1 - compress_rate[1]
        self.inplanes = int(planes * block.expansion * keep_rate_block1)
        for i in range(1, blocks):

            layers.append(block(self.inplanes, planes, compress_rate=compress_rate[2*i : 2*i+2],
                                tmp_name=tmp_name + '_block' + str(i + 1), not_first_block=1))
            keep_rate_blockn = 1 - compress_rate[2*i+1]
            self.inplanes = int(planes * block.expansion * keep_rate_blockn)
        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)

        # 256 x 56 x 56
        x = self.layer2(x)

        # 512 x 28 x 28
        x = self.layer3(x)

        # 1024 x 14 x 14
        x = self.layer4(x)

        # 2048 x 7 x 7
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(pretrained=False, cfg=None):
    model = ResNet(ResBasicBlock, 18, [2, 2, 2, 2], cfg=cfg)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, cfg=None):
    model = ResNet(ResBasicBlock, 34, [3, 4, 6, 3], cfg=cfg)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def main():
    model = resnet18()    
    input = torch.randn(1, 3, 224, 224)
    flops_base18, params_base18 = profile(model, inputs=(input, ) )  #  profile（模型，输入数据）
    model = resnet18(cfg = [40, 40, 40, 40, 80, 80, 80, 80, 180, 180, 180, 180, 400, 400, 400, 400])
    # FLOPS:  865183760.0 
    # PARAMS: 6735080.0 
    # ^FLOPs: 0.5025568094006615
    flops_18, params_18 = profile(model, inputs=(input, ) )  #  profile（模型，输入数据）
    print("FLOPS: ", flops_18, "\nPARAMS:", params_18, "\n^FLOPs:", (flops_base18-flops_18)/flops_base18)
    
    model = model.cuda()
    model = nn.DataParallel(model) 
    print(model)
    for k,m in enumerate(model.modules()):
        print("k:", k, "\nm:", m)

    # for layer in model.modules():
    #     if isinstance(layer, ResBasicBlock):
    #         for k1, layer1 in enumerate(layer.modules()):
    #             if isinstance(layer1, nn.Conv2d):
    #                     print(k1, " ", layer1)
                      









    # model = resnet34()   
    # flops_base34, params_base34 = profile(model, inputs=(input, ) )  #  profile（模型，输入数据）
    # model = resnet34(cfg=[40, 40, 40, 40, 40, 40, 100, 100, 100, 100, 100, 100, 100, 100, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 380, 380, 380, 380, 380, 380])
    # # FLOPS:  2046571380.0 
    # # PARAMS: 12574160.0
    # flops_34, params_34 = profile(model, inputs=(input, ) )  #  profile（模型，输入数据）
    # print("FLOPS: ", flops_34, "\nPARAMS:", params_34, "\n^FLOPs:", (flops_base34-flops_34)/flops_base34)








    # print(model)
    # cfg1=[int(cfg[i]-(cfg[i]*compress_rate[i])) for i in range(len(cfg))]
    # model = model.cuda()
    # model = nn.DataParallel(model) 
    # print(model)
    # inputs = torch.rand((1, 3, 224, 224)).cuda()
    # model = model.cuda().train()
    # output = model(inputs)
    # print(output.shape)
    # for k,m in enumerate(model.modules()):
    #     print("k: ", k, "\tm: ", m)
    # for k, m in enumerate(model.modules()):
    #     if isinstance(m, nn.BatchNorm2d):
    #         print(k, " ", m)
    #         pdb.set_trace()

if __name__ == '__main__':
    main()

