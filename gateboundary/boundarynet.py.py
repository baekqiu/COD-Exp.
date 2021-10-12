import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image 
import cv2
import numpy as np
from res2next import res2next50
from res2net import res2net50_26w_4s
from torch.nn import functional as F


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        #self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        #x = self.relu(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_plane, ratio =16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_plane, in_plane // 16, 1, bias = False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_plane // 16, in_plane, 1, bias = False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)



class NewChannelAttention(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(NewChannelAttention, self).__init__()
        self.outchannel = out_channel
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear1 = nn.Sequential(nn.Linear(out_channel, out_channel//4, bias=False), nn.ReLU())
        self.linear2 = nn.Sequential(nn.Linear(out_channel//4, out_channel, bias=False), nn.Sigmoid())

        self.conv = nn.Sequential(nn.Conv2d(in_channel, out_channel, 1, 1, bias=False),
                                  nn.BatchNorm2d(out_channel),
                                  nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False),
                                   nn.BatchNorm2d(out_channel))

    def forward(self, x):
        n, c, h, w = x.shape
        x = self.conv(x)
        y1 = self.pool(x)
        y1 = y1.reshape((n, -1))
        y1 = self.linear1(y1)
        y1 = self.linear2(y1)
        y1 = y1.reshape((n, self.outchannel, 1, 1))

        y1 = y1.expand_as(x).clone()
        y = x * y1
        return F.relu(y + self.conv2(y))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size = 7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in(3,7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim = True)
        max_out, _ = torch.max(x, dim=1, keepdim = True)
        x = torch.cat([avg_out,max_out], dim = 1)
        x = self.conv1(x)
        return self.sigmoid(x)

class GatedConv(nn.Conv2d):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels, 1, bias=False)
        self.attention = nn.Sequential(
            NewChannelAttention(512, 256),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels , out_channels, 1),
            nn.ReLU(),
            nn.Conv2d(out_channels , 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.conv1 = BasicConv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, feat, gate):
        #print(feat.shape)
        #print(gate.shape)
        attention = self.attention(torch.cat((feat, gate), dim=1))
        out = self.conv1(feat * (attention + 1))
        return out


class CFM(nn.Module):
    def __init__(self, channel):
        super(CFM, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample3 = BasicConv2d(512, 512, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(1024, 1024, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2048, 2048, 3, padding=1)
        self.conv1 = nn.Conv2d(64, channel, 1, 1)
        self.conv_concat1 = BasicConv2d(channel*5, channel*5, 3, padding=1)
        self.conv_concat2 = BasicConv2d(channel*2, channel*2, 3,padding=1)
        self.conv3 = nn.Conv2d(512, channel, 1, 1)
        self.conv4 = nn.Conv2d(1024, channel, 1, 1)
        self.conv5 = nn.Conv2d(2048, channel, 1, 1)
        self.conv2 = nn.Conv2d(512, channel, 1, 1)

    def forward(self, x1, x2, x3, x4, x5, channel = 256):
        x1_1 = self.conv1(x1)
        x2_1 = x2
        x3_1 = self.conv3(self.conv_upsample3(self.upsample(x3)))
        x4_1 = self.conv4(self.conv_upsample4(self.upsample(self.upsample((x4)))))
        x5_1 = self.conv5(self.conv_upsample5(self.upsample(self.upsample(self.upsample(x5)))))

        x = torch.cat((x1_1, x2_1, x3_1, x4_1, x5_1), 1)  #88*88*(256*5)
        x = self.conv_concat1(x)
        #print(x.shape)
        f1 = torch.reshape(x, [-1, 5, 256, 88, 88])
        #print(f1.shape)
        f1 = torch.transpose(f1, 1, 2) 
        #print(f1.shape)
        f1 = torch.reshape(f1, [-1, 1280, 88, 88])
        #print(f1.shape)
        x1_2 = f1[:, 0:256, :, :]
        x2_2 = f1[:, 256:512, :, :]
        x3_2 = f1[:, 512:768, :, :]
        x4_2 = f1[:, 768:1024, :, :]
        x5_2 = f1[:, 1024:1280, :, :]
        x1_3 = self.conv2(self.conv_concat2(torch.cat((x1_1, x1_2), 1)))
        x2_3 = self.conv2(self.conv_concat2(torch.cat((x2_1, x2_2), 1)))
        x3_3 = self.conv2(self.conv_concat2(torch.cat((x3_1, x3_2), 1)))
        x4_3 = self.conv2(self.conv_concat2(torch.cat((x4_1, x4_2), 1)))
        x5_3 = self.conv2(self.conv_concat2(torch.cat((x5_1, x5_2), 1)))


        #return f2
        return x1_3, x2_3, x3_3, x4_3, x5_3

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
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


class Boundarydetection(nn.Module):
    def __init__(self):
        super(Boundarydetection, self).__init__()
        
        self.resnet = res2next50()

        self.CFM = CFM(256)
        self.MA_unit_CA = ChannelAttention(256)
        self.MA_unit_SA = SpatialAttention()
        self.toconv = BasicConv2d(512, 256, 1)

        self.resblock1 = BasicBlock(256, 256)
        self.resblock2 = BasicBlock(256, 256)
        self.resblock3 = BasicBlock(256, 256)
        self.conv1_7 = BasicConv2d(3, 256, kernel_size=(1,7), stride=4, padding=(0,3))
        '''self.to_1_conv1 = nn.Conv2d(128, 1, 1)
        self.to_1_conv2 = nn.Conv2d(256, 1, 1)
        self.to_1_conv3 = nn.Conv2d(256, 1, 1)'''
        self.branch1_conv = nn.Conv2d(256, 1, 1)
        self.branch1_conv2 = BasicConv2d(4, 8, 1)
        self.conv7_1 = BasicConv2d(3, 256, kernel_size=(7,1), stride=4, padding=(3,0))
        self.branch2_conv = nn.Conv2d(256, 1, 1)
        self.branch2_conv2 = BasicConv2d(4, 8, 1)

        self.gateconv1 = GatedConv(512, 256)
        self.gateconv2 = GatedConv(512, 256)
        self.gateconv3 = GatedConv(512, 256)

        self.conv = BasicConv2d(16,16,3,padding=1)
        self.conv1 = BasicConv2d(16, 1, 1)  
        self.downchannel = NewChannelAttention(16, 16)
        self.upsample_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        if self.training:
            self.initialize_weights()

    def forward(self, x):
        #--------feature abstraction-------------------
        x1 = self.resnet.conv1(x)
        x1 = self.resnet.bn1(x1)
        x1 = self.resnet.relu(x1)

        x1 = self.resnet.maxpool1(x1)
        x2 = self.resnet.layer1(x1)
        x3 = self.resnet.layer2(x2)
        x4 = self.resnet.layer3_1(x3)
        x5 = self.resnet.layer4_1(x4)

        res1_2, res2_2, res3_2, res4_2, res5_2 = self.CFM(x1, x2, x3, x4, x5)

        res3_3 = self.toconv(torch.cat((self.MA_unit_CA(res3_2)*res3_2, self.MA_unit_SA(res3_2)*res3_2), dim=1))
        res4_3 = self.toconv(torch.cat((self.MA_unit_CA(res4_2)*res4_2, self.MA_unit_SA(res4_2)*res4_2), dim=1))
        res5_3 = self.toconv(torch.cat((self.MA_unit_CA(res5_2)*res5_2, self.MA_unit_SA(res5_2)*res5_2), dim=1))

        #-------soble算子------------------------------

        edge_x = x
        conv_op = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
        soble_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')/3
        soble_kernel = soble_kernel.reshape((1, 1, 3, 3))
        soble_kernel = np.repeat(soble_kernel, 3, axis=1)
        soble_kernel = np.repeat(soble_kernel, 3, axis=0)
        conv_op.weight.data = torch.from_numpy(soble_kernel).cuda()
        edge_detect = conv_op(edge_x)

        branch1_x_1 = self.conv1_7(x) #d-128
        branch1_x_2 = self.resblock1(branch1_x_1) #128*88*88
        #branch1_x_2 = self.to_1_conv1(branch1_x_2) # 1
        #print(res3_3.shape, branch1_x_2.shape)
        branch1_x_2 = self.gateconv1(res3_3, branch1_x_2) # 256
        branch1_x_3 = self.resblock2(branch1_x_2)#256*88*88
        #branch1_x_3 = self.to_1_conv2(branch1_x_3)
        branch1_x_3 = self.gateconv2(res4_3, branch1_x_3)
        branch1_x_4 = self.resblock3(branch1_x_3)
        #branch1_x_4 = self.to_1_conv3(branch1_x_4)
        branch1_x_4 = self.gateconv3(res5_3, branch1_x_4)
        branch1_x_5 = self.branch1_conv(branch1_x_4)
        branch1_x_5 = self.upsample_4(branch1_x_5)
        branch1_x = self.branch1_conv2(torch.cat((edge_detect, branch1_x_5), 1))

        branch2_x_1 = self.conv7_1(x)
        branch2_x_2 = self.resblock1(branch2_x_1)
        #branch2_x_2 = self.to_1_conv1(branch2_x_2)
        branch2_x_2 = self.gateconv1(res3_3, branch2_x_2)
        branch2_x_3 = self.resblock2(branch2_x_2)
        #branch2_x_3 = self.to_1_conv2(branch2_x_3)
        branch2_x_3 = self.gateconv3(res4_3, branch2_x_3)
        branch2_x_4 = self.resblock2(branch2_x_3)
        #branch2_x_4 = self.to_1_conv3(branch2_x_4)
        branch2_x_4 = self.gateconv3(res5_3, branch2_x_4)
        branch2_x_5 = self.branch2_conv(branch2_x_4)
        branch2_x_5 = self.upsample_4(branch2_x_5)
        branch2_x = self.branch2_conv2(torch.cat((edge_detect, branch2_x_5), 1))
        result = torch.cat((branch1_x, branch2_x), 1)

        res = self.conv1(self.downchannel(self.conv(result)))
        return res

    def initialize_weights(self):
        model = res2net50_26w_4s(pretrained = True)
        pretrained_dict = model.state_dict()      
        all_params = {}

        for k, v in self.resnet.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnet.state_dict().keys())

        self.resnet.load_state_dict(all_params)
        print('[INFO] initialize weights from resnet50')
