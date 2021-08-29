import torch
import torch.nn as nn
import torchvision.models as models
from ResNet import ResNet_2Branch
from torchvision import transforms
from PIL import Image 
import cv2
import numpy as np

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
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

    def forward(self, x1, x2, x3, x4, x5, channel = 258):
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
        '''f2 = torch.cat((x1_3, x2_3, x3_3, x4_3, x5_3), 1)
        print("f2.shape: ", f2.shape)'''

        #return f2
        return x1_3, x2_3, x3_3, x4_3, x5_3

class Resnetbranch(nn.Module):
    def __init__(self, in_channel):
        super(Resnetbranch, self).__init__()
        self.conv1 = BasicConv2d(in_channel, in_channel//4, 1, padding=0)
        self.conv2 = BasicConv2d(in_channel//4, in_channel//4, 3, padding=1)
        self.conv3 = nn.Conv2d(in_channel//4, in_channel, 1)
        self.conv4 = BasicConv2d(in_channel*2, in_channel, 1, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_1 = self.conv3(self.conv2(self.conv1(x)))
        x_2 = torch.cat((x_1, x), 1)
        x_3 = self.relu(x_2)
        x_3 = self.conv4(x_3)

        return x_3




class BEM(nn.Module):
    def __init__(self, in_channel):
        super(BEM, self).__init__()
        self.resblock1 = Resnetbranch(128)
        self.resblock2 = Resnetbranch(128)
        self.resblock3 = Resnetbranch(128)
        self.conv1_7 = BasicConv2d(in_channel, 128, kernel_size=(1,7), padding=(0,3))
        self.branch1_conv = BasicConv2d(128, 1, 1)
        self.branch1_conv2 = BasicConv2d(1, 8, 1)
        self.conv7_1 = BasicConv2d(in_channel, 128, kernel_size=(7,1), padding=(3,0))
        self.branch2_conv = BasicConv2d(128, 1, 1)
        self.branch2_conv2 = BasicConv2d(1, 8, 1)

    def forward(self, image, cannyimage):
        branch1_x_1 = self.conv1_7(image)
        branch1_x_2 = self.resblock1(branch1_x_1)
        branch1_x_3 = self.resblock2(branch1_x_2)
        branch1_x_4 = self.resblock3(branch1_x_3)
        branch1_x_5 = self.branch1_conv(branch1_x_4)
        #print("branch1.canny.shape", image_canny.shape)
        #print("branch1.conv.shape", branch1_x_5.shape)
        #print(image.shape)
        
        edge_detect =cannyimage #(1*352*352)
        #print("edge_detect.shape", edge_detect.shape)

        branch1_x = self.branch1_conv2(torch.add(edge_detect, branch1_x_5))
        branch2_x_1 = self.conv7_1(image)
        branch2_x_2 = self.resblock1(branch2_x_1)
        branch2_x_3 = self.resblock2(branch2_x_2)
        branch2_x_4 = self.resblock3(branch2_x_3)
        branch2_x_5 = self.branch2_conv(branch2_x_4)
        branch2_x = self.branch2_conv2(torch.add(edge_detect, branch2_x_5))
        result = torch.cat((branch1_x, branch2_x), 1)
        #print("result.shape=", result.shape)

        return result


class Cannydetection(nn.Module):
    def __init__(self):
        super(Cannydetection, self).__init__()
        
        self.resnet = ResNet_2Branch()
        self.CFM = CFM(256)
        self.MA_unit_CA = ChannelAttention(256)
        self.MA_unit_SA = SpatialAttention()
        self.BEM = BEM(3)
        self.conv = BasicConv2d(16,16,3,stride=2, padding=1)
        self.conv1 = BasicConv2d(2576, 1, 1)  
        self.upsample_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        if self.training:
            self.initialize_weights()

    def forward(self, x, cannyx):
        #--------feature abstraction-------------------
        x1 = self.resnet.conv1(x)
        x1 = self.resnet.bn1(x1)
        x1 = self.resnet.relu(x1)

        x1 = self.resnet.maxpool(x1)
        x2 = self.resnet.layer1(x1)
        x3 = self.resnet.layer2(x2)
        x4 = self.resnet.layer3_1(x3)
        x5 = self.resnet.layer4_1(x4)

        res1_2, res2_2, res3_2, res4_2, res5_2 = self.CFM(x1, x2, x3, x4, x5)
        res1_3 = torch.cat((self.MA_unit_CA(res1_2)*res1_2, self.MA_unit_SA(res1_2)*res1_2), dim=1)
        res2_3 = torch.cat((self.MA_unit_CA(res2_2)*res2_2, self.MA_unit_SA(res2_2)*res2_2), dim=1)
        res3_3 = torch.cat((self.MA_unit_CA(res3_2)*res3_2, self.MA_unit_SA(res3_2)*res3_2), dim=1)
        res4_3 = torch.cat((self.MA_unit_CA(res4_2)*res4_2, self.MA_unit_SA(res4_2)*res4_2), dim=1)
        res5_3 = torch.cat((self.MA_unit_CA(res5_2)*res5_2, self.MA_unit_SA(res5_2)*res5_2), dim=1)
        Cross_layer_Attention_res = torch.cat((res1_3, res2_3, res3_3, res4_3, res5_3), dim=1)
        #print("Cross_layer_Attention_res.shape", Cross_layer_Attention_res.shape) #36*2560*88*88

        BEM_res = self.BEM(x, cannyx)

        Cross_layer_Attention_res = self.upsample_4(Cross_layer_Attention_res)
        res2 = torch.cat((Cross_layer_Attention_res, BEM_res), dim=1)
        res3 = self.conv1(res2)
        #print(res3.shape)

        return res3

    def initialize_weights(self):
        resnet50 = models.resnet50(pretrained=True)
        pretrained_dict = resnet50.state_dict()      
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
