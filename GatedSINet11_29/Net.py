from numpy.lib.arraypad import pad
import torch
import torch.nn as nn
from Src.backbone.ResNet import ResNet_2Branch
from torch.nn import functional as F
import numpy as np
from Src.tensor_ops import upsample_add, cus_sample

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, relu=False, bn=True):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
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


class RF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RF, self).__init__()
        self.relu = nn.ReLU(True)

        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )

        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), dim=1))

        x = self.relu(x_cat + self.conv_res(x))
        return x



class MSCA(nn.Module):
    def __init__(self, channels=32, r=4):
        super(MSCA, self).__init__()
        out_channels = int(channels // r)
        # local_att
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )

        # global_att
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )

        self.sig = nn.Sigmoid()

    def forward(self, x):

        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        wei = self.sig(xlg)

        return wei


class DGCM(nn.Module):
    def __init__(self, channel=32):
        super(DGCM, self).__init__()
        self.h2l_pool = nn.AvgPool2d((2, 2), stride=2)

        self.h2l = BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1, relu=True)
        self.h2h = BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1, relu=True)

        self.mscah = MSCA()
        self.mscal = MSCA()

        self.upsample_add = upsample_add
        self.conv = BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1, relu=True)

        self.downSample = nn.MaxPool2d(32, stride=32)

    def forward(self, x, gate):
        gate = self.downSample(gate)
        # first conv
        x_h = self.h2h(x)
        x_l = self.h2l(self.h2l_pool(x))
        x_h = x_h * self.mscah(x_h)
        x_l = x_l * self.mscal(x_l)
        out = self.upsample_add(x_l, x_h)
        out = self.conv(out)
        out = out*gate+out
        out = self.conv(out)
        return out


class FuseModel(nn.Module):
    def __init__(self, edge_channel, channel = 32):
        super(FuseModel, self).__init__()
        self.edge_conv = nn.Conv2d(edge_channel, edge_channel, 3, stride=2, padding=1)
        self.edge_conv1 = BasicConv2d(edge_channel,1,1,stride=1, padding=0,relu=True)
        self.high_upconv = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.high_conv1 = BasicConv2d(channel,channel,3,stride=1, padding=1,relu=True)
        self.low_conv = BasicConv2d(channel,channel,3,stride=1,padding=1,relu=True)
        self.conv1 = nn.Conv2d(channel*3, channel*3, 3, stride=1, padding=1)
        self.conv2 = BasicConv2d(channel*3, channel, 1, stride=1, padding=0, relu=True)

    def forward(self, high, low, edge):
        H,W = low.size()[2:]
        if H==22:
            edge_feature = self.edge_conv1(self.edge_conv(self.edge_conv(edge)))
        elif H==44:
            edge_feature = self.edge_conv1(self.edge_conv(edge))
        else:
            edge_feature = self.edge_conv1(edge)
        low_feature = self.low_conv(low)
        high_feature =  self.high_conv1(self.high_upconv(high))
        #print(low_feature.shape, high_feature.shape, edge_feature.shape)
        out1 = low_feature*high_feature
        out2 = low_feature*edge_feature
        out = self.conv2(self.conv1(torch.cat((out1, out2, low_feature), 1)))
        return out

from torchvision.models.resnet import BasicBlock, resnet50

class GatedConv(nn.Conv2d):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels, 1, bias=False)
        self.attention = nn.Sequential(
            nn.BatchNorm2d(in_channels + 1),
            nn.Conv2d(in_channels + 1, in_channels + 1, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels + 1, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, feat, gate):
        attention = self.attention(torch.cat((feat, gate), dim=1))
        out = F.conv2d(feat * (attention + 1), self.weight)
        return out



class Boundarydetection(nn.Module):
    def __init__(self):
        super(Boundarydetection, self).__init__()
        self.res2_conv = nn.Conv2d(512, 1, 1)
        self.res3_conv = nn.Conv2d(1024, 1, 1)
        self.res4_conv = nn.Conv2d(2048, 1, 1)
        self.res1 = BasicBlock(64, 64, 1)
        self.res2 = BasicBlock(32, 32, 1)
        self.res3 = BasicBlock(16, 16, 1)
        self.res1_pre = nn.Conv2d(64, 32, 1)
        self.res2_pre = nn.Conv2d(32, 16, 1)
        self.res3_pre = nn.Conv2d(16, 8, 1)
        self.gate1 = GatedConv(32, 32)
        self.gate2 = GatedConv(16, 16)
        self.gate3 = GatedConv(8, 8)
        self.gate = nn.Conv2d(8, 1, 1, bias=False)
        self.fuse = nn.Conv2d(2, 1, 1, bias=False)

    def forward(self, x0,x2,x3,x4,sobel):
    #88*88*64    #44*44*512    #22*22*1024    #11*11*2048    #352*352*1
        res2 = F.interpolate(self.res2_conv(x2), scale_factor=2, mode='bilinear', align_corners=True)
        res3 = F.interpolate(self.res3_conv(x3), scale_factor=4, mode='bilinear', align_corners=True)
        res4 = F.interpolate(self.res4_conv(x4), scale_factor=8, mode='bilinear', align_corners=True)
        gate1 = self.gate1(self.res1_pre(self.res1(x0)), res2)
        
        gate2 = self.gate2(self.res2_pre(self.res2(gate1)), res3)
        gate3 = self.gate3(self.res3_pre(self.res3(gate2)), res4)
        gate4 = self.gate(gate3)   #1*88*88
        gateforedge = F.interpolate(gate4, scale_factor=4, mode='bilinear', align_corners=True)
        feat = torch.sigmoid(self.fuse(torch.cat((gateforedge, sobel), dim=1)))
        #print(x.shape, gate1.shape, gate2.shape, gate3.shape, gate4.shape, feat.shape)
        return gate1, gate2, gate3, gateforedge, feat

class SINet_ResNet50(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_ResNet50, self).__init__()

        self.resnet = ResNet_2Branch()
        self.spachannel = SpatialAttention()
        self.ca1 = ChannelAttention(512)
        self.ca2 = ChannelAttention(1024)
        self.ca3 = ChannelAttention(2048)

        self.rf01 = RF(320, channel) 
        self.rf2 = RF(3584, channel)
        self.rf3 = RF(3072, channel)
        self.rf4 = RF(2048, channel)

        self.dgcm = DGCM()
        self.fuse3 = FuseModel(8)
        self.fuse2 = FuseModel(8)
        self.fuse1 = FuseModel(8)

        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.soble_bn = nn.BatchNorm2d(3)
        self.soble_relu = nn.ReLU(inplace=True)
        self.edge = Boundarydetection()
        self.toone = nn.Conv2d(3, 1, 1, bias=False)

        self.conv_to1_res = BasicConv2d(32, 32, 3, stride=1, padding=1)
        self.conv_to2_res = BasicConv2d(32, 1, 1, stride=1, padding=0)
        self.conv = BasicConv2d(4,1,1)
        self.relu = nn.Sigmoid()

        if self.training:
            self.initialize_weights()
    


    def forward(self, x):
        #--------------------------------sobel----------------------------------------------------
        edge_x = x
        conv_op = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
        soble_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')/3
        soble_kernel = soble_kernel.reshape((1, 1, 3, 3))
        soble_kernel = np.repeat(soble_kernel, 3, axis=1)
        soble_kernel = np.repeat(soble_kernel, 3, axis=0)
        conv_op.weight.data = torch.from_numpy(soble_kernel).cuda()
        edge_detect = self.soble_relu(self.soble_bn(conv_op(edge_x)))

        # ---- feature abstraction -----
        # - head
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        # - low-level features
        x0 = self.resnet.maxpool(x)    # (BS, 64, 88, 88)
        x1 = self.resnet.layer1(x0)     # (BS, 256, 88, 88)
        x2 = self.resnet.layer2(x1)     # (BS, 512, 44, 44)
        x3 = self.resnet.layer3_1(x2)   # (1024, 22, 22)
        x4 = self.resnet.layer4_1(x3)   # (2048, 11, 11)
        x01 = torch.cat((x0, x1), dim=1)        # (BS, 64+256, 88, 88)
        x01_sa = self.spachannel(x01)*x01          # (BS, 320, 88, 88)
        x2_ca = self.ca1(x2)
        x3_ca = self.ca2(x3)
        x4_ca = self.ca3(x4)

        x2_cat = torch.cat((x2_ca * x2,
                               self.upsample_2(x3_ca * x3),
                               self.upsample_2(self.upsample_2(x4_ca * x4))), dim=1)   # 3584 channels
        x3_cat = torch.cat((x3_ca * x3,
                               self.upsample_2(x4_ca * x4)), dim=1)                    # 3072 channels
        x01_rf = self.rf01(x01_sa)    # (BS, 32, 44, 44)
        x2_rf = self.rf2(x2_cat)      #32*44*44
        x3_rf = self.rf3(x3_cat)      #32*22*22
        x4_rf = self.rf4(x4_ca * x4)  #32*11*11
        edge_detect = self.toone(edge_detect)
        
        edge2, edge3, edge4, edgeres, edgewithsobel = self.edge(x0, x2, x3, x4, edge_detect)

        x4_res = self.dgcm(x4_rf, edgewithsobel)                     #32*11*11
        x3_res = self.fuse3(x4_res, x3_rf, edge4)     #32*22*22
        x2_res = self.fuse2(x3_res, x2_rf, edge4)     #32*44*44
        x1_res = self.fuse1(x2_res, x01_rf, edge4)    #32*88*88
        x4_res = F.interpolate( self.conv_to2_res(self.conv_to1_res((x4_res))), (352, 352), mode="bilinear", align_corners=False)
        x3_res = F.interpolate( self.conv_to2_res(self.conv_to1_res((x3_res))), (352, 352), mode="bilinear", align_corners=False)   #32*22*22
        x2_res = F.interpolate( self.conv_to2_res(self.conv_to1_res((x2_res))), (352, 352), mode="bilinear", align_corners=False)   #32*44*44
        x1_res = F.interpolate( self.conv_to2_res(self.conv_to1_res((x1_res))), (352, 352), mode="bilinear", align_corners=False)

        net_res = self.conv(torch.cat((x1_res, x2_res, x3_res, x4_res), 1))



        
        # ---- output ----
        return x4_res, x3_res, x2_res, x1_res, edgeres, net_res

    def initialize_weights(self):
        model= resnet50(pretrained = True)
        pretrained_dict = model.state_dict()
        #pretrained_dict = model.load_state_dict(model_zoo.load_url(model_urls['res2next50']))
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