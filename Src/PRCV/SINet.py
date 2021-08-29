import torch
import torch.nn as nn
import torchvision.models as models
from .SearchAttention import SA
#from .AIC import AIC
from Src.backbone.ResNet import ResNet_2Branch
from torchvision import transforms
from PIL import Image
import cv2
from Src.utils.Dataloader import test_dataset

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


class RF(nn.Module):
    # Revised from: Receptive Field Block Net for Accurate and Fast Object Detection, 2018, ECCV
    # GitHub: https://github.com/ruinmessi/RFBNet
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


class PDC_SM(nn.Module):
    # Partial Decoder Component (Search Module)
    def __init__(self, channel):
        super(PDC_SM, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(4*channel, 4*channel, 3, padding=1)
        self.conv4 = BasicConv2d(4*channel, 4*channel, 3, padding=1)
        self.conv_inception1 = BasicConv2d(4*channel,channel,1,padding=0)
        self.conv_inception2_1 = BasicConv2d(4*channel,2*channel,1,padding=0)
        self.conv_inception2_2 = BasicConv2d(2*channel,channel,3,padding=1)
        self.conv_inception3_1 = BasicConv2d(4*channel,channel,1,padding=1)
        self.conv_inception3_2 = BasicConv2d(channel,channel,5,padding=1)
        self.conv_inception4_1 = nn.MaxPool2d(3,stride=1,padding=1)
        self.conv_inception4_2 = BasicConv2d(4*channel,channel,1,padding=0)
        self.conv5 = nn.Conv2d(4*channel, 1, 1)

    def forward(self, x1, x2, x3, x4):
        # print x1.shape, x2.shape, x3.shape, x4.shape
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2)), x4), 1)
        #print(x3_2.shape)
        x5_1 = self.conv_inception1(x3_2)
        x5_2 = self.conv_inception2_2(self.conv_inception2_1(x3_2))
        x5_3 = self.conv_inception3_2(self.conv_inception3_1(x3_2))
        x5_4 = self.conv_inception4_2(self.conv_inception4_1(x3_2))
        '''print(x5_1.shape)
        print(x5_2.shape)
        print(x5_3.shape)
        print(x5_4.shape)'''
        x = torch.cat((x5_1,x5_2,x5_3,x5_4),1)
        '''x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)'''
        x = self.conv5(x)

        return x

class PDC_IM(nn.Module):
    # Partial Decoder Component (Identification Module)
    def __init__(self, channel):
        super(PDC_IM, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv4 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3*channel, 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)
        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)

        x3_2 = self.conv_concat3(x3_2)
        x = self.conv4(x3_2)
        #print(x.shape)
        x = self.conv5(x)

        return x

class PDC_EM(nn.Module):
    # Partial Decoder Component (Identification Module)
    def __init__(self, channel):
        super(PDC_EM, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv4 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3*channel, 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)
        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)

        x3_2 = self.conv_concat3(x3_2)
        x = self.conv4(x3_2)
        #print(x.shape)
        x = self.conv5(x)

        return x



class SINet_ResNet50(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_ResNet50, self).__init__()

        self.resnet = ResNet_2Branch()
        self.downSample = nn.MaxPool2d(2, stride=2)
        self.spachannel = SpatialAttention()
        self.spachannelx0 = SpatialAttention()
        self.spachannelx1 = SpatialAttention()
        self.ca1 = ChannelAttention(512)
        self.ca2 = ChannelAttention(1024)
        self.ca3 = ChannelAttention(2048)
        self.ca4 = ChannelAttention(2048)
        
        #self.trans = AIM(iC_list=(64,256,512,1024,2048),oC_list=(64,64,64,64,64)) 

        self.rf_low_sm = RF(320, channel) 
        self.rf2_sm = RF(3584, channel)
        self.rf3_sm = RF(3072, channel)
        self.rf4_sm = RF(2048, channel)
        self.pdc_sm = PDC_SM(channel)

        self.rf_low_im = RF(320,channel)
        self.rf2_im = RF(512, channel)
        self.rf3_im = RF(1024, channel)
        self.rf4_im = RF(2048, channel)
        self.pdc_im = PDC_IM(channel)
        self.pdc_em = PDC_EM(channel)

        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.SA = SA()


        if self.training:
            self.initialize_weights()
    


    def forward(self, x):
        # ---- feature abstraction -----
        # - head
        x0 = self.resnet.conv1(x)
        x0 = self.resnet.bn1(x0)
        x0 = self.resnet.relu(x0)
        # - low-level features
        x0 = self.resnet.maxpool(x0)    # (BS, 64, 88, 88)
        x1 = self.resnet.layer1(x0)     # (BS, 256, 88, 88)
        x2 = self.resnet.layer2(x1)     # (BS, 512, 44, 44)
        x2_init = x2


        # ---- Stage-1: Search Module (SM) ----

        x2_sm = x2                              # (512, 44, 44)
        x3_sm = self.resnet.layer3_1(x2_sm)     # (1024, 22, 22)
        x4_sm = self.resnet.layer4_1(x3_sm)     # (2048, 11, 11)


        x0_sm_sa = self.spachannelx0(x0)
        x1_sm_sa = self.spachannelx1(x1)
        x01 = torch.cat((x0, x1), dim=1)        # (BS, 64+256, 88, 88)
        x01_down = self.downSample(self.spachannel(x01) * x01)         # (BS, 320, 44, 44)
        #x01_down = self.downSample(x01)
        x01_sm_rf = self.rf_low_sm(x01_down)    # (BS, 32, 44, 44)

        x2_sm_ca = self.ca1(x2_sm)
        x3_sm_ca = self.ca2(x3_sm)
        x4_sm_ca = self.ca3(x4_sm)

        x2_sm_cat = torch.cat((x2_sm_ca * x2_sm,
                               self.upsample_2(x3_sm_ca * x3_sm),
                               self.upsample_2(self.upsample_2(x4_sm_ca * x4_sm))), dim=1)   # 3584 channels
        x3_sm_cat = torch.cat((x3_sm_ca * x3_sm,
                               self.upsample_2(x4_sm_ca * x4_sm)), dim=1)                    # 3072 channels
        '''x2_sm_cat = torch.cat((x2_sm,
                               self.upsample_2(x3_sm),
                               self.upsample_2(self.upsample_2(x4_sm))), dim=1)
        x3_sm_cat = torch.cat((x3_sm,
                               self.upsample_2(x4_sm)), dim=1)'''

        x2_sm_rf = self.rf2_sm(x2_sm_cat)
        x3_sm_rf = self.rf3_sm(x3_sm_cat)
        x4_sm_rf = self.rf4_sm(x4_sm_ca * x4_sm)
        #x4_sm_rf = self.rf4_sm(x4_sm)
        camouflage_map_sm = self.pdc_sm(x4_sm_rf, x3_sm_rf, x2_sm_rf, x01_sm_rf)


        # ---- Switcher: Search Attention (SA) ----
        #print(camouflage_map_sm.sigmoid().shape)
        x_low_sa = self.SA(camouflage_map_sm.sigmoid(), x01_down)
        x2_sa = self.SA(camouflage_map_sm.sigmoid(), x2_init)    # (512, 44, 44)

        # ---- Stage-2: Identification Module (IM) ----
        x3_im = self.resnet.layer3_2(x2_sa)                 # (1024, 22, 22)
        #x3_im = torch.mul(self.downSample(camouflage_map_sm.sigmoid()),x3_im)
        x4_im = self.resnet.layer4_2(x3_im)                 # (2048, 11, 11)
        #x4_im = torch.mul(self.downSample(self.downSample(camouflage_map_sm.sigmoid())),x4_im)

        x_low_im_rf = self.rf_low_im(x_low_sa)
        x2_im_rf = self.rf2_im(x2_sa)
        x3_im_rf = self.rf3_im(x3_im)
        x4_im_rf = self.rf4_im(x4_im)
        # - decoder
        camouflage_map_im = self.pdc_im(x4_im_rf, x3_im_rf, x2_im_rf)#, x_low_im_rf

        #/////////////////////////////////////////
        #camouflage_map_em = self.pdc_em(x4_im_rf, x3_im_rf, x2_im_rf)
        '''new_x2_sa = self.SA(camouflage_map_im.sigmoid(),x2_init)
        new_x3_im = self.resnet.layer3_2(new_x2_sa)
        new_x4_im = self.resnet.layer4_2(new_x3_im)
        new_x2_im_rf = self.rf2_im(new_x2_sa)
        new_x3_im_rf = self.rf3_im(new_x3_im)
        new_x4_im_rf = self.rf4_im(new_x4_im)
        camouflage_map_em = self.pdc_em(new_x4_im_rf, new_x3_im_rf, new_x2_im_rf)'''

        # ---- output ----
        return self.upsample_8(camouflage_map_sm), self.upsample_8(camouflage_map_im)#, self.upsample_8(camouflage_map_em)



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