import torch
import torch.nn as nn
import torchvision.models as models
from .SearchAttention import SA
#from .CAandSA import CA
from .AIC import AIC
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


class conv_2nV1(nn.Module):
    def __init__(self,in_hc=64,in_lc=256,out_c=64,main=0):
        super(conv_2nV1,self).__init__()
        self.main = main
        mid_c = min(in_hc,in_lc)
        self.relu = nn.ReLU(True)
        self.h2l_pool = nn.AvgPool2d((2,2),stride=2)
        self.l2h_up = nn.Upsample(scale_factor=2,mode="nearest")

        #stage 0 
        self.h2h_0 = nn.Conv2d(in_hc,mid_c,3,1,1)
        self.l2l_0 = nn.Conv2d(in_lc,mid_c,3,1,1)
        self.bnh_0 = nn.BatchNorm2d(mid_c)
        self.bnl_0 = nn.BatchNorm2d(mid_c)

        #stage 1
        self.h2h_1 = nn.Conv2d(mid_c,mid_c,3,1,1)
        self.h2l_1 = nn.Conv2d(mid_c,mid_c,3,1,1)
        self.l2h_1 = nn.Conv2d(mid_c,mid_c,3,1,1)
        self.l2l_1 = nn.Conv2d(mid_c,mid_c,3,1,1)
        self.bnl_1 = nn.BatchNorm2d(mid_c)
        self.bnh_1 = nn.BatchNorm2d(mid_c)

        if self.main == 0:
            #stage 2
            self.h2h_2 = nn.Conv2d(mid_c,mid_c,3,1,1)
            self.l2h_2 = nn.Conv2d(mid_c,mid_c,3,1,1)
            self.bnh_2 = nn.BatchNorm2d(mid_c)

            #stage 3
            self.h2h_3 = nn.Conv2d(mid_c,out_c,3,1,1)
            self.bnh_3 = nn.BatchNorm2d(out_c)

            self.identity = nn.Conv2d(in_hc,out_c,1)

        elif self.main == 1:
            self.h2l_2 = nn.Conv2d(mid_c,mid_c,3,1,1)
            self.l2l_2 = nn.Conv2d(mid_c,mid_c,3,1,1)
            self.bnl_2 = nn.BatchNorm2d(mid_c)

            #stage 3
            self.l2l_3 = nn.Conv2d(mid_c,out_c,3,1,1)
            self.bnl_3 = nn.BatchNorm2d(out_c)

            self.identity = nn.Conv2d(in_lc,out_c,1)

        else:
            raise NotImplementedError

    def forward(self,in_h,in_l):
        #stage 0
        h = self.relu(self.bnh_0(self.h2h_0(in_h)))
        l = self.relu(self.bnl_0(self.l2l_0(in_l)))


        if self.main == 0:
            #stage 1
            h2h = self.h2h_1(h)
            #h2l = self.h2l_1(self.h2l_pool(h))
            h2l = self.h2l_1(h) 
            l2l = self.l2l_1(l)
            #l2h = self.l2h_1(self.l2h_up(l))
            l2h = self.l2h_1(l)
            h = self.relu(self.bnh_1(h2h + l2h))
            l = self.relu(self.bnl_1(l2l + h2l))
            #stage 2
            h2h = self.h2h_2(h)
            #l2h = self.l2h_2(self.l2h_up(l))
            l2h = self.l2h_2(l)
            h_fuse = self.relu(self.bnh_2(h2h + l2h))

            #stage 3
            out = self.relu(self.bnh_3(self.h2h_3(h_fuse)) + self.identity(in_h))
        elif self.main == 1:
            #stage 1
            #h2h = self.h2h_1(h)
            h2l = self.h2l_1(self.h2l_pool(h))
            h2h = self.h2h_1(h) 
            l2l = self.l2l_1(l)
            l2h = self.l2h_1(self.l2h_up(l))
            #l2h = self.l2h_1(l)
            h = self.relu(self.bnh_1(h2h + l2h))
            l = self.relu(self.bnl_1(l2l + h2l))
            #stage 2
            h2l = self.h2l_2(self.h2l_pool(h))
            l2l = self.l2l_2(l)
            l_fuse = self.relu(self.bnl_2(h2l + l2l))

            #stage 3
            out = self.relu(self.bnl_3(self.l2l_3(l_fuse)) + self.identity(in_l))
        else:
            raise NotImplementedError

        return out


class conv_3nV1(nn.Module):
    def __init__(self,in_hc=64,in_mc=256,in_lc=512,out_c=64):
        super(conv_3nV1,self).__init__()
        mid_c = min(in_hc,in_mc,in_lc)
        self.relu = nn.ReLU(True)
        self.downsample = nn.AvgPool2d((2,2),stride=2)
        self.upsample = nn.Upsample(scale_factor=2,mode="nearest")

        #stage 0 
        self.h2h_0 = nn.Conv2d(in_hc,mid_c,3,1,1)
        self.m2m_0 = nn.Conv2d(in_mc,mid_c,3,1,1)
        self.l2l_0 = nn.Conv2d(in_lc,mid_c,3,1,1)
        self.bnh_0 = nn.BatchNorm2d(mid_c)
        self.bnm_0 = nn.BatchNorm2d(mid_c)
        self.bnl_0 = nn.BatchNorm2d(mid_c)

        #stage 1
        self.h2h_1 = nn.Conv2d(mid_c,mid_c,3,1,1)
        self.h2m_1 = nn.Conv2d(mid_c,mid_c,3,1,1)
        self.m2h_1 = nn.Conv2d(mid_c,mid_c,3,1,1)
        self.m2m_1 = nn.Conv2d(mid_c,mid_c,3,1,1)
        self.m2l_1 = nn.Conv2d(mid_c,mid_c,3,1,1)
        self.l2m_1 = nn.Conv2d(mid_c,mid_c,3,1,1)
        self.l2l_1 = nn.Conv2d(mid_c,mid_c,3,1,1)
        self.bnh_1 = nn.BatchNorm2d(mid_c)
        self.bnm_1 = nn.BatchNorm2d(mid_c)
        self.bnl_1 = nn.BatchNorm2d(mid_c)

        
        #stage 2
        self.h2m_2 = nn.Conv2d(mid_c,mid_c,3,1,1)
        self.l2m_2 = nn.Conv2d(mid_c,mid_c,3,1,1)
        self.m2m_2 = nn.Conv2d(mid_c,mid_c,3,1,1)
        self.bnm_2 = nn.BatchNorm2d(mid_c)

        #stage 3
        self.m2m_3 = nn.Conv2d(mid_c,out_c,3,1,1)
        self.bnm_3 = nn.BatchNorm2d(out_c)

        self.identity = nn.Conv2d(in_mc,out_c,1)

    def forward(self,in_h,in_m,in_l):
        #stage 0
        h = self.relu(self.bnh_0(self.h2h_0(in_h)))
        m = self.relu(self.bnm_0(self.m2m_0(in_m)))
        l = self.relu(self.bnl_0(self.l2l_0(in_l)))

        #stage 1
        h2h = self.h2h_1(h)
        m2h = self.m2h_1(self.upsample(m))

        h2m = self.h2m_1(self.downsample(h))
        m2m = self.m2m_1(m)
        l2m = self.l2m_1(self.upsample(l))

        m2l = self.m2l_1(self.downsample(m))
        l2l = self.l2l_1(l)

        h = self.relu(self.bnh_1(h2h + m2h))
        m = self.relu(self.bnm_1(h2m + m2m + l2m))
        l = self.relu(self.bnl_1(m2l + l2l))

        #stage 2
        h2m = self.h2m_2(self.downsample(h))
        m2m = self.m2m_2(m)
        l2m = self.l2m_2(self.upsample(l))
        m = self.relu(self.bnm_2(h2m + m2m + l2m))

        #stage 3
        out = self.relu(self.bnm_3(self.m2m_3(m)) + self.identity(in_m))
       
        return out


class conv_3nV2(nn.Module):
    def __init__(self,in_hc=64,in_mc=256,in_lc=512,out_c=64):
        super(conv_3nV2,self).__init__()
        mid_c = min(in_hc,in_mc,in_lc)
        self.relu = nn.ReLU(True)
        self.downsample = nn.AvgPool2d((2,2),stride=2)
        self.upsample = nn.Upsample(scale_factor=2,mode="nearest")

        #stage 0 
        self.h2h_0 = nn.Conv2d(in_hc,mid_c,3,1,1)
        self.m2m_0 = nn.Conv2d(in_mc,mid_c,3,1,1)
        self.l2l_0 = nn.Conv2d(in_lc,mid_c,3,1,1)
        self.bnh_0 = nn.BatchNorm2d(mid_c)
        self.bnm_0 = nn.BatchNorm2d(mid_c)
        self.bnl_0 = nn.BatchNorm2d(mid_c)

        #stage 1
        self.h2h_1 = nn.Conv2d(mid_c,mid_c,3,1,1)
        self.h2m_1 = nn.Conv2d(mid_c,mid_c,3,1,1)
        self.m2h_1 = nn.Conv2d(mid_c,mid_c,3,1,1)
        self.m2m_1 = nn.Conv2d(mid_c,mid_c,3,1,1)
        self.m2l_1 = nn.Conv2d(mid_c,mid_c,3,1,1)
        self.l2m_1 = nn.Conv2d(mid_c,mid_c,3,1,1)
        self.l2l_1 = nn.Conv2d(mid_c,mid_c,3,1,1)
        self.bnh_1 = nn.BatchNorm2d(mid_c)
        self.bnm_1 = nn.BatchNorm2d(mid_c)
        self.bnl_1 = nn.BatchNorm2d(mid_c)

        
        #stage 2
        self.h2m_2 = nn.Conv2d(mid_c,mid_c,3,1,1)
        self.l2m_2 = nn.Conv2d(mid_c,mid_c,3,1,1)
        self.m2m_2 = nn.Conv2d(mid_c,mid_c,3,1,1)
        self.bnm_2 = nn.BatchNorm2d(mid_c)

        #stage 3
        self.m2m_3 = nn.Conv2d(mid_c,out_c,3,1,1)
        self.bnm_3 = nn.BatchNorm2d(out_c)

        self.identity = nn.Conv2d(in_mc,out_c,1)

    def forward(self,in_h,in_m,in_l):
        #stage 0
        h = self.relu(self.bnh_0(self.h2h_0(in_h)))
        m = self.relu(self.bnm_0(self.m2m_0(in_m)))
        l = self.relu(self.bnl_0(self.l2l_0(in_l)))

        #stage 1
        h2h = self.h2h_1(h)
        m2h = self.m2h_1(m)

        h2m = self.h2m_1(h)
        m2m = self.m2m_1(m)
        l2m = self.l2m_1(self.upsample(l))

        m2l = self.m2l_1(self.downsample(m))
        l2l = self.l2l_1(l)

        h = self.relu(self.bnh_1(h2h + m2h))
        m = self.relu(self.bnm_1(h2m + m2m + l2m))
        l = self.relu(self.bnl_1(m2l + l2l))

        #stage 2
        h2m = self.h2m_2(h)
        m2m = self.m2m_2(m)
        l2m = self.l2m_2(self.upsample(l))
        m = self.relu(self.bnm_2(h2m + m2m + l2m))

        #stage 3
        out = self.relu(self.bnm_3(self.m2m_3(m)) + self.identity(in_m))
       
        return out



class AIM(nn.Module):
    def __init__(self,iC_list,oC_list):
        super(AIM,self).__init__()
        ic0,ic1,ic2,ic3,ic4 = iC_list
        oc0,oc1,oc2,oc3,oc4 = oC_list
        self.conv0 = conv_2nV1(in_hc=ic0,in_lc=ic1,out_c=oc0,main=0)
        self.conv1 = conv_3nV2(in_hc=ic0,in_mc=ic1,in_lc=ic2,out_c=oc1)
        self.conv2 = conv_3nV1(in_hc=ic1,in_mc=ic2,in_lc=ic3,out_c=oc2)
        self.conv3 = conv_3nV1(in_hc=ic2,in_mc=ic3,in_lc=ic4,out_c=oc3)
        self.conv4 = conv_2nV1(in_hc=ic3,in_lc=ic4,out_c=oc4,main=1)

    def forward(self,*xs):
        out_xs = []
        out_xs.append(self.conv0(xs[0],xs[1]))
        out_xs.append(self.conv1(xs[0],xs[1],xs[2]))
        out_xs.append(self.conv2(xs[1],xs[2],xs[3]))
        out_xs.append(self.conv3(xs[2],xs[3],xs[4]))
        out_xs.append(self.conv4(xs[3],xs[4]))

        return out_xs





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
        '''self.conv_inception1 = BasicConv2d(4*channel,channel,1,padding=0)
        self.conv_inception2_1 = BasicConv2d(4*channel,2*channel,1,padding=0)
        self.conv_inception2_2 = BasicConv2d(2*channel,channel,3,padding=1)
        self.conv_inception3_1 = BasicConv2d(4*channel,channel,1,padding=1)
        self.conv_inception3_2 = BasicConv2d(channel,channel,5,padding=1)
        self.conv_inception4_1 = nn.MaxPool2d(3,stride=1,padding=1)
        self.conv_inception4_2 = BasicConv2d(4*channel,channel,1,padding=0)'''
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
        '''x5_1 = self.conv_inception1(x3_2)
        x5_2 = self.conv_inception2_2(self.conv_inception2_1(x3_2))
        x5_3 = self.conv_inception3_2(self.conv_inception3_1(x3_2))
        x5_4 = self.conv_inception4_2(self.conv_inception4_1(x3_2))'''
        '''print(x5_1.shape)
        print(x5_2.shape)
        print(x5_3.shape)
        print(x5_4.shape)'''
        #x = torch.cat((x5_1,x5_2,x5_3,x5_4),1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
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


class SINet_ResNet50(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_ResNet50, self).__init__()

        self.resnet = ResNet_2Branch()
        self.downSample = nn.MaxPool2d(2, stride=2)
        
        self.trans = AIM(iC_list=(64,256,512,1024,2048),oC_list=(64,64,64,64,64)) 

        self.rf_low_sm = RF(128, channel) 
        self.rf2_sm = RF(640, channel)
        self.rf3_sm = RF(128, channel)
        self.rf4_sm = RF(64, channel)
        self.pdc_sm = PDC_SM(channel)

        self.rf2_im = RF(512, channel)
        self.rf3_im = RF(1024, channel)
        self.rf4_im = RF(2048, channel)
        self.pdc_im = PDC_IM(channel)

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

        #---------AIMS----------------------------
        x0,x1,x2,x3_sm,x4_sm = self.trans(
            x0,x1,x2,x3_sm,x4_sm
        )
        x01 = torch.cat((x0, x1), dim=1)        # (BS, 64+256, 88, 88)
        x01_down = self.downSample(x01)         # (BS, 320, 44, 44)
        x01_sm_rf = self.rf_low_sm(x01_down)    # (BS, 32, 44, 44)

        x2_sm_cat = torch.cat((x2_sm,
                               self.upsample_2(x3_sm),
                               self.upsample_2(self.upsample_2(x4_sm))), dim=1)   # 3584 channels
        x3_sm_cat = torch.cat((x3_sm,
                               self.upsample_2(x4_sm)), dim=1)                    # 3072 channels

        x2_sm_rf = self.rf2_sm(x2_sm_cat)
        x3_sm_rf = self.rf3_sm(x3_sm_cat)
        x4_sm_rf = self.rf4_sm(x4_sm)
        camouflage_map_sm = self.pdc_sm(x4_sm_rf, x3_sm_rf, x2_sm_rf, x01_sm_rf)
        


        # ---- Switcher: Search Attention (SA) ----
        #print(camouflage_map_sm.sigmoid().shape)
        x2_sa = self.SA(camouflage_map_sm.sigmoid(), x2_init)    # (512, 44, 44)

        # ---- Stage-2: Identification Module (IM) ----
        x3_im = self.resnet.layer3_2(x2_sa)                 # (1024, 22, 22)
        #x3_im = torch.mul(self.downSample(camouflage_map_sm.sigmoid()),x3_im)
        x4_im = self.resnet.layer4_2(x3_im)                 # (2048, 11, 11)
        #x4_im = torch.mul(self.downSample(self.downSample(camouflage_map_sm.sigmoid())),x4_im)

        x2_im_rf = self.rf2_im(x2_sa)
        x3_im_rf = self.rf3_im(x3_im)
        x4_im_rf = self.rf4_im(x4_im)
        # - decoder
        camouflage_map_im = self.pdc_im(x4_im_rf, x3_im_rf, x2_im_rf)
       

        # ---- output ----
        return self.upsample_8(camouflage_map_sm), self.upsample_8(camouflage_map_im)

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