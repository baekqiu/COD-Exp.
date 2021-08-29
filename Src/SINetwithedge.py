import torch
import torch.nn as nn
import torchvision.models as models
from .SearchAttention import SA
from Src.backbone.ResNet import ResNet_2Branch
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

        self.downsample = BasicConv2d(1, 1, 3, stride=2, padding=1)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(4*channel, 4*channel, 3, padding=1)
        self.conv4 = BasicConv2d(4*channel, 4*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(4*channel, 1, 1)

    def forward(self, x1, x2, x3, x4, edge):
        # print x1.shape, x2.shape, x3.shape, x4.shape
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) * self.conv_upsample3(self.upsample(x2)) * x3
        x3_1 = torch.add(x3_1, self.downsample(edge))

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)
        x2_2 = torch.add(x2_2, self.downsample(self.downsample(edge)))

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2)), x4), 1)
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
        x = self.conv5(x)

        return x

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
        self.branch1_conv2 = BasicConv2d(3, 8, 1)
        self.conv7_1 = BasicConv2d(in_channel, 128, kernel_size=(7,1), padding=(3,0))
        self.branch2_conv = BasicConv2d(128, 1, 1)
        self.branch2_conv2 = BasicConv2d(3, 8, 1)

    def forward(self, image):
        branch1_x_1 = self.conv1_7(image)
        branch1_x_2 = self.resblock1(branch1_x_1)
        branch1_x_3 = self.resblock2(branch1_x_2)
        branch1_x_4 = self.resblock3(branch1_x_3)
        branch1_x_5 = self.branch1_conv(branch1_x_4)
        #print("branch1.canny.shape", image_canny.shape)
        #print("branch1.conv.shape", branch1_x_5.shape)
        #print(image.shape)
        #-------soble算子------------------------------
        #edge_x = torch.transpose(image, 3, 1)
        #edge_x = torch.transpose(edge_x, 2, 3)
        edge_x = image
        conv_op = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
        soble_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')/3
        soble_kernel = soble_kernel.reshape((1, 1, 3, 3))
        soble_kernel = np.repeat(soble_kernel, 3, axis=1)
        soble_kernel = np.repeat(soble_kernel, 3, axis=0)
        conv_op.weight.data = torch.from_numpy(soble_kernel).cuda()
        edge_detect = conv_op(edge_x)
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


'''class AIC(nn.Module):
	"""docstring for AIC"""
	def __init__(self,input,index=1):
		super(AIC, self).__init__()
		self.hidden=input
		self.weights = nn.Conv2d(self.hidden, 6, kernel_size=1)
		self.softmax = torch.softmax

		self.conv1 = nn.Conv2d(input, self.hidden, kernel_size=1, padding=0)
		self.conv1_bn=nn.BatchNorm2d(self.hidden, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.conv1_relu = nn.ReLU(inplace=True)

		self.x1 = nn.Conv2d(self.hidden, self.hidden, kernel_size=[1,3], padding=[0,1])
		self.x1_bn = nn.BatchNorm2d(self.hidden, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.x1_relu = nn.ReLU(inplace=True)

		self.x2 = nn.Conv2d(self.hidden, self.hidden, kernel_size=[1,5], padding=[0,2])
		self.x2_bn = nn.BatchNorm2d(self.hidden, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.x2_relu = nn.ReLU(inplace=True)

		self.x3 = nn.Conv2d(self.hidden, self.hidden, kernel_size=[1,7], padding=[0,3])
		self.x3_bn = nn.BatchNorm2d(self.hidden, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.x3_relu = nn.ReLU(inplace=True)


		self.x_relu = nn.ReLU(inplace=True)
		self.x_bn = nn.InstanceNorm2d(self.hidden)

		self.y1 = nn.Conv2d(self.hidden, self.hidden, kernel_size=[3,1], padding=[1,0])
		self.y1_bn = nn.BatchNorm2d(self.hidden, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.y1_relu = nn.ReLU(inplace=True)

		self.y2 = nn.Conv2d(self.hidden, self.hidden, kernel_size=[5,1], padding=[2,0])
		self.y2_bn = nn.BatchNorm2d(self.hidden, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.y2_relu = nn.ReLU(inplace=True)

		self.y3 = nn.Conv2d(self.hidden, self.hidden, kernel_size=[7,1], padding=[3,0])
		self.y3_bn = nn.BatchNorm2d(self.hidden, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.y3_relu = nn.ReLU(inplace=True)


		self.y_relu = nn.ReLU(inplace=True)
		self.y_bn = nn.InstanceNorm2d(self.hidden)

		self.conv2 = nn.Conv2d(self.hidden, input, kernel_size=1, padding=0)
		self.conv2_bn=nn.BatchNorm2d(self.hidden, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.conv2_relu = nn.ReLU(inplace=True)

		self.bn2 = nn.BatchNorm2d(input, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.relu2 = nn.ReLU(inplace=True)

		self.conv3 = nn.Conv2d(input, input, kernel_size=1, padding=0)
		self.bn3 = nn.BatchNorm2d(input, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		self.relu3 = nn.ReLU(inplace=True)
	def forward(self, x):
		residual = x
		x_input = self.conv1(x)
		x_input=self.conv1_bn(x_input)
		x_input=self.conv1_relu(x_input)

		y_input=x_input#记录一下卷积后的特征作为y的输入

		w = self.weights(x_input)#生成权重
		wx = self.softmax(w[:,:3],1)
		wy = self.softmax(w[:,3:],1)


		x1 = self.x1(x_input)
		x1=self.x1_bn(x1)
		x1=self.x1_relu(x1)

		x2 = self.x2(x_input)
		x2 =self.x2_bn(x2)
		x2=self.x2_relu(x2)

		x3 = self.x3(x_input)
		x3=self.x3_bn(x3)
		x3=self.x3_relu(x3)

		x_input = x1.transpose(0,1).mul(wx[:,0])+x2.transpose(0,1).mul(wx[:,1])+x3.transpose(0,1).mul(wx[:,2])
		x_input = x_input.transpose(0,1)
		x_input = self.x_bn(x_input)
		x_input = self.x_relu(x_input)
         
		y1 = self.y1(y_input)
		y1 = self.y1_bn(y1)
		y1 = self.y1_relu(y1)

		y2 = self.y2(y_input)
		y2 = self.y2_bn(y2)
		y2 = self.y2_relu(y2)

		y3 = self.y3(y_input)
		y3 = self.y3_bn(y3)
		y3 = self.y3_relu(y3)

		y_input = y1.transpose(0,1).mul(wy[:,0])+y2.transpose(0,1).mul(wy[:,1])+y3.transpose(0,1).mul(wy[:,2])
		y_input = y_input.transpose(0,1)
		y_input = self.y_bn(y_input)
		y_input = self.y_relu(y_input)

		x=x_input+y_input
		#x=self.conv(x)
		#x=self.bn(x)

		x = self.conv2(x)
		x=self.conv2_bn(x)
		x=self.conv2_relu(x)

		x =x+residual
		x = self.bn2(x)
		#x = self.relu_bn(x)
		x=self.relu2(x)

		x=self.conv3(x)
		x=self.bn3(x)
		x=self.relu3(x)
		return x'''

        
        

class SINet_ResNet50(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_ResNet50, self).__init__()

        self.resnet = ResNet_2Branch()
        
        self.BEM = BEM(3)
        self.boundary_conv1 = BasicConv2d(144, 144, 3, stride=2, padding=1)
        self.boundary_conv2 = BasicConv2d(144, 144, 3, stride=2, padding=1)
        #self.boundary_conv3 = BasicConv2d(144, 144, 3, stride=1, padding=1)
        self.boundary_conv = BasicConv2d(144, 1, 1)
        self.boundary_sigmoid = nn.Sigmoid()

        self.downSample = nn.MaxPool2d(2, stride=2)

        self.rf_low_sm = RF(320, channel)
        self.rf2_sm = RF(3584, channel)
        self.rf3_sm = RF(3072, channel)
        self.rf4_sm = RF(2048, channel)
        self.pdc_sm = PDC_SM(channel)

        self.rf2_im = RF(512, channel)
        self.rf3_im = RF(1024, channel)
        self.rf4_im = RF(2048, channel)
        self.pdc_im = PDC_IM(channel)

        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample_16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.upsample_32 = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)
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

        BEM_res = self.BEM(x)
        #edge_res = self.boundary_conv(self.boundary_conv2(self.boundary_conv1(BEM_res)))
        #print(edge_res.shape)
        # ---- Stage-1: Search Module (SM) ----
        x01 = torch.cat((x0, x1), dim=1)        # (BS, 64+256, 88, 88)
        x01_down = self.downSample(x01)         # (BS, 320, 44, 44)
        x01_sm_rf = self.rf_low_sm(x01_down)    # (BS, 32, 44, 44)

        x2_sm = x2                              # (512, 44, 44)
        x3_sm = self.resnet.layer3_1(x2_sm)     # (1024, 22, 22)
        x4_sm = self.resnet.layer4_1(x3_sm)     # (2048, 11, 11)

        x2_sm_cat = torch.cat((x2_sm,
                               self.upsample_2(x3_sm),
                               self.upsample_2(self.upsample_2(x4_sm))), dim=1)   # 3584 channels
        x3_sm_cat = torch.cat((x3_sm,
                               self.upsample_2(x4_sm)), dim=1)                    # 3072 channels

        x2_sm_rf = self.rf2_sm(x2_sm_cat)
        x3_sm_rf = self.rf3_sm(x3_sm_cat)
        x4_sm_rf = self.rf4_sm(x4_sm)

        BEM_res_1 = torch.cat((BEM_res, self.upsample_8(x01_sm_rf), self.upsample_8(x2_sm_rf), self.upsample_16(x3_sm_rf), self.upsample_32(x4_sm_rf)), dim=1)   #353*352*144
        edge_res = self.boundary_sigmoid(self.boundary_conv(self.boundary_conv2(self.boundary_conv1(BEM_res_1))))


        camouflage_map_sm = self.pdc_sm(x4_sm_rf, x3_sm_rf, x2_sm_rf, x01_sm_rf, edge_res)

        # ---- Switcher: Search Attention (SA) ----
        x2_sa = self.SA(camouflage_map_sm.sigmoid(), x2)    # (512, 44, 44)

        # ---- Stage-2: Identification Module (IM) ----
        x3_im = self.resnet.layer3_2(x2_sa)                 # (1024, 22, 22)
        x4_im = self.resnet.layer4_2(x3_im)                 # (2048, 11, 11)

        x2_im_rf = self.rf2_im(x2_sa)
        x3_im_rf = self.rf3_im(x3_im)
        x4_im_rf = self.rf4_im(x4_im)
        # - decoder
        camouflage_map_im = self.pdc_im(x4_im_rf, x3_im_rf, x2_im_rf)

        # ---- output ----
        return self.upsample_8(camouflage_map_sm), self.upsample_8(camouflage_map_im), BEM_res_1

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
