import imageio
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import os
import argparse
from scipy import misc  # NOTES: pip install scipy == 1.2.2
from Cannynet import Cannydetection
from dataloadcanny import test_dataset
from cannytrainer import eval_mae, numpy2tensor
from PIL import Image
import cv2
import matplotlib.image as mpimg
import torchvision.transforms as T


parser = argparse.ArgumentParser() 
parser.add_argument('--testsize', type=int, default=352, help='the snapshot input size')
parser.add_argument('--model_Boundary_path', type=str,
                default='/home/kai/Desktop/xxq/SINet-master/Snapshot/boundarycanny/BoundaryNetFractional_100.pth')
parser.add_argument('--test_save_Boundary', type=str,
                    default='/home/kai/Desktop/xxq/SINet-master/canny/Result/')  
opt = parser.parse_args()

model1 = Cannydetection().cuda()
model1.load_state_dict(torch.load(opt.model_Boundary_path))
model1.eval()
#model2.eval()

for dataset in ['CHAMELEON']:
    save_path_Boundary = opt.test_save_Boundary + dataset + '/'
    os.makedirs(save_path_Boundary, exist_ok=True)
 
    test_loader = test_dataset('/home/kai/Desktop/xxq/SINet-master/Dataset/TestDataset/{}/Image/'.format(dataset),
                               '/home/kai/Desktop/xxq/SINet-master/Dataset/TestDataset/{}/Cannypic/'.format(dataset),
                               '/home/kai/Desktop/xxq/SINet-master/Dataset/TestDataset/{}/Edge/'.format(dataset),
                               opt.testsize)
    img_count = 1
    
    MAE = 0
    #MAE3 = 0
    for iteration in range(test_loader.size):
        image,frationaledge, edge,  name = test_loader.load_data()
        '''frationaledge = np.asarray(frationaledge, np.float32)
        frationaledge /= (frationaledge.max() + 1e-8)'''
        edge = np.asarray(edge, np.float32)
        edge /= (edge.max() + 1e-8)
        image = image.cuda()
        frationaledge = Variable(frationaledge).cuda()

        cam2 = model1(image, frationaledge)

        #print(gt.shape)
        cam2 = F.upsample(cam2, size=edge.shape, mode='bilinear', align_corners=True)
        cam2 = cam2.sigmoid().data.cpu().numpy().squeeze()

        # normalize
        cam2 = (cam2 - cam2.min()) / (cam2.max() - cam2.min() + 1e-8)

        imageio.imsave(save_path_Boundary+name, cam2)
        '''t = imageio.imread(save_path_Boundary+name)
        t = cv2.adaptiveThreshold(t, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        imageio.imsave(save_path_Boundary+name, t)'''

        mae = eval_mae(numpy2tensor(cam2), numpy2tensor(edge))

        # coarse score
        data = open("/home/kai/Desktop/xxq/SINet-master/log.txt","a")
        print('[Eval-Test] Dataset: {}, Image: {} ({}/{}), MAE: {}'.format(dataset, name, img_count,
                                                                           test_loader.size, mae),file=data)#, mae3, MAE3: {}
        data.close()
        MAE = MAE + mae
        #MAE3 = MAE3 +mae3

        img_count += 1

avg_MAE = MAE/test_loader.size
print("avg_MAE: ",avg_MAE)
print("\n[Congratulations! Testing Done]")


