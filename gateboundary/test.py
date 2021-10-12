import imageio
import torch
import torch.nn.functional as F
import numpy as np
import os
import cv2 as cv
import argparse
from scipy import misc  # NOTES: pip install scipy == 1.2.2
from boundarynet import Boundarydetection
from SINetnew import SINet_ResNet50
from dataloader1 import test_dataset
from boundarytrainer import eval_mae, numpy2tensor
import torchvision.transforms as T


parser = argparse.ArgumentParser() 
parser.add_argument('--testsize', type=int, default=352, help='the snapshot input size')
parser.add_argument('--model_Boundary_path', type=str,
                default='Snapshot/gate-boundary/gate-BoundaryNet_500.pth')
parser.add_argument('--model_sinet_path', type=str,
                default='SINet_40.pth')
parser.add_argument('--test_save_Boundary', type=str,
                    default='./result/withca/boundary/')
parser.add_argument('--test_save_sinet', type=str, 
                    default='./result/withca/sinet/')      
parser.add_argument('--test_save', type=str,
                    default='./result/withca/Union/')      
opt = parser.parse_args()

model1 = Boundarydetection().cuda()
model2 = SINet_ResNet50().cuda()
model1.load_state_dict(torch.load(opt.model_Boundary_path))
model2.load_state_dict(torch.load(opt.model_sinet_path))
model1.eval()
model2.eval()


for dataset in ['CHAMELEON']:
    save_path_Boundary = opt.test_save_Boundary + dataset + '/'
    save_path_sinet = opt.test_save_sinet + dataset + '/'
    save_path_union = opt.test_save + dataset + '/'
    os.makedirs(save_path_Boundary, exist_ok=True)
    os.makedirs(save_path_sinet, exist_ok=True)
    os.makedirs(save_path_union, exist_ok=True)
 
    test_loader = test_dataset('TestDataset/{}/Image/'.format(dataset),
                               'TestDataset/{}/Edge/'.format(dataset),
                               'TestDataset/{}/GT/'.format(dataset),
                               opt.testsize)
    img_count = 1
    
    MAE = 0
    MAE2 = 0
    MAE3 = 0
    for iteration in range(test_loader.size):
        image, edge, gt, name = test_loader.load_data()

        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        edge = np.asarray(edge, np.float32)
        edge /= (edge.max() + 1e-8)
        image = image.cuda()

        cam2 = model1(image)
        #print(cam2.shape)
        _, gt_result = model2(image)
        #print("gt", gt_result)
        #norm_edge = data_normal(cam2)
        #print("edge", norm_edge)
        #procam2 =  torch.mul(-1, torch.nn.MaxPool2d(3, 1, 1)(torch.mul(-1, cam2)))
        #cam2 = procam2
        #result = torch.mul(cam2, gt_result)
        result = torch.mul(cam2, gt_result)
        #result = torch.mul(-1, torch.nn.MaxPool2d(3, 1, 1)(torch.mul(-1, result)))
        
        
        #print(gt.shape)
        cam2 = F.upsample(cam2, size=edge.shape, mode='bilinear', align_corners=True)
        cam = F.upsample(gt_result, size=gt.shape, mode='bilinear', align_corners=True)
        result = F.upsample(result, size=gt.shape, mode='bilinear', align_corners=True)

        cam2 = cam2.sigmoid().data.cpu().numpy().squeeze()
        cam = cam.sigmoid().data.cpu().numpy().squeeze()
        result = result.sigmoid().data.cpu().numpy().squeeze()
        # normalize
        cam2 = (cam2 - cam2.min()) / (cam2.max() - cam2.min() + 1e-8)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        result =1 - (result - result.min()) / (result.max() - result.min() + 1e-8)

        imageio.imsave(save_path_sinet+name, cam)
        imageio.imsave(save_path_union+name, result)
        imageio.imsave(save_path_Boundary+name, cam2)
    
   

        mae_edge = eval_mae(numpy2tensor(cam2), numpy2tensor(edge))
        mae_gt = eval_mae(numpy2tensor(cam), numpy2tensor(gt))
        mae_result = eval_mae(numpy2tensor(result), numpy2tensor(gt))

        # coarse score
        data = open("log.txt","a")
        print('[Eval-Test] Dataset: {}, Image: {} ({}/{}), MAE_edge: {}'.format(dataset, name, img_count,
                                                                           test_loader.size, mae_edge),file=data)#, mae3, MAE3: {}
        data.close()
        MAE = MAE + mae_edge
        MAE2 = MAE2 + mae_gt
        MAE3 = MAE3 + mae_result
        #MAE3 = MAE3 +mae3

        img_count += 1

avg_MAE_edge = MAE/test_loader.size
avg_MAE_gt = MAE2/test_loader.size
avg_MAE_result = MAE3/test_loader.size
print("avg_MAE_edge: ",avg_MAE_edge)
print("avg_MAE_gt: ",avg_MAE_gt)
print("avg_MAE_result: ",avg_MAE_result)
print("\n[Congratulations! Testing Done]")


