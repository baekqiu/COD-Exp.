import imageio
import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
from scipy import misc  # NOTES: pip install scipy == 1.2.2
from Src.Boundarynet import Boundarydetection
#from Src.SINetnew import SINet_ResNet50
from Src.utils.dataloader1 import test_dataset
from Src.utils.boundarytrainer import eval_mae, numpy2tensor
from PIL import Image
import cv2
import matplotlib.image as mpimg
import torchvision.transforms as T


parser = argparse.ArgumentParser() 
parser.add_argument('--testsize', type=int, default=352, help='the snapshot input size')
parser.add_argument('--model_Boundary_path', type=str,
                default='/home/kai/Desktop/xxq/SINet-master/Snapshot/iccv-vspw/BoundaryNet_200.pth')
'''parser.add_argument('--model_sinet_path', type=str,
                default='./initmodels/SINet_40.pth')'''
parser.add_argument('--test_save_Boundary', type=str,
                    default='./Result/iccv-vspw/Boundary/')
'''parser.add_argument('--test_save_sinet', type=str, 
                    default='./Result/Boundaryextend/sinet/')      
parser.add_argument('--test_save', type=str,
                    default='./Result/Boundaryextend/Union/')      '''   
parser.add_argument('--data_dir', type=str, default='/home/kai/Desktop/vspw/VSPW_480p/data/')      
opt = parser.parse_args()

with open("/home/kai/Desktop/vspw/VSPW_480p/test.txt", "r") as f:
    filelist = f.readlines()

model1 = Boundarydetection().cuda()
#model2 = SINet_ResNet50().cuda()
model1.load_state_dict(torch.load(opt.model_Boundary_path))
#model2.load_state_dict(torch.load(opt.model_sinet_path))
model1.eval()
#model2.eval()

for dataset in filelist:
    #save_path_Boundary = opt.test_save_Boundary + dataset + '/'
    save_path_Boundary = os.path.join(opt.data_dir, dataset[:-1], 'Edge/')
    data_path = os.path.join(opt.data_dir, dataset[:-1], 'origin/')
    '''save_path_sinet = opt.test_save_sinet + dataset + '/'
    save_path = opt.test_save + dataset + '/' '''
    os.makedirs(save_path_Boundary, exist_ok=True)
    '''os.makedirs(save_path_sinet, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)'''
 
    test_loader = test_dataset(data_path,
                               opt.testsize)
    print(test_loader.size)
    img_count = 1
    
    MAE_result = 0
    #MAE3 = 0
    for iteration in range(test_loader.size):
        image, name = test_loader.load_data()

        '''gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        edge = np.asarray(edge, np.float32)
        edge /= (edge.max() + 1e-8)'''
        image = image.cuda()

        cam2 = model1(image)
        #gt_result = model2(image)
        #result = torch.mul(cam2, gt_result)
        
        
        
        #print(gt.shape)
        #cam2 = F.upsample(cam2, size=edge.shape, mode='bilinear', align_corners=True)
        cam2 = cam2.sigmoid().data.cpu().numpy().squeeze()

        #cam = F.upsample(gt_result, size=gt.shape, mode='bilinear', align_corners=True)
        #cam = cam.sigmoid().data.cpu().numpy().squeeze()
        
        '''result = F.upsample(result, size=gt.shape, mode='bilinear', align_corners=True)
        result = result.sigmoid().data.cpu().numpy().squeeze()'''
        # normalize
        cam2 = (cam2 - cam2.min()) / (cam2.max() - cam2.min() + 1e-8)

        #cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        #result =1 - (result - result.min()) / (result.max() - result.min() + 1e-8)

        #misc.imsave(save_path_sinet+name, cam)
        #misc.imsave(save_path_Boundary+name, cam2)
        #misc.imsave(save_path+name, result)
        imageio.imsave(save_path_Boundary+name, cam2)
    
   

        '''mae_edge = eval_mae(numpy2tensor(cam2), numpy2tensor(edge))
        mae_gt = eval_mae(numpy2tensor(cam), numpy2tensor(gt))
        mae_result = eval_mae(numpy2tensor(result), numpy2tensor(gt))'''

        '''# coarse score
        data = open("/home/kai/Desktop/xxq/SINet-master/log.txt","a")
        print('[Eval-Test] Dataset: {}, Image: {} ({}/{}), MAE_edge: {}, MAE_gt: {}, MAE_result: {}'.format(dataset, name, img_count,
                                                                           test_loader.size, mae_edge, mae_gt, mae_result),file=data)#, mae3, MAE3: {}
        data.close()
        MAE_edge = MAE_edge + mae_edge
        MAE_gt = MAE_gt + mae_gt
        MAE_result = MAE_result + mae_result
        #MAE3 = MAE3 +mae3

        img_count += 1

avg_MAE_edge = MAE_edge/test_loader.size
avg_MAE_gt = MAE_gt/test_loader.size
avg_MAE_result = MAE_result/test_loader.size
print("avg_MAE_edge: ",avg_MAE_edge)
print("avg_MAE_gt: ",avg_MAE_gt)
print("avg_MAE_result: ",avg_MAE_result)'''
print("\n[Congratulations! Testing Done]")


