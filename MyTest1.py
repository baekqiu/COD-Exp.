#SINet+edge
import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
from scipy import misc  # NOTES: pip install scipy == 1.2.2
import imageio
from Src.SINet import SINet_ResNet50
from Src.utils.dataloader import test_dataset
from Src.utils.trainer2 import eval_mae, numpy2tensor
from PIL import Image
import cv2
import matplotlib.image as mpimg
import torchvision.transforms as T
from skimage import img_as_ubyte
from skimage import io
import skimage


parser = argparse.ArgumentParser() 
parser.add_argument('--testsize', type=int, default=352, help='the snapshot input size')
parser.add_argument('--model_path', type=str,
                default='/home/kai/Desktop/xxq/SINet-master/Snapshot/6_27/SINet_60.pth')
parser.add_argument('--test_save_res2', type=str,
                    default='./Result/6_27_2/res2/')
parser.add_argument('--test_save_edge', type=str,
                    default='./Result/6_27_2/edge/')
parser.add_argument('--test_save_res1', type=str,
                    default='./Result/6_27_2/res1/')
opt = parser.parse_args()

model = SINet_ResNet50().cuda()
model.load_state_dict(torch.load(opt.model_path))
model.eval()

for dataset in ['COD10K']:
    save_path_res1 = opt.test_save_res1 + dataset + '/'
    save_path_res2 = opt.test_save_res2 + dataset + '/'
    save_path_edge = opt.test_save_edge + dataset + '/'
    os.makedirs(save_path_res1, exist_ok=True)
    os.makedirs(save_path_res2, exist_ok=True)
    os.makedirs(save_path_edge, exist_ok=True)


    test_loader = test_dataset('./Dataset/TestDataset/{}/Image/'.format(dataset),
                               './Dataset/TestDataset/{}/GT/'.format(dataset), 
                               './Dataset/TestDataset/{}/Edge/'.format(dataset), opt.testsize)
    img_count = 1
    MAE = 0
    #MAE3 = 0
    for iteration in range(test_loader.size):
        image, gt, edge, name = test_loader.load_data()

        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        with torch.no_grad():
            cam1,cam2,cam3= model(image)

        
        #print(gt.shape)
        cam1 = F.upsample(cam1, size=gt.shape, mode='bilinear', align_corners=True)
        cam2 = F.upsample(cam2, size=gt.shape, mode='bilinear', align_corners=True)
        cam3 = F.upsample(cam3, size=gt.shape, mode='bilinear', align_corners=True)
        cam1 = cam1.sigmoid().data.cpu().numpy().squeeze()
        cam2 = cam2.sigmoid().data.cpu().numpy().squeeze()
        cam3 = cam3.sigmoid().data.cpu().numpy().squeeze()
        
        # normalize
        cam1 = (cam1 - cam1.min()) / (cam1.max() - cam1.min() + 1e-8)
        cam2 = (cam2 - cam2.min()) / (cam2.max() - cam2.min() + 1e-8)
        cam3 = (cam3 - cam3.min()) / (cam3.max() - cam3.min() + 1e-8)

        #misc.imsave(save_path_res2+name, cam2)
        #misc.imsave(save_path_edge+name, cam3),cam1
        imageio.imsave(save_path_res1+name, cam1)
        #print(cam3)
        
        imageio.imsave(save_path_res2+name, cam2)
        imageio.imsave(save_path_edge+name, cam3)
        #io.imsave(os.path.join(save_path,name),img_as_ubyte(cam))
    
   

        mae = eval_mae(numpy2tensor(cam2), numpy2tensor(gt))
        #mae3 = eval_mae(numpy2tensor(cam3), numpy2tensor(gt))

        '''if mae>mae3:
            misc.imsave(save_path+name, cam3)
        else:
            misc.imsave(save_path+name, cam)'''
        # coarse score
        data = open("/home/kai/Desktop/xxq/SINet-master/6_27test.txt","a")
        print('[Eval-Test] Dataset: {}, Image: {} ({}/{}), MAE: {}'.format(dataset, name, img_count,
                                                                           test_loader.size, mae),file=data)#, mae3, MAE3: {}
        data.close()
        MAE = MAE + mae
        #MAE3 = MAE3 +mae3

        img_count += 1

avg_MAE = MAE/test_loader.size
#avg_MAE3 = MAE3/test_loader.size
print("avg_MAE: ",avg_MAE)
#print("avg_MAE3: ",avg_MAE3)
print("\n[Congratulations! Testing Done]")


