#SINet+edge
import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
from scipy import misc  # NOTES: pip install scipy == 1.2.2
import imageio
from Src.SINetres2netcopy import SINet_Res2Net50
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
                default='/home/kai/Desktop/xxq/SINet-master/Snapshot/7/7_smimedgeresnet/SINet_100.pth')
parser.add_argument('--test_save_res', type=str,
                    default='./Result/7_8_smimedgeresnet/')
opt = parser.parse_args()

model = SINet_Res2Net50().cuda()
model.load_state_dict(torch.load(opt.model_path))
model.eval()

for dataset in ['11']:
    save_path_res = opt.test_save_res + dataset + '/'
    os.makedirs(save_path_res, exist_ok=True)

    test_loader = test_dataset('./Dataset/TestDataset/{}/Image/'.format(dataset),
                               './Dataset/TestDataset/{}/GT/'.format(dataset), 
                               './Dataset/TestDataset/{}/Edge/'.format(dataset), opt.testsize)
    img_count = 1
    MAE = 0
    #MAE3 = 0
    for iteration in range(test_loader.size):
        image, gt, edge, name = test_loader.load_data()

        '''gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)'''
        image = image.cuda()

        with torch.no_grad():
            cam1,cam2, edge= model(image)

        
        #print(gt.shape)
        '''cam2 = F.upsample(cam2, size=gt.shape, mode='bilinear', align_corners=True)'''
        cam2 = cam2.sigmoid().data.cpu().numpy().squeeze()
        
        # normalize
        cam2 = (cam2 - cam2.min()) / (cam2.max() - cam2.min() + 1e-8)
        imageio.imsave(save_path_res+name, cam2)
        #print(cam3)
        #io.imsave(os.path.join(save_path,name),img_as_ubyte(cam))
    
   

        '''mae = eval_mae(numpy2tensor(cam2), numpy2tensor(gt))'''
        #mae3 = eval_mae(numpy2tensor(cam3), numpy2tensor(gt))

        '''if mae>mae3:
            misc.imsave(save_path+name, cam3)
        else:
            misc.imsave(save_path+name, cam)'''
        # coarse score
        '''data = open("/home/kai/Desktop/xxq/SINet-master/7_7test.txt","a")
        print('[Eval-Test] Dataset: {}, Image: {} ({}/{}), MAE: {}'.format(dataset, name, img_count,
                                                                           test_loader.size, mae),file=data)#, mae3, MAE3: {}
        data.close()
        MAE = MAE + mae
        #MAE3 = MAE3 +mae3'''

        img_count += 1

'''avg_MAE = MAE/test_loader.size
#avg_MAE3 = MAE3/test_loader.size
print("avg_MAE: ",avg_MAE)
#print("avg_MAE3: ",avg_MAE3)'''
print("\n[Congratulations! Testing Done]")


