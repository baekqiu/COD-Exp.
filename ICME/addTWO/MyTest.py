import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
from scipy import misc  # NOTES: pip install scipy == 1.2.2
from Src.SINet import SINet_ResNet50
from Src.utils.Dataloader import test_dataset
from Src.utils.trainer import eval_mae, numpy2tensor
from PIL import Image
import cv2
import matplotlib.image as mpimg
import torchvision.transforms as T


parser = argparse.ArgumentParser() 
parser.add_argument('--testsize', type=int, default=352, help='the snapshot input size')
parser.add_argument('--model_path', type=str,
                    default='./Snapshot/2020-CVPR-SINet/SINet_20.pth')
parser.add_argument('--test_save', type=str,
                    default='./Result/2020-CVPR-SINet-New/')
opt = parser.parse_args()

model = SINet_ResNet50().cuda()
model.load_state_dict(torch.load(opt.model_path))
model.eval()

for dataset in ['COD10K']:
    save_path = opt.test_save + dataset + '/'
    os.makedirs(save_path, exist_ok=True)

    test_loader = test_dataset('./Dataset/TestDataset/{}/Image/'.format(dataset),
                               './Dataset/TestDataset/{}/GT/'.format(dataset), opt.testsize)
    img_count = 1
    MAE = 0
    for iteration in range(test_loader.size):
        image, gt, name = test_loader.load_data()

        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        _, cam = model(image)
        
        #print(gt.shape)
        cam = F.upsample(cam, size=gt.shape, mode='bilinear', align_corners=True)
        cam = cam.sigmoid().data.cpu().numpy().squeeze()
        
        # normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        

        misc.imsave(save_path+name, cam)
        
        


        mae = eval_mae(numpy2tensor(cam), numpy2tensor(gt))
        # coarse score
        data = open("/home/kai/Desktop/xxq/SINet-master/log.txt","a")
        print('[Eval-Test] Dataset: {}, Image: {} ({}/{}), MAE: {}'.format(dataset, name, img_count,
                                                                           test_loader.size, mae),file=data)
        data.close()
        MAE = MAE + mae

        img_count += 1

avg_MAE = MAE/test_loader.size
print("avg_MAE: ",avg_MAE)
print("\n[Congratulations! Testing Done]")


