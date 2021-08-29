#SINet+edge
import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
from scipy import misc  # NOTES: pip install scipy == 1.2.2
import imageio
from Src.SINetres2net import SINet_Res2Net50
from Src.utils.dataloader1 import test_dataset
from Src.utils.trainer1 import eval_mae, numpy2tensor
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
                default='/home/kai/Desktop/xxq/SINet-master/Snapshot/7/7SMIMRESNET/SINet_120.pth')
parser.add_argument('--test_save_res', type=str,
                    default='./Result/8.26/')
opt = parser.parse_args()

model = SINet_Res2Net50().cuda()
model.load_state_dict(torch.load(opt.model_path))
model.eval()

for dataset in ['11']:
    save_path_res = opt.test_save_res + dataset + '/'
    os.makedirs(save_path_res, exist_ok=True)

    test_loader = test_dataset('./Dataset/TestDataset/{}/'.format(dataset),
                                opt.testsize)
    img_count = 1
    for iteration in range(test_loader.size):
        image, name = test_loader.load_data()

        image = image.cuda()

        with torch.no_grad():
            cam1,cam2= model(image)

        
        #print(gt.shape)
        cam2 = cam2.sigmoid().data.cpu().numpy().squeeze()
        
        # normalize
        cam2 = (cam2 - cam2.min()) / (cam2.max() - cam2.min() + 1e-8)
        imageio.imsave(save_path_res+name, cam2)



        img_count += 1

print("\n[Congratulations! Testing Done]")


