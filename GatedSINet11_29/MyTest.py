#SINet+edge
import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
import imageio
from Src.SINet import SINet_ResNet50
from Src.utils.dataloader import test_dataset
from Src.utils.trainer2 import eval_mae, numpy2tensor
import torchvision


parser = argparse.ArgumentParser() 
parser.add_argument('--testsize', type=int, default=352, help='the snapshot input size')
parser.add_argument('--model_path', type=str,
                default='Snapshot/11_29GatedSINet/GatedSINet1129_Incep_lrclip_190.pth')
parser.add_argument('--test_save_midresult', type=str,
                    default='./Result/GatedSINet/midresult/')
parser.add_argument('--test_save_edge', type=str,
                    default='./Result/GatedSINet/edge/')
parser.add_argument('--test_save_result', type=str,
                    default='./Result/GatedSINet/result/')
opt = parser.parse_args()

model = SINet_ResNet50().cuda()
model.load_state_dict(torch.load(opt.model_path))
model.eval()

for dataset in ['CAMO']:
    save_path_midresult = opt.test_save_midresult + dataset + '/'
    save_path_result = opt.test_save_result + dataset + '/'
    save_path_edge = opt.test_save_edge + dataset + '/'
    os.makedirs(save_path_midresult, exist_ok=True)
    os.makedirs(save_path_result, exist_ok=True)
    os.makedirs(save_path_edge, exist_ok=True)


    test_loader = test_dataset('./Dataset/TestDataset/{}/Image/'.format(dataset),
                               './Dataset/TestDataset/{}/GT/'.format(dataset), 
                               './Dataset/TestDataset/{}/Edge/'.format(dataset), opt.testsize)
    img_count = 1
    MAE_camou = 0
    MAE_edge = 0
    for iteration in range(test_loader.size):
        image, gt, edge, name = test_loader.load_data()

        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        edge = np.asarray(edge, np.float32)
        edge /= (edge.max() + 1e-8)
        image = image.cuda()


        with torch.no_grad():
            x4_res, x3_res, x2_res, x1_res, edgeres= model(image)
            results = [x4_res, x3_res, x2_res, x1_res]
            res = x1_res

        results_all = torch.zeros((len(results), 1, 352, 352))
        for i in range(len(results)):
            results_all[i, 0, :, :] = results[i]

        torchvision.utils.save_image(results_all, os.path.join(save_path_midresult, name))

        
        net_res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=True)
        edgeres = F.upsample(edgeres, size=edge.shape, mode='bilinear', align_corners=True)
        net_res = net_res.sigmoid().data.cpu().numpy().squeeze()
        edgeres = edgeres.sigmoid().data.cpu().numpy().squeeze()
        
        imageio.imsave(save_path_result+name, net_res)
        
        imageio.imsave(save_path_edge+name, edgeres)
        # normalize
        net_res = (net_res - net_res.min()) / (net_res.max() - net_res.min() + 1e-8)
        edgeres = (edgeres - edgeres.min()) / (edgeres.max() - edgeres.min() + 1e-8)

        mae_camou = eval_mae(numpy2tensor(net_res), numpy2tensor(gt))
        mae_edge = eval_mae(numpy2tensor(edgeres), numpy2tensor(edge))


        data = open("/home/kai/Desktop/xxq/SINet-master/GatedSINet1129Test.txt","a")
        print('[Eval-Test] Dataset: {}, Image: {} ({}/{}), MAE_camou: {}, MAE_edge: {}'.format(dataset, name, img_count,
                                                                           test_loader.size, mae_camou, mae_edge),file=data)
        data.close()
        MAE_camou = MAE_camou + mae_camou
        MAE_edge = MAE_edge +mae_edge

        img_count += 1

avg_MAE_camou = MAE_camou/test_loader.size
avg_MAE_edge = MAE_edge/test_loader.size
print("avg_MAE_camou: ",avg_MAE_camou)
print("avg_MAE_edge: ",avg_MAE_edge)
print("\n[Congratulations! Testing Done]")
