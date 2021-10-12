#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 boundarytrain.py

import torch
import argparse 
import numpy as np
from boundarynet import Boundarydetection
from dataloader1 import get_loader
from boundarytrainer import boundarytrainer
from apex import amp
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import torchvision
import sys
from torchvision import datasets, transforms
import os
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import copy
from torch.nn import DataParallel


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int,default=-1,help='DDP parameter, do not modify')#不需要赋值，启动命令 torch.distributed.launch会自动赋值
    parser.add_argument("--distribute",action='store_true',help='whether using multi gpu train')
    parser.add_argument("--distribute_mode",type=str,default='DDP',help="using which mode to ")
    parser.add_argument('--epoch', type=int, default=1200,
                        help='epoch number, default=30')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='init learning rate, try `lr=1e-4`')
    parser.add_argument('--batchsize', type=int, default=4,
                        help='training batch size (Note: ~500MB per img in GPU)')
    parser.add_argument('--clip', type=float, default=0.5,
                        help='gradient clipping margin')
    parser.add_argument('--trainsize', type=int, default=352,
                        help='the size of training image, try small resolutions for speed (like 256)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='choose which gpu you use')
    parser.add_argument('--save_epoch', type=int, default=10,
                        help='every N epochs save your trained snapshot') 
    parser.add_argument('--save_model', type=str, default='./Snapshot/gate-boundary/')
    parser.add_argument('--train_img_dir', type=str, default='./TrainDataset2/Image/')
    parser.add_argument('--train_edge_dir', type=str, default='./TrainDataset2/Edge/') 
    opt = parser.parse_args()

    #torch.cuda.set_device(0)

    torch.distributed.init_process_group(backend='nccl', init_method="env://", world_size=torch.cuda.device_count(),
                                             rank=opt.local_rank)
    torch.cuda.set_device(opt.local_rank)

    train_loader = get_loader(opt.train_img_dir, opt.train_edge_dir, opt.local_rank, batchsize=opt.batchsize,
                              trainsize=opt.trainsize, num_workers=1)
    total_step = len(train_loader)

    print('-' * 30, "\n[Training Dataset INFO]\nimg_dir: {}\nedge_gt_dir: {}\nLearning Rate: {}\nBatch Size: {}\n"
                    "Training Save: {}\ntotal_num: {}\n".format(opt.train_img_dir, opt.train_edge_dir, opt.lr,
                                                              opt.batchsize, opt.save_model, total_step), '-' * 30)
    # TIPS: you also can use deeper network for better performance like channel=64
    model_Boundary = Boundarydetection().cuda(opt.local_rank)
    model_Boundary.load_state_dict(torch.load('Snapshot/gate-boundary/gate-BoundaryNet_60.pth'))
   
    model = DDP(model_Boundary, device_ids=[opt.local_rank], find_unused_parameters=True)
    
    #print('-' * 30, model_SINet, '-' * 30)

    optimizer = torch.optim.Adam(model_Boundary.parameters(), opt.lr)
    LogitsBCE = torch.nn.BCEWithLogitsLoss()
    #LogitsBCE = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([9]).to(opt.gpu))

    net, optimizer = amp.initialize(model_Boundary, optimizer, opt_level='O1')     # NOTES: Ox not 0x

    plt.figure()
    
    y = []
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 40, eta_min=0)
    for epoch_iter in range(1, opt.epoch):
        boundarytrainer(train_loader=train_loader, model=model_Boundary,
                optimizer=optimizer, epoch=epoch_iter,
                opt=opt, loss_func=LogitsBCE, total_step=total_step)
        scheduler.step()
        y.append(scheduler.get_lr()[0])
        print("learning rate:  ", scheduler.get_lr()[0])


    x = list(range(len(y)))
    plt.plot(x,y)
    plt.title("learing rate's curve changes as epoch goes on!")
    plt.savefig('./gate-boundary.jpg')
    plt.show()