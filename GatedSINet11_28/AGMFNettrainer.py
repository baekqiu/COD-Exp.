from numpy.core.defchararray import array
import torch
from torch.autograd import Variable
from datetime import datetime 
import os
from apex import amp
from torch.autograd.grad_mode import no_grad
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
import numpy as np
from torchvision import transforms



def eval_mae(y_pred, y):
    """
    evaluate MAE (for test or validation phase)
    :param y_pred:
    :param y:
    :return: Mean Absolute Error
    """
    return torch.abs(y_pred - y).mean()


def numpy2tensor(numpy):
    """
    convert numpy_array in cpu to tensor in gpu
    :param numpy:
    :return: torch.from_numpy(numpy).cuda()
    """
    return torch.from_numpy(numpy).cuda()


def clip_gradient(optimizer, grad_clip):
    """
    recalibrate the misdirection in the training
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay


def trainer(train_loader, model, optimizer, epoch, opt, loss_func, total_step):
    """
    Training iteration
    :param train_loader:
    :param model:
    :param optimizer:
    :param epoch:
    :param opt:
    :param loss_func:
    :param total_step:
    :return:
    """
    #f1 = open('/home/kai/Desktop/xxq/SINet-master/log.txt','w+')
    model.train()
    for step, data_pack in enumerate(train_loader):
        optimizer.zero_grad()
        images, edges, gts = data_pack
        images = Variable(images).cuda()
        gts = Variable(gts).cuda()
        edges = Variable(edges).cuda()

        #num = 0

        cam_edge2, cam_sm, cam_im= model(images)#, cam_em 
        loss_sm = loss_func(cam_sm, gts)
        loss_im = loss_func(cam_im, gts)
        #loss_edge = loss_func(cam_edge, edges)
        loss_edge2 = loss_func(cam_edge2, edges)
        loss_total = 0.5*loss_sm + 1.0*loss_im + 0.5*loss_edge2
        with amp.scale_loss(loss_total, optimizer) as scale_loss:
            scale_loss.backward()

        clip_gradient(optimizer, opt.clip)
        optimizer.step()
        #scheduler.step()
       
        if step % 40 == 0 or step == total_step:
            
            data = open("traincannylog.txt","a")
            print('[{}] => [Epoch Num: {:03d}/{:03d}] => [Global Step: {:04d}/{:04d}] => [Loss_s: {:.4f} Loss_i: {:0.4f} Loss_edge2: {:.4f} Loss_total: {:.4f}]'. 
                  format(datetime.now(), epoch, opt.epoch, step, total_step, loss_sm.data, loss_im.data,  loss_edge2.data, loss_total.data)) # 
            print('[{}] => [Epoch Num: {:03d}/{:03d}] => [Global Step: {:04d}/{:04d}] => [Loss_s: {:.4f} Loss_i: {:0.4f} Loss_edge2: {:.4f} Loss_total: {:.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, step, total_step, loss_sm.data, loss_im.data,  loss_edge2.data, loss_total.data), file=data)
            data.close()
        
  

    save_path = opt.save_model
    os.makedirs(save_path, exist_ok=True)

    if (epoch+1) % opt.save_epoch == 0 :
        torch.save(model.state_dict(), save_path + 'GatedSINet_%d.pth' % (epoch+1))


    
    