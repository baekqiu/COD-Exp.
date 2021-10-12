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


def boundarytrainer(train_loader, model, optimizer, epoch, opt, loss_func, total_step):
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
            #print(images.shape)
        #img = np.array(images)

        #灰度化处理
        #img = 255*np.array(images).astype('uint8')
        #print(type(images))
        #img = cv2.imread('images')
        #img = img.numpy()
        #print(images)

        #print(type(img))
        #print(img.shape)
        #img = cv2.imread('/home/kai/Desktop/xxq/SINet-master/Dataset/TrainDataset2/Image/camourflage_00003.jpg')
        
    '''
        img = images.narrow(0, 0, 3)
        print(img.shape)
        #img = images.permute(1,2,0)
        transform1 = transforms.ToPILImage(mode="L")
        imge1 = transform1(np.uint8(img.numpy()))
        img = transforms.transforms.ToTensor()(imge1).numpy()
        grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #高斯滤波降噪
        gaussian = cv2.GaussianBlur(grayImage, (3, 3), 0)

        #腐蚀化
        kernel = np.ones((5,5), np.uint8)
        k_ys = cv2.morphologyEx(gaussian, cv2.MORPH_OPEN, kernel)

        #Canny算子
        Canny = cv2.Canny(k_ys, 50, 150)
        print(Canny.shape)
        edgeimage = torch.from_numpy(edgeimage)

       
        edgeimage = Variable(edgeimage).cuda()
    '''
    #f1 = open('/home/kai/Desktop/xxq/SINet-master/log.txt','w+')
    model.train()
    for step, data_pack in enumerate(train_loader): 
        
        optimizer.zero_grad()
        images, edge= data_pack

        images = Variable(images).cuda()
        edge = Variable(edge).cuda()
        

    #num = 0
        
        boundaryimg=model(images)
        
        loss_edge = loss_func(boundaryimg, edge)
        
    
        with amp.scale_loss(loss_edge, optimizer) as scale_loss:
            scale_loss.backward()

        clip_gradient(optimizer, opt.clip)
        optimizer.step()
        
        if step % 40 == 0 or step == total_step:
            
            data = open("traincannylog.txt","a")
            print('[{}] => [Epoch Num: {:03d}/{:03d}] => [Global Step: {:04d}/{:04d}] => [Loss_edge: {:.4f}'. 
                  format(datetime.now(), epoch, opt.epoch, step, total_step, loss_edge.data)) # 
            print('[{}] => [Epoch Num: {:03d}/{:03d}] => [Global Step: {:04d}/{:04d}] => [Loss_s: {:.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, step, total_step, loss_edge.data), file=data)
            data.close()
  

    save_path = opt.save_model
    os.makedirs(save_path, exist_ok=True)

    if (epoch+1) % opt.save_epoch == 0 and opt.local_rank==0:
        torch.save(model.state_dict(), save_path + 'gate-BoundaryNet_%d.pth' % (epoch+61))