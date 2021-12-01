#trainer with edge

import torch
from torch.autograd import Variable
from datetime import datetime
import os
from apex import amp
import torch.nn.functional as F
import matplotlib.pyplot as plt



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
    model.train()
    for step, data_pack in enumerate(train_loader):
        optimizer.zero_grad()
        images, gts, edges = data_pack
        images = Variable(images).cuda()
        gts = Variable(gts).cuda()
        edges = Variable(edges).cuda()

        #num = 0

        x4_res, x3_res, x2_res, x1_res, edgewithsobel, net_res= model(images)#, cam_em 
        loss4 = loss_func(x4_res, gts)
        loss3 = loss_func(x3_res, gts)
        loss2 = loss_func(x2_res, gts)
        loss1 = loss_func(x1_res, gts)
        loss_res = loss_func(net_res, gts)        
        loss_edge = loss_func(edgewithsobel, edges)
        loss_total = 0.3*loss4 + 0.3*loss3 + 0.5*loss2 + 0.5*loss1 + 1*loss_res + 0.5*loss_edge
        #print(loss_total)
        with amp.scale_loss(loss_total, optimizer) as scale_loss:
            scale_loss.backward()

        # clip_gradient(optimizer, opt.clip)
        optimizer.step()
        #scheduler.step()
       
        if step % 10 == 0 or step == total_step:
            
            data = open("/home/kai/Desktop/xxq/SINet-master/GatedSINet1129train.txt","a")
            print('[{}] => [Epoch Num: {:03d}/{:03d}] => [Global Step: {:04d}/{:04d}] => [Loss_4: {:.4f} Loss_3: {:.4f} Loss_2: {:.4f} Loss_1: {:.4f} Loss_res: {:0.4f} Loss_edge: {:.4f}, Loss_total: {:.4f}]'. 
                  format(datetime.now(), epoch, opt.epoch, step, total_step, loss4.data, loss3.data, loss2.data, loss1.data, loss_res.data, loss_edge.data, loss_total.data)) # 
            print('[{}] => [Epoch Num: {:03d}/{:03d}] => [Global Step: {:04d}/{:04d}] => [Loss_4: {:.4f} Loss_3: {:.4f} Loss_2: {:.4f} Loss_1: {:.4f} Loss_res: {:0.4f} Loss_edge: {:.4f}, Loss_total: {:.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, step, total_step, loss4.data, loss3.data, loss2.data, loss1.data, loss_res.data, loss_edge.data, loss_total.data), file=data)
            data.close()
        
  

    save_path = opt.save_model
    os.makedirs(save_path, exist_ok=True)

    if (epoch+1) % opt.save_epoch == 0:
        torch.save(model.state_dict(), save_path + 'GatedSINet1129_samegate_%d.pth' % (epoch+1))


    
    
