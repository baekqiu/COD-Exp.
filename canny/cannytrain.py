import torch
import argparse 
from Cannynet import Cannydetection
from dataloadcanny import get_loader
from cannytrainer import boundarytrainer, adjust_lr
from apex import amp
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=240,
                        help='epoch number, default=30')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='init learning rate, try `lr=1e-4`')
    parser.add_argument('--batchsize', type=int, default=2,
                        help='training batch size (Note: ~500MB per img in GPU)')
    parser.add_argument('--trainsize', type=int, default=352,
                        help='the size of training image, try small resolutions for speed (like 256)')
    parser.add_argument('--clip', type=float, default=0.5,
                        help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1,
                        help='decay rate of learning rate per decay step')
    parser.add_argument('--decay_epoch', type=int, default=30,
                        help='every N epochs decay lr')
    parser.add_argument('--gpu', type=int, default=1,
                        help='choose which gpu you use')
    parser.add_argument('--save_epoch', type=int, default=2,
                        help='every N epochs save your trained snapshot')
    parser.add_argument('--save_model', type=str, default='.canny/boundarycanny')
    parser.add_argument('--train_img_dir', type=str, default='./Dataset/TrainDataset2/Image/')
    parser.add_argument('--train_edge_dir', type=str, default='./Dataset/TrainDataset2/Edge/') 
    parser.add_argument('--train_canny_dir', type=str, default='./Dataset/TrainDataset2/FractionalEdge/')
    opt = parser.parse_args()

    torch.cuda.set_device(opt.gpu)
    #torch.cuda.set_device('cuda:'+str('01'))

    # TIPS: you also can use deeper network for better performance like channel=64
    model_Boundary = Cannydetection().cuda()
    #model_Boundary.load_state_dict(torch.load('/home/kai/Desktop/xxq/SINet-master/Snapshot/boundarycanny/BoundaryNetFractional_20.pth'))
   
    #print('-' * 30, model_SINet, '-' * 30)

    optimizer = torch.optim.Adam(model_Boundary.parameters(), opt.lr)
    LogitsBCE = torch.nn.BCEWithLogitsLoss()
    #LogitsBCE = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([9]).to(opt.gpu))

    net, optimizer = amp.initialize(model_Boundary, optimizer, opt_level='O1')     # NOTES: Ox not 0x

    train_loader = get_loader(opt.train_img_dir, opt.train_edge_dir, opt.train_canny_dir, batchsize=opt.batchsize,
                              trainsize=opt.trainsize, num_workers=12)
    total_step = len(train_loader)

    print('-' * 30, "\n[Training Dataset INFO]\nimg_dir: {}\nedge_gt_dir: {}\nedge_dir: {}\nLearning Rate: {}\nBatch Size: {}\n"
                    "Training Save: {}\ntotal_num: {}\n".format(opt.train_img_dir, opt.train_edge_dir, opt.train_canny_dir, opt.lr,
                                                              opt.batchsize, opt.save_model, total_step), '-' * 30)

    plt.figure()
    
    y = []
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 40, eta_min=0) 
    for epoch_iter in range(1, opt.epoch):
        #adjust_lr(optimizer, epoch_iter, opt.decay_rate, opt.decay_epoch)
        
        boundarytrainer(train_loader=train_loader, model=model_Boundary,
                optimizer=optimizer, epoch=epoch_iter,
                opt=opt, loss_func=LogitsBCE, total_step=total_step)
        scheduler.step()
        y.append(scheduler.get_lr()[0])
        print("learning rate:  ", scheduler.get_lr()[0])


    x = list(range(len(y)))
    plt.plot(x,y)
    plt.title("learing rate's curve changes as epoch goes on!")
    plt.savefig('./FractionalEdge.jpg')
    plt.show()
