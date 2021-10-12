#dataloader only for boundary detection
import os
import tarfile
from PIL import Image
from scipy.stats.stats import _two_sample_transform
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import numpy as np
import torch
import torch.distributed as dist


class CamObjDataset(data.Dataset):
    def __init__(self, image_root, edge_root, trainsize):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.edges = [edge_root + f for f in os.listdir(edge_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.images = sorted(self.images)
        self.edges = sorted(self.edges)
        self.filter_files()
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.edge_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        edge = self.binary_loader(self.edges[index])
        image = self.img_transform(image)
        edge = self.edge_transform(edge)
        return image, edge

    def filter_files(self):
        #print(len(self.images),len(self.edges))
        assert len(self.images) == len(self.edges)
        images = []
        edges = []
        for img_path, edge_path in zip(self.images, self.edges):
            img = Image.open(img_path)
            edge = Image.open(edge_path)
            if img.size == edge.size:
                images.append(img_path)
                edges.append(edge_path)
        self.images = images
        self.edges = edges

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def resize(self, img, edge):
        assert img.size == edge.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), edge.resize((w, h), Image.NEAREST)
        else:
            return img, edge

    def __len__(self):
        return self.size


class test_dataset:
    """load test dataset (batchsize=1)"""
    def __init__(self, image_root, edge_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.edges = [edge_root + f for f in os.listdir(edge_root) if f.endswith('.jpg')
                       or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                       or f.endswith('.png')]
        self.images = sorted(self.images)
        self.edges = sorted(self.edges)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.edge_transform = transforms.ToTensor()
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        edge = self.binary_loader(self.edges[self.index])
        gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, edge, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')




class test_loader_faster(data.Dataset):
    def __init__(self, image_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.images = sorted(self.images)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.size = len(self.images)

    def __getitem__(self, index):
        images = self.rgb_loader(self.images[index])
        images = self.transform(images)

        img_name_list = self.images[index]

        return images, img_name_list

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def tensor_loader(self):
        image = torch.tensor.cpu().clone()
        image = image.squeeze(0)
        image = torch.unloader(image)
        return image

    def __len__(self):
        return self.size


def get_loader(image_root, edge_root, local_rank, batchsize, trainsize, shuffle=True, num_workers=12, pin_memory=True):
    # `num_workers=0` for more stable training
    dataset = CamObjDataset(image_root, edge_root, trainsize)
    word_size = dist.get_world_size()
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset,num_replicas = word_size,rank = local_rank)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=(train_sampler is None),
                                  num_workers=num_workers,
                                  pin_memory=pin_memory,
                                  sampler=train_sampler)

    return data_loader
