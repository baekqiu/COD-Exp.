import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import numpy as np
import torch


class CamObjDataset(data.Dataset):
    def __init__(self, image_root, edge_root, canny_root, trainsize):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.cannys = [canny_root + f for f in os.listdir(canny_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.edges = [edge_root + f for f in os.listdir(edge_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        
        self.images = sorted(self.images)
        self.cannys = sorted(self.cannys)
        self.edges = sorted(self.edges)
        #self.filter_files()
        #print("len(self.cannys)",len(self.cannys))
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.canny_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.edge_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index]) 
        edge = self.binary_loader(self.edges[index])
        canny = self.binary_loader(self.cannys[index])
        image = self.img_transform(image)
        canny = self.canny_transform(canny)
        edge = self.edge_transform(edge)
        return image, canny, edge

    def filter_files(self):
        assert (len(self.images) == len(self.edges))&(len(self.cannys) == len(self.edges))
        images = []
        cannys = []
        edges = []
        for img_path, canny_path, edge_path in zip(self.images, self.cannys, self.edges):
            img = Image.open(img_path)
            canny = Image.open(canny_path)
            edge = Image.open(edge_path)
            if (img.size == edge.size) & (edge.size == canny.size):
                images.append(img_path)
                cannys.append(canny_path)
                edges.append(edge_path)
        self.images = images
        self.cannys = canny
        self.edges = edges

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def resize(self, img, canny, edge):
        assert (img.size == edge.size)&(img.size == canny.size)
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), canny.resize((w, h), Image.NEAREST), edge.resize((w, h), Image.NEAREST)
        else:
            return img, canny, edge

    def __len__(self):
        return self.size


class test_dataset:
    """load test dataset (batchsize=1)"""
    def __init__(self, image_root, canny_root, edge_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.cannys = [canny_root + f for f in os.listdir(canny_root) if f.endswith('.jpg')
                       or f.endswith('.png')]
        self.edges = [edge_root + f for f in os.listdir(edge_root) if f.endswith('.jpg')
                       or f.endswith('.png')]
        self.images = sorted(self.images)
        self.cannys = sorted(self.cannys)
        self.edges = sorted(self.edges)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.canny_tranform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor()])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        edge = self.binary_loader(self.edges[self.index])
        canny = self.binary_loader(self.cannys[self.index])
        image = self.transform(image).unsqueeze(0)
        canny = self.canny_tranform(canny)
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, canny, edge, name

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


def get_loader(image_root, canny_root, edge_root, batchsize, trainsize, shuffle=True, num_workers=0, pin_memory=True):
    # `num_workers=0` for more stable training
    dataset = CamObjDataset(image_root, canny_root, edge_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)

    return data_loader
