import torch
from torch.utils.data import Dataset
import os
from PIL import Image
from bs4 import BeautifulSoup
from utils import transform

class COCODataset(Dataset):
    def __init__(self, data_folder, split):
        self.split = split.lower()
        self.data_folder = data_folder
        assert self.split in {'trainval', 'test',}

        for top, dir, file in os.walk(os.path.join(self.data_folder, 'VOC' + self.split + '_06-Nov-2007', 'VOCdevkit', 'VOC2007', 'JPEGImages')):
            self.images = sorted(os.path.join(top, name) for name in file)
        for top, dir, file in os.walk(os.path.join(self.data_folder, 'VOC' + self.split + '_06-Nov-2007', 'VOCdevkit', 'VOC2007', 'Annotations')):
            self.annotations = sorted(os.path.join(top, name) for name in file)
        self.classes = {'person': 0, 'bird': 1, 'cat': 2, 'cow': 3, 'dog': 4, 'horse': 5, 'sheep': 6, 'aeroplane': 7, 'bicycle': 8, 'boat': 9, 'bus': 10, 'car': 11, 'motorbike': 12, 'train': 13, 'bottle': 14, 'chair': 15, 'diningtable': 16, 'pottedplant': 17, 'sofa': 18, 'tvmonitor': 19}

    def __getitem__(self, ind):
        image = Image.open(self.images[ind], mode='r')
        image = image.convert('RGB')

        with open(self.annotations[ind], 'r') as f:
            file = f.read()

        reader = BeautifulSoup(file, 'xml')
        labels = [0] * 20
        bboxes = []
        objects = reader.find_all('object')
        for object in objects:
            for name in object.findChildren('name', recursive = False):
                labels[self.classes[name.text]] = 1
            for bbox in object.findChildren('bndbox', recursive = False):
                bboxes.append([float(children.text) for children in bbox.contents if children.text != '\n'])

        labels = torch.FloatTensor(labels)
        bboxes = torch.FloatTensor(bboxes)

        image, bboxes = transform(image, bboxes)

        return image, labels, bboxes
    
    def __len__(self):
        return len(self.images)
    
        

        

