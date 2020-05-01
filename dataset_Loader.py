import torch
import torch.utils.data as data_utl
import glob
import numpy as np
import random
import os
import cv2
from PIL import Image
from matplotlib import cm
import torchvision.transforms as transforms
import csv
import math

class datasetLoader(data_utl.Dataset):

    def __init__(self, split_file, root, train_test, random=True, c2i={}):
        self.class_to_id = c2i
        self.id_to_class = []

        # Class assignment
        for i in range(len(c2i.keys())):
            for k in c2i.keys():
                if c2i[k] == i:
                    self.id_to_class.append(k)
        cid = 0

        # Image pre-processing
        self.data = []
        self.transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485],std=[0.229])
            ])

        # Reading data from CSV file
        SegInfo=[]
        with open(split_file, 'r') as f:
            for l in f.readlines():
                if len(l) <= 5:
                    continue
                v= l.strip().split(',')
                if train_test == v[0]:
                    image_name = v[2]
                    imagePath = root +image_name
                    if len(v)==6:
                        SegInfo = [v[3], v[4], v[5]]
                    c = v[1]
                    if c not in self.class_to_id:
                        self.class_to_id[c] = cid
                        self.id_to_class.append(c)
                        cid += 1
                    # Storing data with imagepath class and seginfo
                    self.data.append([imagePath, self.class_to_id[c], SegInfo])


        self.split_file = split_file
        self.root = root
        self.random = random
        self.train_test = train_test


    def __getitem__(self, index):
        imagePath, cls, SegInfo = self.data[index]
        imageName = imagePath.split('\\')[-1]

        # Reading of the image
        path = imagePath
        img = Image.open(path)

        # Segmentation of the iris image
        if len(SegInfo)==3:
            min_x = math.floor(int(SegInfo[0]) - int(SegInfo[2]) - 5)
            min_y = math.floor(int(SegInfo[1]) - int(SegInfo[2]) - 5)
            max_x = math.floor(int(SegInfo[0]) + int(SegInfo[2]) + 5)
            max_y = math.floor(int(SegInfo[1]) + int(SegInfo[2]) + 5)
            img = img.crop([min_x,min_y,max_x,max_y])

        # Applying transformation
        tranform_img = self.transform(img)
        img.close()

        # Repeat NIR channel thrice before feeding into the network
        tranform_img= tranform_img.repeat(3,1,1)

        return tranform_img[0:3,:,:], cls, imageName

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':

    dataseta = IARPA_dataset('../TempData/Iris_OCT_Splits_Val/test_train_split03.csv', 'G:/My Drive/Renu/IPARA-Project-Documentation/GCT2/OCT_Data/', train_test='train')

    for i in range(len(dataseta)):
        print(len(dataseta.data))
