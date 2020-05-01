import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import glob
import math
import os
import csv
import numpy as np
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    device = torch.device('cuda')
    parser.add_argument('-imageFolder', default='Imagessdfsd',type=str)
    parser.add_argument('-modelPath',  default='Model/DensePAD_Model.pth',type=str)
    args = parser.parse_args()


    # Load weights of single binary DesNet121 model
    weights = torch.load(args.modelPath)
    DensePAD = models.densenet121(pretrained=True)
    num_ftrs = DensePAD.classifier.in_features
    DensePAD.classifier = nn.Linear(num_ftrs, 2)
    DensePAD.load_state_dict(weights['state_dict'])
    DensePAD = DensePAD.to(device)
    DensePAD.eval()


    # Transformation specified for the pre-processing
    transform = transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485], std=[0.229])
            ])

    imagesScores=[]

    imageFiles = glob.glob(os.path.join(args.imageFolder,'*.bmp'))
    for imgFile in imageFiles:

            # Read the image
            image = Image.open(imgFile)

            #Segmentation and cropping of an image. We have used Veri Segmentor, other segmentation technique could be used
            if not os.path.exists('tempVeriEye'):
                os.mkdir('tempVeriEye')
            f = open('tempVeriEye/tempSegFile.csv', 'w+')
            f.write(imgFile)
            f.close()
            os.system('VeriEyeSegmenter %s %s %s no' % ('tempVeriEye/tempSegFile.csv', 'tempVeriEye', 'null'))
            if os.path.isfile('tempVeriEye/segmentationDetails.csv'):
                f = open('tempVeriEye/segmentationDetails.csv', "r")
                s = f.readline()
                SegInfo = f.readline().split(',')[1:4]
                if len(SegInfo) == 3:
                    min_x = math.floor(int(SegInfo[0]) - int(SegInfo[2]) - 5)
                    min_y = math.floor(int(SegInfo[1]) - int(SegInfo[2]) - 5)
                    max_x = math.floor(int(SegInfo[0]) + int(SegInfo[2]) + 5)
                    max_y = math.floor(int(SegInfo[1]) + int(SegInfo[2]) + 5)
                    image = image.crop([min_x, min_y, max_x, max_y])


            # Image transformation
            tranformImage = transform(image)
            image.close()
            tranformImage = tranformImage.repeat(3, 1, 1)
            tranformImage = tranformImage[0:3,:,:].unsqueeze(0)
            tranformImage = tranformImage.to(device)

            # Output from single binary CNN model
            output = DensePAD(tranformImage)
            PAScore = output.detach().cpu().numpy()[:, 1]

            # Normalization of output score between [0,1]
            PAScore = np.minimum(np.maximum((PAScore+15)/35,0),1)
            imagesScores.append([imgFile, PAScore[0]])


    # Writing the scores in the csv file
    with open(os.path.join(args.imageFolder,'Scores.csv'),'w',newline='') as fout:
        writer = csv.writer(fout)
        writer.writerows(imagesScores)
