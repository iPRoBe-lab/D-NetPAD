import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import glob
import os
import csv
import numpy as np
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    device = torch.device('cuda')
    parser.add_argument('-imageFolder', default='CroppedImages',type=str)
    parser.add_argument('-modelPath',  default='Model/D-NetPAD_Model.pth',type=str)
    args = parser.parse_args()


    # Load weights of single binary DesNet121 model
    weights = torch.load(args.modelPath)
    DNetPAD = models.densenet121(pretrained=True)
    num_ftrs = DNetPAD.classifier.in_features
    DNetPAD.classifier = nn.Linear(num_ftrs, 2)
    DNetPAD.load_state_dict(weights['state_dict'])
    DNetPAD = DNetPAD.to(device)
    DNetPAD.eval()


    # Transformation specified for the pre-processing
    transform = transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485], std=[0.229])
            ])

    imagesScores=[]

    imageFiles = glob.glob(os.path.join(args.imageFolder,'*.jpg'))
    for imgFile in imageFiles:

            # Read the image
            image = Image.open(imgFile)

            # Image transformation
            tranformImage = transform(image)
            image.close()
            tranformImage = tranformImage.repeat(3, 1, 1) # for NIR images having one channel
            tranformImage = tranformImage[0:3,:,:].unsqueeze(0)
            tranformImage = tranformImage.to(device)

            # Output from single binary CNN model
            output = DNetPAD(tranformImage)
            PAScore = output.detach().cpu().numpy()[:, 1]

            # Normalization of output score between [0,1]
            PAScore = np.minimum(np.maximum((PAScore+15)/35,0),1)
            imagesScores.append([imgFile, PAScore[0]])


    # Writing the scores in the csv file
    with open(os.path.join(args.imageFolder,'Scores.csv'),'w',newline='') as fout:
        writer = csv.writer(fout)
        writer.writerows(imagesScores)
