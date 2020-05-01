import os
import argparse
import csv

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-sourceFile', required=True, default='../TempData/Iris_IARPA_Splits/test_train_splitGCT3.csv',type=str)
    parser.add_argument('-destFile', required=True, default='../TempData/Iris_IARPA_Splits/test_train_splitGCT3-Seg.csv', type=str)
    parser.add_argument('-datasetRoot', required=True,default='G:/My Drive/Renu/Iris_Image_Database/', type=str)
    args = parser.parse_args()
    dataWrite =[]

    with open(args.sourceFile, 'r') as f:
        for l in f.readlines():
            v = l.strip().split(',')
            imagePartialPath = v[2]
            imageName = imagePartialPath.split('\\')[-1]

            imagePath = args.datasetRoot + imagePartialPath

            # Getting the segmentation info of a image using VeriEyeSegmenter
            if not os.path.exists('tempVeriEye'):
                os.mkdir('tempVeriEye')
            f = open('tempVeriEye/tempSegFile.csv', 'w+')
            f.write(imagePath)
            f.close()
            os.system('VeriEyeSegmenter %s %s %s no' % ('tempVeriEye/tempSegFile.csv', 'tempVeriEye', 'null'))
            if os.path.isfile('tempVeriEye/segmentationDetails.csv'):
                f = open('tempVeriEye/segmentationDetails.csv', "r")
                s = f.readline()
                segDetails = f.readline().split(',')[1:4]
                if len(segDetails) == 3:
                    dataWrite.append([v[0],v[1],v[2], segDetails[0], segDetails[1], segDetails[2]])


    # Writing segmentation info into the file
    with open(args.destFile,'w', newline='') as fout:
        writer = csv.writer(fout)
        writer.writerows(dataWrite)