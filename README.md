
# D-NetPAD
Code for Iris Presentation Attack Detection based on DenseNet Architecture.

# Requirement
Pytorch, Numpy, Scipy, Pillow

# Description
The D-NetPAD takes a cropped iris image as input and produces a PA score between 0 and 1, where 0 means bonafide and 1 means presentation attack. Sample cropped iris images are provided in CropperImages folder.

<img src="https://github.com/sharmaGIT/D-NetPAD/blob/master/Images/Architecture.jpg" width="800" height="200">

# Testing
The model can be downloaded from here. Copy the model into Model folder and run the command:

python test_D-NetPAD.py -imageFolder CroppedImages

PA score CSV file will be created in the folder of images.

# Training
python train_D-NetPAD.py -csvPath cseFilePath -datasetPath datasetImagesPath -outputPath resultPath

CSV file contains ground truth of datast images. The format of the dataset CSV file is

train,Live,imageFile1.png <br />
train,Spoof,imageFile2.png <br />
test,Live,imageFile3.png <br />
test,Spoof,imageFile4.png <br />

# Fine Tuning
python fineTrain_D-NetPAD.py -csvPath cseFilePath -datasetPath datasetImagesPath -outputPath resultPath

# Citation
If you are using the code, please cite the paper:

Renu Sharma, Arun Ross,"D-NetPAD: An Explainable and Interpretable Iris Presentation Attack Detector",arXiv, 2020.
