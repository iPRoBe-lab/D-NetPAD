# D-NetPAD
Code for Iris Presentation Attack Detection

# Requirement
Pytorch, Numpy, Scipy, Pillow

# Testing
python test_D-NetPAD.py -imageFolder Images

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
