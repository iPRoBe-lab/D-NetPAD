import os
import argparse
import json
from Evaluation import evaluation
import matplotlib.pyplot as plt
import numpy as np
import torchvision.models as models
import torch
import torch.nn as nn
import torch.optim as optim
from dataset_Loader import datasetLoader


# Description of all argument
parser = argparse.ArgumentParser()
parser.add_argument('-batchSize', type=int, default=20)
parser.add_argument('-nEpochs', type=int, default=50)
parser.add_argument('-csvPath', required=True, default= '../TempData/Iris_IARPA_Splits/test_train_split.csv',type=str)
parser.add_argument('-datasetPath', required=True, default= '/PathToDatasetFolder/',type=str)
parser.add_argument('-outputPath', required=True, default= '/OutputPath/',type=str)
parser.add_argument('-method', default= 'DesNet121',type=str)
parser.add_argument('-nClasses', default= 2,type=int)

args = parser.parse_args()
device = torch.device('cuda')


# Definition of model architecture
model = models.densenet121(pretrained=True)
num_ftrs = model.classifier.in_features
model.classifier = nn.Linear(num_ftrs, args.nClasses)
model = model.to(device)


# Creation of Log folder: used to save the trained model
log_path = os.path.join(args.outputPath, 'Logs')
if not os.path.exists(log_path):
    os.mkdir(log_path)


# Creation of result folder: used to save the performance of trained model on the test set
result_path = os.path.join(args.outputPath , 'Results')
if not os.path.exists(result_path):
    os.mkdir(result_path)


# Dataloader for train and test data
dataseta = datasetLoader(args.csvPath,args.datasetPath,train_test='train')
dl = torch.utils.data.DataLoader(dataseta, batch_size=args.batchSize, shuffle=True, num_workers=0, pin_memory=True)
dataset = datasetLoader(args.csvPath,args.datasetPath, train_test='test', c2i=dataseta.class_to_id)
test = torch.utils.data.DataLoader(dataset, batch_size=args.batchSize, shuffle=True, num_workers=0, pin_memory=True)
dataloader = {'train': dl, 'test':test}


# Description of hyperparameters
lr = 0.005
solver = optim.SGD(model.parameters(), lr=lr, weight_decay=1e-6, momentum=0.9)
lr_sched = optim.lr_scheduler.StepLR(solver, step_size=12, gamma=0.1)
criterion = nn.CrossEntropyLoss()


# File for logging the training process
with open(os.path.join(log_path,'params.json'), 'w') as out:
    hyper = vars(args)
    json.dump(hyper, out)
log = {'iterations':[], 'epoch':[], 'validation':[], 'train_acc':[], 'val_acc':[]}

    

#####################################################################################
#
############### Training of the model and logging ###################################
#
#####################################################################################
train_loss=[]
test_loss=[]
bestAccuracy = 0
bestEpoch=0
for epoch in range(args.nEpochs):

    for phase in ['train', 'test']:
        train = (phase=='train')
        if phase == 'train':
            model.train()
        else:
            model.eval()
            
        tloss = 0.
        acc = 0.
        tot = 0
        c = 0
        testPredScore = []
        testTrueLabel = []
        imgNames=[]
        with torch.set_grad_enabled(train):
            for data, cls, imageName in dataloader[phase]:

                # Data and ground truth
                data = data.to(device)
                cls = cls.to(device)

                # Running model over data
                outputs = model(data)

                # Prediction of accuracy
                pred = torch.max(outputs,dim=1)[1]
                corr = torch.sum((pred == cls).int())
                acc += corr.item()
                tot += data.size(0)
                loss = criterion(outputs, cls)

                # Optimization of weights for training data
                if phase == 'train':
                    solver.zero_grad()
                    loss.backward()
                    solver.step()
                    log['iterations'].append(loss.item())
                elif phase == 'test':
                    temp = outputs.detach().cpu().numpy()
                    scores = np.stack((temp[:,0], np.amax(temp[:,1:args.nClasses], axis=1)), axis=-1)
                    testPredScore.extend(scores)
                    testTrueLabel.extend((cls.detach().cpu().numpy()>0)*1)
                    imgNames.extend(imageName)

                    
                tloss += loss.item()
                c += 1

        # Logging of train and test results
        if phase == 'train':
            log['epoch'].append(tloss/c)
            log['train_acc'].append(acc/tot)
            print('Epoch: ', epoch, 'Train loss: ',tloss/c, 'Accuracy: ', acc/tot)
            train_loss.append(tloss / c)

        elif phase == 'test':
            log['validation'].append(tloss / c)
            log['val_acc'].append(acc / tot)
            print('Epoch: ', epoch, 'Test loss:', tloss / c, 'Accuracy: ', acc / tot)
            lr_sched.step(tloss / c)
            test_loss.append(tloss / c)
            accuracy = acc / tot
            if (accuracy >= bestAccuracy):
                bestAccuracy =accuracy
                testTrueLabels = testTrueLabel
                testPredScores = testPredScore
                bestEpoch = epoch
                save_best_model = os.path.join(log_path,args.method+'_best.pth')
                states = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': solver.state_dict(),
                }
                torch.save(states, save_best_model)
                testImgNames= imgNames

    with open(os.path.join(log_path,args.method+'_log.json'), 'w') as out:
        json.dump(log, out)
    torch.save(model.state_dict(), os.path.join(log_path, args.method+'_model.pt'))



# Plotting of train and test loss
plt.figure()
plt.xlabel('Epoch Count')
plt.ylabel('Loss')
plt.plot(np.arange(0, args.nEpochs), train_loss[:], color='r')
plt.plot(np.arange(0, args.nEpochs), test_loss[:], 'b')
plt.legend(('Train Loss', 'Validation Loss'), loc='upper right')
plt.savefig(os.path.join(result_path, args.method+'_Loss.jpg'))


# Evaluation of test set utilizing the trained model
obvResult = evaluation()
errorIndex, predictScore, threshold = obvResult.get_result(args.method, testImgNames, testTrueLabels, testPredScores, result_path)


