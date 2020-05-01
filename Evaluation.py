import pickle
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
import pylab as pl
import csv
import os


class evaluation:
    def __init__(self):
        return None

    def plot_confusion_matrix(self,cm,method,path, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center", fontsize=30,
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label',fontsize=20)
        plt.xlabel('Predicted label',fontsize=20)
        plt.tight_layout()
        plt.savefig(os.path.join(path,method +'_ConfMatrix.jpg'))

    def get_threshold(self, fprs, thresholds, fpr):
    # Getting threshold for particular fpr
       threshold = 0
       for x in range(0, fprs.size):
         if fprs[x] >= fpr:
             break
         threshold = thresholds[x]
       return threshold

    def get_result(self, method,imgNames, true_label,predict_score,path, minThreshold = -1):

        # Getting predicted scores
        predict = np.array(predict_score)
        if(len(predict.shape)==2):
            predict =  predict[:, 1]
        elif(len(predict.shape)==3):
            predict = predict[:, :, 1]

        # Normalization of scores in [0,1]
        predictScore = (predict-min(predict))/ (max(predict) - min(predict))
        print('Max Score:'+ str(max(predict)))
        print('Min Score:'+ str(min(predict)))

        # Saving image or video name with match score
        if imgNames != 'None':
            imgNameScore=[]
            for i in range(len(imgNames)):
                imgNameScore.append([imgNames[i], true_label[i], predictScore[i]])
            with open(os.path.join(path, method + '_Match_Scores.csv'), 'w', newline='') as fout:
                writer = csv.writer(fout)
                writer.writerows(imgNameScore)

        # Histogram plot
        live = []
        [live.append(predictScore[i]) for i in range(len(true_label)) if (true_label[i] == 0)]
        spoof = []
        [spoof.append(predictScore[j]) for j in range(len(true_label)) if (true_label[j] == 1)]
        bins = np.linspace(np.min(np.array(spoof + live)), np.max(np.array(spoof + live)), 60)
        plt.figure()
        plt.hist(live, bins, alpha=0.5, label='Bonafide', density=True, edgecolor='black', facecolor='g')
        plt.hist(spoof, bins, alpha=0.5, label='PA', density=True, edgecolor='black',facecolor='r' )
        plt.legend(loc='upper right', fontsize=15)
        plt.xlabel('Scores')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(path, method +"_Histogram.jpg"))

        # Plot ROC curves in semilog scale
        (fprs, tprs, thresholds) = roc_curve(true_label, predictScore)
        plt.figure()
        plt.semilogx(fprs, tprs, label=method)
        plt.grid(True, which="major")
        plt.legend(loc='lower right', fontsize=15)
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.xticks([0.001, 0.01, 0.1, 1])
        plt.xlabel('False Detection Rate')
        plt.ylabel('True Detection Rate')
        plt.xlim((0.0005, 1.01))
        plt.ylim((0, 1.02))
        plt.plot([0.002, 0.002], [0, 1], color='#A0A0A0', linestyle='dashed')
        plt.plot([0.001, 0.001], [0, 1], color='#A0A0A0', linestyle='dashed')
        plt.plot([0.01, 0.01], [0, 1], color='#A0A0A0', linestyle='dashed')
        plt.savefig(os.path.join(path,method +"_ROC.jpg"))

        #Plot Raw ROC curves
        plt.figure()
        plt.plot(fprs, tprs)
        plt.grid(True, which="major")
        plt.legend(method, loc='lower right', fontsize=15)
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.xticks([0.01, 0.1, 1])
        plt.xlabel('False Detection Rate')
        plt.ylabel('True Detection Rate')
        plt.xlim((0.0005, 1.01))
        plt.ylim((0, 1.02))
        plt.savefig(os.path.join(path,method +"_RawROC.jpg"))

        # Calculation of TDR at 0.2% , 0.1% and  5% FDR
        with open(os.path.join(path , method +'_TDR-ACER.csv'), mode='w+') as fout:
            fprArray = [0.002,0.001, 0.01, 0.05]
            for fpr in fprArray:
                tpr = np.interp(fpr, fprs, tprs)
                threshold = self.get_threshold(fprs, thresholds, fpr)
                fout.write("TDR @ FDR, threshold: %f @ %f ,%f\n" % (tpr, fpr, threshold))
                print("TDR @ FDR, threshold: %f @ %f ,%f " % (tpr, fpr, threshold))

        # Calculation of APCER, BPCER and ACER
            if minThreshold == -1:
                minACER= 1000
                for thresh in pl.frange(0,1,0.025):
                    APCER = np.count_nonzero(np.less(spoof,thresh))/len(spoof)
                    BPCER = np.count_nonzero(np.greater_equal(live,thresh))/len(live)
                    ACER = (APCER + BPCER)/2
                    if ACER < minACER:
                        minThreshold = thresh
                        minAPCER = APCER
                        minBPCER = BPCER
                        minACER = ACER
                fout.write("APCER and BPCER @ ACER, threshold: %f and %f @ %f, %f\n" % (minAPCER, minBPCER, minACER, minThreshold))
                print("APCER and BPCER @ ACER, threshold: %f and %f @ %f, %f\n" % (minAPCER, minBPCER, minACER, minThreshold))
            else:
                APCER = np.count_nonzero(np.less(spoof, minThreshold)) / len(spoof)
                BPCER = np.count_nonzero(np.greater_equal(live, minThreshold)) / len(live)
                ACER = (APCER + BPCER) / 2
                fout.write("APCER and BPCER @ ACER, threshold: %f and %f @ %f, %f\n" % (APCER, BPCER, ACER, minThreshold))
                print("APCER and BPCER @ ACER, threshold: %f and %f @ %f, %f\n" % (APCER, BPCER, ACER, minThreshold))



        # Calculation of Confusion matrix
        #threshold = self.get_threshold(fprs, thresholds, 0.002)
        predict = predictScore >= minThreshold
        predict_label =[]
        [predict_label.append(int(predict[i])) for i in range(len(predict))]
        conf_matrix = confusion_matrix(true_label, predict_label)   # 0 for live and 1 for spoof
        print(conf_matrix)

        # Plot non-normalized confusion matrix
        np.set_printoptions(precision=2)
        class_names = ['0', '1']
        self.plot_confusion_matrix(conf_matrix, method,path, classes=class_names,normalize=False)

        # Saving evaluation measures
        pickle.dump((fprs,tprs,minThreshold,tpr,fpr,conf_matrix), open(os.path.join(path,method +".pickle"), "wb"))
        errorIndex=[]
        [errorIndex.append(i) for i in range(len(true_label)) if true_label[i] != predict_label[i]]
        return errorIndex, predictScore, minThreshold


if __name__ == '__main__':
    true_label= [0,0,0,0,0,1,1,1,1,1]
    predict_score = [[0.8,0.20], [0.44, 0.56],[0.6,0.4],[0.7, 0.3], [0.9,0.1],[0.2,0.8],[0.3,0.7],[0.4,0.6],[0.01,0.99],[0.1, 0.9]]
    obvResult = evaluation()
    predict_score = np.array(predict_score)
    predict_score = predict_score.reshape((10,1, 2))
    result = obvResult.get_result('Test', 'None', true_label, predict_score, "../TempData/results")

