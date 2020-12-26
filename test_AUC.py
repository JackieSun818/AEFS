import numpy as np
import matplotlib.pylab
from sklearn.metrics import auc, roc_curve, precision_recall_curve
from my_function import *
from tqdm import tqdm

fold = 0
# n = 1307
m = 1996

y_true = np.loadtxt('dataset/test_DPI.txt').tolist()
y_pre = np.loadtxt('result/y_pre_DPI.txt').tolist()
# y_pre = np.loadtxt('result/y_pre_DPI.txt').tolist()
idx = []
auc_list = []
aupr_list = []
tpr_list = []
fpr_list = []
recall_list = []
precision_list = []
c = 0
for i in tqdm(range(len(y_true))):
    if np.sum(np.array(y_true[i])) == 0:
        c += 1
        continue
    else:
        tpr1, fpr1, precision1, recall1 = tpr_fpr_precision_recall(np.array(y_true[i]), np.array(y_pre[i]))
        fpr_list.append(fpr1)
        tpr_list.append(tpr1)
        precision_list.append(precision1)
        recall_list.append(recall1)
        auc_list.append(auc(fpr1, tpr1))
        aupr_list.append(auc(recall1, precision1)+recall1[0]*precision1[0])

coverage = []
for i in tpr_list:
    try:
        coverage.append(i.index(1.0)+1)
    except:
        print('1')
print(np.mean(np.array(coverage)))

tpr = equal_len_list(tpr_list)
fpr = equal_len_list(fpr_list)
precision = equal_len_list(precision_list)
recall = equal_len_list(recall_list)
tpr_mean = np.mean(tpr, axis=0)
fpr_mean = np.mean(fpr, axis=0)
recall_mean = np.mean(recall, axis=0)
precision_mean = np.mean(precision, axis=0)
print('The auc of prediction is:', auc(fpr_mean, tpr_mean))
print('The aupr of prediction is:', auc(recall_mean, precision_mean)+recall_mean[0]*precision_mean[0])
matplotlib.pylab.plt.figure(2)
matplotlib.pylab.plt.plot(fpr_mean, tpr_mean, 'r')
matplotlib.pylab.plt.plot(recall_mean, precision_mean, 'b')
matplotlib.pylab.plt.show()

np.savetxt('result/AEFS_fpr.txt', fpr_mean)
np.savetxt('result/AEFS_tpr.txt', tpr_mean)
np.savetxt('result/AEFS_recall.txt', recall_mean)
np.savetxt('result/AEFS_p.txt', precision_mean)
np.savetxt('result/AEFS_AUC_list.txt', auc_list)
np.savetxt('result/AEFS_AUPR_list.txt', aupr_list)