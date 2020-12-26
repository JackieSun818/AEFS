import numpy as np
# from pylab import *
from sklearn.metrics import auc, roc_curve, precision_recall_curve
from my_function import *
from tqdm import tqdm

fold = 0
# n = 1307
m = 1996

y_true = np.loadtxt("k_fold/fold_1_test_DPI.txt").tolist()
# y_pre = np.loadtxt('result/1996潜在靶标_MACCS_GELU_双损失_EPOCH'+str(m*9)+'.txt').tolist()
y_pre = np.loadtxt('result/fold1_pred.txt').tolist()

auc_list = []
aupr_list = []
c = 0
for i in tqdm(range(len(y_true))):
    if np.sum(np.array(y_true[i])) == 0:
        # auc_list.append(1)
        # aupr_list.append(1)
        c += 1
        continue
    else:
        tpr1, fpr1, precision1, recall1 = tpr_fpr_precision_recall(np.array(y_true[i]), np.array(y_pre[i]))
        auc_list.append(auc(fpr1, tpr1))
        aupr_list.append(auc(recall1, precision1)+recall1[0]*precision1[0])
print('AUC : ', np.mean(np.array(auc_list)))
print('AUPR : ', np.mean(np.array(aupr_list)))
print('全0行：', c)

# label = null_list(n)
# score = null_list(n)
#
# for i in range(index.shape[0]):
#     for j in range(index.shape[1]):
#         ele_index = index[i, j]
#         r = int(ele_index / m)
#         c = int(ele_index % m)
#         label[r].append(A[r, c])
#         score[r].append(predict[i, j])
#
# auc_list = []
# aupr_list = []
# tpr = []
# fpr = []
# precision = []
# recall = []
#
# for i in range(n):
#     if 1 not in label[i]:
#         auc_list.append('0')
#         aupr_list.append('0')
#         continue
#
#     # fpr1, tpr1, thresholds = roc_curve(np.array(label[i]), np.array(score[i]), drop_intermediate=False)
#     # recall1, precision1, thresholds = precision_recall_curve(np.array(label[i]), np.array(score[i]))
#     tpr1, fpr1, precision1, recall1 = tpr_fpr_precision_recall(np.array(label[i]), np.array(score[i]))
#     tpr.append(tpr1)
#     fpr.append(fpr1)
#     precision.append(precision1)
#     recall.append(recall1)
#     auc_list.append(auc(fpr1, tpr1))
#     aupr_list.append(auc(recall1, precision1))
#
# tpr = np.array(equal_len_list(tpr))
# fpr = np.array(equal_len_list(fpr))
# recall = np.array(equal_len_list(recall))
# precision = np.array(equal_len_list(precision))
#
# tpr_mean = np.mean(tpr, axis=0)
# fpr_mean = np.mean(fpr, axis=0)
# recall_mean = np.mean(recall, axis=0)
# precision_mean = np.mean(precision, axis=0)
# # np.savetxt('represent_learning/predict_result_of_GBDT/tpr_of_gbdt.txt', tpr_mean)
# # np.savetxt('represent_learning/predict_result_of_GBDT/fpr_of_gbdt.txt', fpr_mean)
# # np.savetxt('represent_learning/predict_result_of_GBDT/recall_of_gbdt.txt', recall_mean)
# # np.savetxt('represent_learning/predict_result_of_GBDT/precision_of_gbdt.txt', precision_mean)
# # np.savetxt('represent_learning/predict_result_of_GBDT/auc_list_of_ARGAGBDT.txt', np.array(auc_list))
# # np.savetxt('represent_learning/predict_result_of_GBDT/aupr_list_of_ARGAGBDT.txt', np.array(aupr_list))
# np.savetxt('predict_result_of_ARGA_withoutGAN/tpr_of_gbdt.txt', tpr_mean)
# np.savetxt('predict_result_of_ARGA_withoutGAN/fpr_of_gbdt.txt', fpr_mean)
# np.savetxt('predict_result_of_ARGA_withoutGAN/recall_of_gbdt.txt', recall_mean)
# np.savetxt('predict_result_of_ARGA_withoutGAN/precision_of_gbdt.txt', precision_mean)
# np.savetxt('predict_result_of_ARGA_withoutGAN/auc_list_of_ARGAGBDT.txt', np.array(auc_list))
# np.savetxt('predict_result_of_ARGA_withoutGAN/aupr_list_of_ARGAGBDT.txt', np.array(aupr_list))
#
# predict0 = np.loadtxt('predict_result_of_ARGA_withoutGAN/ARGA0.txt')
#
# print('The auc of prediction is:', auc(fpr_mean, tpr_mean))
# print('The aupr of prediction is:', auc(recall_mean, precision_mean)+recall_mean[0]*precision_mean[0])
# plt.figure(2)
# plt.plot(fpr_mean, tpr_mean, 'r')
# plt.plot(recall_mean, precision_mean, 'b')
# plt.show()
