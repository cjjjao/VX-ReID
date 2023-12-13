from __future__ import print_function, absolute_import

import sys

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
"""Cross-Modality ReID"""
import pdb

# def eval_sysu(distmat, q_pids, g_pids, q_camids, g_camids, max_rank = 20,gall_img = None,que_img = None):
#     """Evaluation with sysu metric
#     Key: for each query identity, its gallery images from the same camera view are discarded. "Following the original setting in ite dataset"
#     """
#     num_q, num_g = distmat.shape
#     if num_g < max_rank:
#         max_rank = num_g
#         print("Note: number of gallery samples is quite small, got {}".format(num_g))
#     indices = np.argsort(distmat, axis=1)
#     pred_label = g_pids[indices]
#     matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
#     for i in range(0,50,5):
#         qimg = Image.fromarray(np.uint8(que_img[i]))
#         gall_img1 = gall_img[indices[i]]
#         plt.figure(figsize=(15, 5))  # 设置窗口大小
#         plt.suptitle('ReID Result')
#         plt.subplot(1, 11, 1), plt.title('query')
#         plt.imshow(qimg), plt.axis('off')
#         flag = 0
#         for k in range(10):
#             gimg = Image.fromarray(np.uint8(gall_img1[k]))
#             ax = plt.subplot(1, 11, k + 2)
#             plt.title(str(k))
#             plt.imshow(gimg)
#             ax.spines['bottom'].set_color('red')
#             ax.spines['left'].set_color('red')  ####设置左边坐标轴的粗细
#             ax.spines['right'].set_color('red')  ###设置右边坐标轴的粗细
#             ax.spines['top'].set_color('red')
#             ax.spines['bottom'].set_linewidth(2)
#             ax.spines['left'].set_linewidth(2)
#             ax.spines['right'].set_linewidth(2)
#             ax.spines['top'].set_linewidth(2)
#             ax.set_xticks([])
#             ax.set_yticks([])
#             if q_pids[i] == pred_label[i][k]:
#                 ax.spines['bottom'].set_color('green')
#                 ax.spines['left'].set_color('green')  ####设置左边坐标轴的粗细
#                 ax.spines['right'].set_color('green')  ###设置右边坐标轴的粗细
#                 ax.spines['top'].set_color('green')
#                 flag = 1
#         if flag == 1:
#             plt.savefig('./heatmap/pic' + str(i) + '.png')
#     # compute cmc curve for each query
#     new_all_cmc = []
#     all_cmc = []
#     all_AP = []
#     all_INP = []
#     num_valid_q = 0. # number of valid query
#     for q_idx in range(num_q):
#         # get query pid and camid
#         q_pid = q_pids[q_idx]
#         q_camid = q_camids[q_idx]
#
#         # remove gallery samples that have the same pid and camid with query
#         order = indices[q_idx]
#         remove = (q_camid == 3) & (g_camids[order] == 2)
#         keep = np.invert(remove)
#
#         # compute cmc curve
#         # the cmc calculation is different from standard protocol
#         # we follow the protocol of the author's released code
#         new_cmc = pred_label[q_idx][keep]
#         new_index = np.unique(new_cmc, return_index=True)[1]
#         new_cmc = [new_cmc[index] for index in sorted(new_index)]
#
#         new_match = (new_cmc == q_pid).astype(np.int32)
#         new_cmc = new_match.cumsum()
#         new_all_cmc.append(new_cmc[:max_rank])
#
#         orig_cmc = matches[q_idx][keep] # binary vector, positions with value 1 are correct matches
#         if not np.any(orig_cmc):
#             # this condition is true when query identity does not appear in gallery
#             continue
#
#         cmc = orig_cmc.cumsum()
#
#         # compute mINP
#         # refernece Deep Learning for Person Re-identification: A Survey and Outlook
#         pos_idx = np.where(orig_cmc == 1)
#         pos_max_idx = np.max(pos_idx)
#         inp = cmc[pos_max_idx]/ (pos_max_idx + 1.0)
#         all_INP.append(inp)
#
#         cmc[cmc > 1] = 1
#
#         all_cmc.append(cmc[:max_rank])
#         num_valid_q += 1.
#
#         # compute average precision
#         # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
#         num_rel = orig_cmc.sum()
#         tmp_cmc = orig_cmc.cumsum()
#         tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
#         tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
#         AP = tmp_cmc.sum() / num_rel
#         all_AP.append(AP)
#
#     assert num_valid_q > 0, "Error: all query identities do not appear in gallery"
#
#     all_cmc = np.asarray(all_cmc).astype(np.float32)
#     all_cmc = all_cmc.sum(0) / num_valid_q   # standard CMC
#
#     new_all_cmc = np.asarray(new_all_cmc).astype(np.float32)
#     new_all_cmc = new_all_cmc.sum(0) / num_valid_q
#     mAP = np.mean(all_AP)
#     mINP = np.mean(all_INP)
#     return new_all_cmc, mAP, mINP

# def eval_sysu(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=20, gall_img=None, que_img=None, query_path=None):
#     """Evaluation with sysu metric
#     Key: for each query identity, its gallery images from the same camera view are discarded. "Following the original setting in ite dataset"
#     """
#     #####-------------
#     class Example(QWidget):
#         def __init__(self):
#             super(QWidget, self).__init__()
#             self.Init_UI()
#
#         def Init_UI(self):
#             grid = QGridLayout()
#             self.setLayout(grid)
#
#             self.setGeometry(300, 300, 154, 328)
#             self.setWindowTitle('VI-ReID')
#
#             grid.setSpacing(10)
#             self.label = QLabel()
#             self.label.setScaledContents(True)
#             # label.setFixedSize(144,288)
#             self.rlabel = QLabel()
#             self.rlabel.setScaledContents(True)
#
#             grid.addWidget(self.label, 0, 0)
#             grid.addWidget(self.rlabel, 0, 1)
#
#             btn = QPushButton('选择图片')
#             btn.clicked.connect(self.openimage)
#
#             rbn = QPushButton('ReID')
#             rbn.clicked.connect(self.resultimg)
#
#             grid.addWidget(btn, 1, 0)
#             grid.addWidget(rbn, 1, 1)
#             # self.show()
#
#         def openimage(self):
#             imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")
#             jpg = QtGui.QPixmap(imgName).scaled(192, 384)
#             self.label.setPixmap(jpg)
#             imgName = ".."+ str(imgName[-38:])
#             for i, fna in enumerate(query_path):
#                 if fna == imgName:
#                     qimg = Image.fromarray(np.uint8(que_img[i]))
#                     gall_img1 = gall_img[indices[i]]
#                     plt.figure(figsize=(15, 5))  # 设置窗口大小
#                     #plt.suptitle('ReID Result')
#                     plt.subplot(1, 11, 1), plt.title('query')
#                     plt.imshow(qimg), plt.axis('off')
#                     for k in range(10):
#                         gimg = Image.fromarray(np.uint8(gall_img1[k]))
#                         ax = plt.subplot(1, 11, k + 2)
#                         plt.title(str(k))
#                         plt.imshow(gimg)
#                         ax.spines['bottom'].set_color('red')
#                         ax.spines['left'].set_color('red')  ####设置左边坐标轴的粗细
#                         ax.spines['right'].set_color('red')  ###设置右边坐标轴的粗细
#                         ax.spines['top'].set_color('red')
#                         ax.spines['bottom'].set_linewidth(2)
#                         ax.spines['left'].set_linewidth(2)
#                         ax.spines['right'].set_linewidth(2)
#                         ax.spines['top'].set_linewidth(2)
#                         ax.set_xticks([])
#                         ax.set_yticks([])
#                         if q_pids[i] == pred_label[i][k]:
#                             ax.spines['bottom'].set_color('green')
#                             ax.spines['left'].set_color('green')  ####设置左边坐标轴的粗细
#                             ax.spines['right'].set_color('green')  ###设置右边坐标轴的粗细
#                             ax.spines['top'].set_color('green')
#                     plt.savefig('./r1.png')
#                     break
#                         # plt.show()
#         def resultimg(self):
#             jpg = QtGui.QPixmap('./r1.png').scaled(1600, 300)
#             self.rlabel.setPixmap(jpg)
#     #####-------------
#     num_q, num_g = distmat.shape
#     if num_g < max_rank:
#         max_rank = num_g
#         print("Note: number of gallery samples is quite small, got {}".format(num_g))
#     indices = np.argsort(distmat, axis=1)
#     pred_label = g_pids[indices]
#     matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
#     app = QApplication(sys.argv)
#     ex = Example()
#     ex.show()
#     app.exit(app.exec_())
#     # compute cmc curve for each query
#     new_all_cmc = []
#     all_cmc = []
#     all_AP = []
#     all_INP = []
#     num_valid_q = 0.  # number of valid query
#     for q_idx in range(num_q):
#         # get query pid and camid
#         q_pid = q_pids[q_idx]
#         q_camid = q_camids[q_idx]
#
#         # remove gallery samples that have the same pid and camid with query
#         order = indices[q_idx]
#         remove = (q_camid == 3) & (g_camids[order] == 2)
#         keep = np.invert(remove)
#
#         # compute cmc curve
#         # the cmc calculation is different from standard protocol
#         # we follow the protocol of the author's released code
#         new_cmc = pred_label[q_idx][keep]
#         new_index = np.unique(new_cmc, return_index=True)[1]
#         new_cmc = [new_cmc[index] for index in sorted(new_index)]
#
#         new_match = (new_cmc == q_pid).astype(np.int32)
#         new_cmc = new_match.cumsum()
#         new_all_cmc.append(new_cmc[:max_rank])
#
#         orig_cmc = matches[q_idx][keep]  # binary vector, positions with value 1 are correct matches
#         if not np.any(orig_cmc):
#             # this condition is true when query identity does not appear in gallery
#             continue
#
#         cmc = orig_cmc.cumsum()
#
#         # compute mINP
#         # refernece Deep Learning for Person Re-identification: A Survey and Outlook
#         pos_idx = np.where(orig_cmc == 1)
#         pos_max_idx = np.max(pos_idx)
#         inp = cmc[pos_max_idx] / (pos_max_idx + 1.0)
#         all_INP.append(inp)
#
#         cmc[cmc > 1] = 1
#
#         all_cmc.append(cmc[:max_rank])
#         num_valid_q += 1.
#
#         # compute average precision
#         # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
#         num_rel = orig_cmc.sum()
#         tmp_cmc = orig_cmc.cumsum()
#         tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
#         tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
#         AP = tmp_cmc.sum() / num_rel
#         all_AP.append(AP)
#
#     assert num_valid_q > 0, "Error: all query identities do not appear in gallery"
#
#     all_cmc = np.asarray(all_cmc).astype(np.float32)
#     all_cmc = all_cmc.sum(0) / num_valid_q  # standard CMC
#
#     new_all_cmc = np.asarray(new_all_cmc).astype(np.float32)
#     new_all_cmc = new_all_cmc.sum(0) / num_valid_q
#     mAP = np.mean(all_AP)
#     mINP = np.mean(all_INP)
#     return new_all_cmc, mAP, mINP


def eval_sysu(distmat, q_pids, g_pids, q_camids, g_camids, max_rank = 20):
    """Evaluation with sysu metric
    Key: for each query identity, its gallery images from the same camera view are discarded. "Following the original setting in ite dataset"
    """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    pred_label = g_pids[indices]
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    new_all_cmc = []
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0. # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (q_camid == 3) & (g_camids[order] == 2)
        keep = np.invert(remove)

        # compute cmc curve
        # the cmc calculation is different from standard protocol
        # we follow the protocol of the author's released code
        new_cmc = pred_label[q_idx][keep]
        new_index = np.unique(new_cmc, return_index=True)[1]
        new_cmc = [new_cmc[index] for index in sorted(new_index)]

        new_match = (new_cmc == q_pid).astype(np.int32)
        new_cmc = new_match.cumsum()
        new_all_cmc.append(new_cmc[:max_rank])

        orig_cmc = matches[q_idx][keep] # binary vector, positions with value 1 are correct matches
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()

        # compute mINP
        # refernece Deep Learning for Person Re-identification: A Survey and Outlook
        pos_idx = np.where(orig_cmc == 1)
        pos_max_idx = np.max(pos_idx)
        inp = cmc[pos_max_idx]/ (pos_max_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q   # standard CMC

    new_all_cmc = np.asarray(new_all_cmc).astype(np.float32)
    new_all_cmc = new_all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)
    return new_all_cmc, mAP, mINP
def eval_regdb(distmat, q_pids, g_pids, max_rank = 20):
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0. # number of valid query
    
    # only two cameras
    q_camids = np.ones(num_q).astype(np.int32)
    g_camids = 2* np.ones(num_g).astype(np.int32)
    
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        raw_cmc = matches[q_idx][keep] # binary vector, positions with value 1 are correct matches
        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = raw_cmc.cumsum()

        # compute mINP
        # refernece Deep Learning for Person Re-identification: A Survey and Outlook
        pos_idx = np.where(raw_cmc == 1)
        pos_max_idx = np.max(pos_idx)
        inp = cmc[pos_max_idx]/ (pos_max_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)
    return all_cmc, mAP, mINP