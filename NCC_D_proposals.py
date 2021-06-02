#!/usr/bin/python

import vot
import sys
import time
import cv2
import collections

import torch
import numpy as np
from ncc.NCC import NCC
import math

import os
import matplotlib.pyplot as plt
from matplotlib import patches

import math
from math import atan2, cos, sin, sqrt, pi
import scipy
from scipy import ndimage, misc
from skimage import measure, filters, feature
from sklearn.cluster import KMeans


from sklearn.neighbors import NearestNeighbors
from scipy.optimize import minimize

from skimage.feature import hog

import torchvision.models as models

import torch
import torch.nn as nn
# from torchvision.models.utils import load_state_dict_from_url
from torch.utils.model_zoo import load_url as load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Dict, Optional, cast
from torch import Tensor
from collections import OrderedDict
from torchvision.models.resnet import *
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.models.resnet import model_urls

class IntResNet(ResNet):
    def __init__(self,output_layer,*args):
        self.output_layer = output_layer
        super().__init__(*args)

        self._layers = []
        for l in list(self._modules.keys()):
            self._layers.append(l)
            if l == output_layer:
                break
        self.layers = OrderedDict(zip(self._layers,[getattr(self,l) for l in self._layers]))

    def _forward_impl(self, x):
        for l in self._layers:
            x = self.layers[l](x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

def new_resnet(arch: str,
               outlayer: str,
                block: Type[Union[BasicBlock, Bottleneck]],
                layers: List[int],
                pretrained: bool,
                progress: bool,
                **kwargs: Any ) -> IntResNet:

    '''model_urls = {
        'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
        'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
        'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
        'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
        'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
        'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
        'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
        'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
        'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
    }'''

    model = IntResNet(outlayer, block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

class NCCTracker(torch.nn.Module):

    def __init__(self, image, region):
        super(NCCTracker, self).__init__()

        self.window = max(region[2], region[3]) * 2

        left = max(region[0], 0)
        top = max(region[1], 0)

        right = min(region[0] + region[2], image.shape[1] - 1)
        bottom = min(region[1] + region[3], image.shape[0] - 1)

        self.template = np.asarray(image[int(top):int(bottom), int(left):int(right)], dtype=np.float32)
        self.position = (region[0] + region[2] / 2, region[1] + region[3] / 2)
        self.size = (region[2], region[3])

        self.depth = np.median(self.template[np.nonzero(self.template)])
        self.std = np.std(self.template[np.nonzero(self.template)]) + 0.0000000001


        self.init_gaussian_model(self.template)


        template = self.template
        template = (template - self.depth) / self.std
        template = np.asarray(template, dtype=np.float32)
        template = torch.from_numpy(template[np.newaxis, ...])
        template = template.cuda()

        self.ncc = NCC(template)
        self.ncc = self.ncc.cuda()


    def init_gaussian_model(self, template):
        H, W = template.shape
        cnt = [W//2, H//2]
        x0 = max(0, cnt[0]-15)
        y0 = max(0, cnt[1]-15)
        x1 = min(W, cnt[0]+15)
        y1 = min(H, cnt[1]+15)

        temp = template[y0:y1, x0:x1]

        depth = np.median(temp[np.nonzero(temp)])
        std = np.std(temp[np.nonzero(temp)])
        prob = scipy.stats.norm(depth, std).pdf(template)

        mask = prob > 0.7 * np.median(prob)


        self.mask = mask

        valid_idx = np.nonzero(mask)
        Y, X = valid_idx
        D = template[np.nonzero(mask)]

        xyd = np.c_[X.ravel(), Y.ravel(), D.ravel()]

        mu = np.mean(xyd, axis=0)
        std = np.cov(xyd.T)

        self.gaussian_model = scipy.stats.multivariate_normal(mean=mu, cov=std)

        # XX, YY = np.meshgrid(np.arange(0, W), np.arange(0, H))
        # DD = template.ravel()
        # all_xyd = np.c_[XX.ravel(), YY.ravel(), DD.ravel()]
        #
        # new_prob = self.gaussian_model.pdf(all_xyd)
        # new_img = np.reshape(new_prob, (H,W))
        #
        #
        # ax = plt.subplot(1,3,1)
        # ax.imshow(template)
        #
        # ax = plt.subplot(1,3,2)
        # ax.imshow(prob)
        #
        # ax = plt.subplot(1,3,3)
        # ax.imshow(new_img)
        # plt.show()

        # return 0

    def track(self, image):

        H, W = image.shape

        left = max(round(self.position[0] - float(self.window) / 2), 0)
        top = max(round(self.position[1] - float(self.window) / 2), 0)

        right = min(round(self.position[0] + float(self.window) / 2), image.shape[1] - 1)
        bottom = min(round(self.position[1] + float(self.window) / 2), image.shape[0] - 1)

        if right - left < self.template.shape[1] or bottom - top < self.template.shape[0]:
            return vot.Rectangle(self.position[0] + self.size[0] / 2, self.position[1] + self.size[1] / 2, self.size[0], self.size[1])

        cut = image[int(top):int(bottom), int(left):int(right)]

        # matches = cv2.matchTemplate(cut, self.template, cv2.TM_CCOEFF_NORMED)
        # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matches)
        search_img = np.asarray(cut, dtype=np.float32)
        search_img = (search_img - self.depth) / self.std
        search_img = torch.from_numpy(search_img[np.newaxis, np.newaxis, ...])
        search_img = search_img.cuda()

        response = self.ncc(search_img)
        response = response.detach().cpu().numpy()
        response = np.squeeze(response)

        ''' Peaks '''
        # response = ndimage.maximum_filter(response, size=15, mode='constant')
        peaks = feature.peak_local_max(response, min_distance=15, num_peaks=20)

        R = 1
        C = 4

        plt.cla()
        plt.clf()
        ax1 = plt.subplot(R, C, 1)
        ax1.imshow(image)
        rect = patches.Rectangle((int(left), int(top)), int(right) - int(left), int(bottom)-int(top), edgecolor='b', facecolor='none')
        ax1.add_patch(rect)


        ax2 = plt.subplot(R, C, 2)
        ax2.imshow(cut)

        ax3 = plt.subplot(R, C, 3)
        ax3.imshow(response)

        ax4 = plt.subplot(R, C,4)
        ax4.imshow(tracker.template)


        max_score = 0
        pred_bbox = None

        model = new_resnet('resnet50','conv1',Bottleneck, [3, 4, 6, 3],True,True)
        model = model.to('cuda:0')


        # input_template = (self.template - self.depth) / self.std * 255
        # input_template = torch.from_numpy(input_template[np.newaxis, np.newaxis, ...])
        # input_template = input_template.repeat(1, 3, 1, 1)
        # input_template = input_template.cuda()
        #
        # out_template = model(input_template)
        # out_template = out_template.detach().cpu().numpy().squeeze()

        for ii, pk in enumerate(peaks):

            '''  What peaks can provide ??

            '''

            ax3.scatter(pk[1], pk[0])
            ax2.scatter(pk[1], pk[0])

            pk_in_frame = [pk[0]+top, pk[1]+left]

            proposal_xywh = [pk_in_frame[1] - self.size[0]//2, pk_in_frame[0] - self.size[1]//2, self.size[0], self.size[1]]

            proposal_img = np.zeros((self.size[1], self.size[0]), dtype=np.float32)

            x0 = max(0, pk_in_frame[1] - self.size[0]//2)
            y0 = max(0, pk_in_frame[0] - self.size[1]//2)
            x1 = min(W, pk_in_frame[1] + self.size[0]//2)
            y1 = min(H, pk_in_frame[0] + self.size[1]//2)

            x0_s = int(x0 - (pk_in_frame[1] - self.size[0]//2))
            y0_s = int(y0 - (pk_in_frame[0] - self.size[1]//2))

            ww = int(x1 - x0)
            hh = int(y1 - y0)

            proposal_img[y0_s:y0_s+hh, x0_s:x0_s+ww] = image[y0:y1, x0:x1]

            xx, yy = np.meshgrid(np.arange(self.size[0]), np.arange(self.size[1]))
            dd = proposal_img.ravel()
            xyd = np.c_[xx.ravel(), yy.ravel(), dd.ravel()]
            prob = self.gaussian_model.pdf(xyd)
            prob = np.reshape(prob, (self.size[1], self.size[0]))

            valid_pixels = prob > 0.7 * np.median(prob)
            score = np.sum(np.logical_and(valid_pixels, self.mask)) / (np.sum(np.logical_or(valid_pixels, self.mask)) + 0.00001)
            #
            # xyd = np.asarray([pk_in_frame[1], pk_in_frame[0], image[pk_in_frame[0], pk_in_frame[1]]])
            # score = self.gaussian_model.pdf(xyd)

            # fig = plt.figure(0)
            # plt.cla()
            # plt.clf()
            # ax = fig.add_subplot(1, 4, 1)
            # ax.imshow(proposal_img)
            #
            # ax = fig.add_subplot(1,4,2)
            # ax.imshow(valid_pixels)
            # ax.set_title(str(score))
            #
            # ax = fig.add_subplot(1,4,3)
            # ax.imshow(self.template)
            #
            #
            # ax = fig.add_subplot(1,4,4)
            # ax.imshow(image)
            # rect = patches.Rectangle((int(proposal_xywh[0]), int(proposal_xywh[1])), int(proposal_xywh[2]), int(proposal_xywh[3]),  edgecolor='r', facecolor='none')
            # ax.add_patch(rect)
            #
            # plt.show(block=False)
            # plt.pause(0.1)





            # from scipy import signal
            # score = signal.correlate2d(self.template, proposal_img)

            # Resnet
            #
            # input_x = (proposal_img - self.depth) / self.std * 255
            # input_x = torch.from_numpy(input_x[np.newaxis, np.newaxis, ...])
            # input_x = input_x.repeat(1, 3, 1, 1)
            # input_x = input_x.cuda()
            # out = model(input_x)



            # out = out.detach().cpu().numpy().squeeze()


            # print(proposal_img.shape, out.shape)

            # R=8
            # C=8
            #
            # for ii in range(1, 65):
            #     ax = plt.subplot(R, C, ii)
            #     ax.imshow(out[ii-1, :, :])
            # plt.show()
            #
            #
            # proposal_values = (proposal_img.ravel() - self.depth) / self.std
            # proposal_hist, bin_edges = np.histogram(proposal_values, bins=10)

            # ax1 = plt.subplot(1,2,1)
            # ax1.hist(proposal_values, bins=10)
            #
            #

            # template_values = (self.template.ravel() - self.depth) / self.std
            # template_hist, bin_edges = np.histogram(template_values, bins=10)


            # cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
            # score = cos(out, out_template)
            # score = score.detach().cpu().numpy().squeeze()

            # ax = plt.subplot(1,1,1)
            # ax.imshow(score)
            # plt.show()
            # score = np.mean(score)
            # print(score.shape)

            #
            # ax2 = plt.subplot(1,2,2)
            # ax2.hist(template_values, bins=20)


            # from scipy.stats import ks_2samp
            # score = ks_2samp(proposal_hist, template_hist)
            # score = score.pvalue

            # plt.show()



            # norm = np.linalg.norm(score)
            # score = score/norm


            # ax = plt.subplot(1,3,1)
            # ax.imshow(score)
            # ax.set_title(str(np.max(score)))
            # ax = plt.subplot(1,3,2)
            # ax.imshow(proposal_img)
            #
            # ax = plt.subplot(1,3,3)
            # ax.imshow(self.template)
            #
            # plt.show()

            # print(ii, 'correlated score : ', score)

            ''' Choose the proposal '''

            # # from skimage.feature import hog
            # hog_image = hog(proposal_img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=False)
            # # print(hog_image.shape)
            #
            # score = np.dot(self.hog_image, hog_image) / (np.linalg.norm(self.hog_image) * np.linalg.norm(hog_image))
            # print(score)

            # ax = plt.subplot(2,2,1)
            # ax.imshow(self.template)
            #
            # ax2 = plt.subplot(2,2,2)
            # ax2.imshow(self.hog_image)
            #
            # ax3 = plt.subplot(2,2,3)
            # ax3.imshow(proposal_img)
            #
            # ax4 = plt.subplot(2,2,4)
            # ax4.imshow(hog_image)
            #
            # plt.show()

            # proposal = np.asarray(proposal_img, dtype=np.float32)
            # proposal = (proposal - self.depth) / self.std
            # proposal = torch.from_numpy(proposal[np.newaxis, np.newaxis, ...])
            # proposal = proposal.cuda()
            #
            # score = self.ncc(proposal)
            # score = score.detach().cpu().numpy().squeeze()
            # score = np.max(score)

            if score > max_score:
                max_score = score
                pred_bbox = proposal_xywh

                # if score > 0.6:


            ori_score = response[pk[0], pk[1]]

            # print(ii, 'correlated score : ', score, ori_score)

            # ax = plt.subplot(R, C, 5+ii)
            # ax.imshow(proposal_img)
            # ax.scatter(self.size[0]//2, self.size[1]//2, c='r')
            # ax.set_title('proposal '+str(score) + '   vs   '+str(ori_score))

        if max_score > 0.6:
            self.depth = np.median(proposal_img[np.nonzero(proposal_img)])
            self.std = np.std(proposal_img[np.nonzero(proposal_img)])

            x0 = max(0, pred_bbox[0])
            x0 = min(W, pred_bbox[0])
            y0 = max(0, pred_bbox[1])
            y0 = min(H, pred_bbox[1])
            x1 = min(W, pred_bbox[0]+pred_bbox[2])
            y1 = min(H, pred_bbox[1]+pred_bbox[3])

            print(self.position)
            self.position = [int(x0), int(y0), int(x1-x0), int(y1-y0)]

            print(self.position)
            # rect = patches.Rectangle((int(proposal_xywh[0]), int(proposal_xywh[1])), int(proposal_xywh[2]), int(proposal_xywh[3]),  edgecolor='r', facecolor='none')
            # ax1.add_patch(rect)

        print('score : ', max_score)

        rect = patches.Rectangle((int(pred_bbox[0]), int(pred_bbox[1])), int(pred_bbox[2]), int(pred_bbox[3]),  edgecolor='r', facecolor='none')
        ax1.add_patch(rect)

        ax1.set_title(str(max_score))

        plt.show(block=False)
        # plt.show()
        plt.pause(0.1)

        pred_conf = ori_score

        return pred_bbox, pred_conf

if __name__ == '__main__':


    root_path = '/home/yan/Data2/DOT-results/CDTB-ST/sequences/'

    sequences = os.listdir(root_path)
    sequences.remove('list.txt')
    # sequences = ['backpack_blue']

    for seq in sequences[2:]:
        print(seq)

        data_path = root_path + '%s/depth'%seq

        frame_id = 0
        init_frame = cv2.imread(os.path.join(data_path, '%08d.png'%(frame_id+1)), -1)

        with open(root_path+'%s/groundtruth.txt'%seq, 'r') as fp:
            gt_bboxes = fp.readlines()
        gt_bboxes = [box.strip() for box in gt_bboxes]

        init_box = gt_bboxes[frame_id]
        init_box = [int(float(bb)) for bb in init_box.split(',')]

        tracker = NCCTracker(init_frame, init_box)
        tracker = tracker.cuda()

        for frame_id in range(1, len(gt_bboxes)):

            tic_track = time.time()
            frame = cv2.imread(os.path.join(data_path, '%08d.png'%(frame_id+1)), -1)
            pred_bbox, pred_conf = tracker.track(frame)

            toc_track = time.time()
            print('Track a frame time : ', toc_track - tic_track)
