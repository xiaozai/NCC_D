from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from tqdm import tqdm
from torch import nn

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import patches
from ncc.NCC import NCC

import time
import math
from math import atan2, cos, sin, sqrt, pi
import scipy
from scipy import ndimage, misc
from skimage import measure, filters, feature
from sklearn.cluster import KMeans
from scipy import signal

from sklearn.neighbors import NearestNeighbors
from scipy.optimize import minimize

from sklearn.cluster import KMeans
import scipy.stats

from scipy.signal import find_peaks

from skimage.feature import hog
from skimage import data, exposure

import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

blue = lambda x: '\033[94m' + x + '\033[0m'

class NCC_depth(torch.nn.Module):

    def __init__(self, frame, xywh, visualize=False):
        super(NCC_depth, self).__init__()

        self.angle = 5
        self.flag = 'init'
        self.visualize = visualize

        x0, y0, w, h = xywh

        self.pos = xywh
        self.size = [w, h]

        self.template = frame[y0:y0+h, x0:x0+w]


        self.d_pixels = self.template.ravel()
        self.hist, self.bin_edges = np.histogram(self.d_pixels[np.nonzero(self.d_pixels)])

        try:
            peak_idx, _ = find_peaks(self.hist, height=0)
            peak_idx = np.where(self.hist == np.max(self.hist[peak_idx]))
            peak_idx = peak_idx[0]
            bins = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
            self.depth = bins[peak_idx]
        except:
            self.depth = np.median(self.template[np.nonzero(self.template)])

        self.std = np.std(self.template[np.nonzero(self.template)])

        self.mask = self.get_mask_by_gaussian(self.template, self.depth, std=self.std)

        self.ncc, self.cnt_of_target = self.initalize_ncc(self.template)


    def initalize_ncc(self, template):
        H, W = template.shape

        template = (template - self.depth) * 1.0 / self.std
        template = np.asarray(template, dtype=np.float32)

        fd, hog_image = hog(template, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        hog_image_rescaled = np.asarray(hog_image_rescaled, dtype=np.float32)

        template = torch.from_numpy(hog_image_rescaled[np.newaxis, ...])
        template = template.cuda()

        ncc = NCC(template)
        ncc = ncc.cuda()

        ''' get target center '''
        try:
            response = ncc(torch.unsqueeze(template, 0))
            response = response.detach().cpu().numpy().squeeze()
            peak_y, peak_x = np.where(response == np.max(response))
            target_cnt = [peak_x[0], peak_y[0]]
        except:
            target_cnt = [W//2, H//2]

        return ncc, target_cnt

    def get_depth(self, template, cnt=None):
        '''
        1) Assume that the center pixel always belongs to the target
           Crop the center area and use the median depth values as the target depth

        2) Assume that the target pixels appear most in the template,
           Choose the peaks of the depth histogram of the template,
           The highest or the middle peak belong to the target
        '''
        H, W = template.shape

        if cnt is None:
            x0 = max(0, W//2 - 20)
            y0 = max(0, H//2 - 20)
            x1 = min(W, W//2+20)
            y1 = min(H, H//2+20)
        else:
            x0 = max(0, cnt[0] - 20)
            y0 = max(0, cnt[1] - 20)
            x1 = min(W, cnt[0] + 20)
            y1 = min(H, cnt[1] + 20)

        center = template[y0:y1, x0:x1]
        depth_seed = np.median(center[np.nonzero(center)])

        std = np.std(template[np.nonzero(template)])

        return depth_seed, std


    def generate_search_region(self, frame, scale=None, center=None):

        H, W = frame.shape
        x0, y0, w, h = self.pos

        if scale is None:
            scale = 8 if self.flag in ['occlusion', 'fully_occlusion', 'not_found'] else 3

        if center is not None:
            center_x, center_y = center[0], center[1]
            center_x = max(0, center_x)
            center_x = min(W, center_x)
            center_y = max(0, center_y)
            center_y = min(H, center_y)
        else:
            center_x, center_y = int(x0 + w//2), int(y0 + h//2)

        new_w, new_h = w*scale, h*scale

        new_x0 = int(max(0, center_x - new_w//2))
        new_y0 = int(max(0, center_y - new_h//2))
        new_x1 = int(min(W, center_x + new_w//2))
        new_y1 = int(min(H, center_y + new_h//2))

        if new_x0 == 0:
            new_x1 = int(min(W, new_x0+new_w))

        if new_y0 == 0:
            new_y1 = int(min(H, new_y0+new_h))

        search_region = [new_x0, new_y0, new_x1-new_x0, new_y1-new_y0]
        search_img = frame[new_y0:new_y1, new_x0:new_x1]

        return search_img, search_region

    def get_mask_by_gaussian(self, image, mu, std=None):

        if std is None:
            std = self.std
        prob = scipy.stats.norm(mu, std).pdf(image)
        # prob = prob > 0.7*np.median(prob[np.nonzero(prob)])
        prob = (prob - np.min(prob))/ (np.max(prob) - np.min(prob))

        prob = np.asarray(prob*255, dtype=np.uint8)
        return prob

    def get_response(self, search_img):

        search_img = (search_img - self.depth) * 1.0 / self.std
        search_img = np.asarray(search_img, dtype=np.float32)

        try:
            fd, hog_image = hog(search_img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)
            hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
            hog_image_rescaled = np.asarray(hog_image_rescaled, dtype=np.float32)
        except:
            hog_image_rescaled = search_img

        hog_image_rescaled = torch.from_numpy(hog_image_rescaled[np.newaxis, np.newaxis, ...])
        hog_image_rescaled = hog_image_rescaled.cuda()

        response = self.ncc(hog_image_rescaled)
        response = response.detach().cpu().numpy()
        response = np.squeeze(response)

        return response

    def localize(self, response):
        ''' find the peak response in the response map of the search_image'''
        response = ndimage.maximum_filter(response, size=7, mode='constant')
        peaks = feature.peak_local_max(response, min_distance=7, num_peaks=1000)

        return peaks

    def rotate_image(self, image, image_center, angle):
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

    def track(self, frame, gt_bbox=None):

        if gt_bbox is not None:
            last_center = [self.pos[0]+self.pos[2]//2, self.pos[1]+self.pos[3]//2]
            self.pos = gt_bbox
            template = frame[gt_bbox[1]:gt_bbox[1]+gt_bbox[3], gt_bbox[0]:gt_bbox[0]+gt_bbox[2]]
            self.depth, self.std = self.get_depth(template, cnt=None)

        search_img, search_region = self.generate_search_region(frame)
        response = self.get_response(search_img.copy())

        peaks = self.localize(response)

        if len(peaks) < 10:
            print(len(peaks))

        if len(peaks) == 0:
            peaks = np.asarray([[last_center[0]-search_region[0], last_center[1]-search_region[1]]])
        else:
            n_clusters = min(len(peaks), 5)
            kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(peaks)
            peaks = kmeans.cluster_centers_

        peak_depths = np.asarray([self.get_depth(search_img, cnt=[int(pk[1]), int(pk[0])]) for pk in peaks])
        depth_diff = np.asarray([abs(pd - self.depth) for pd in peak_depths])
        valid_idx = np.where(depth_diff < 500)[0]

        peaks = peaks[valid_idx]
        peak_depths = peak_depths[depth_diff < 500]

        xy_dist = np.asarray([np.sqrt((pk[1]+search_region[0]-last_center[0])**2 + (pk[0]+search_region[1]-last_center[1])**2) for pk in peaks])
        valid_idx = np.where(xy_dist < max(self.template.shape[0], self.template.shape[1]))[0]

        peaks = peaks[valid_idx]
        peak_depths = peak_depths[xy_dist < max(self.template.shape[0], self.template.shape[1])]
        peak_xy_dist =  xy_dist[xy_dist < max(self.template.shape[0], self.template.shape[1])]

        plt.cla()
        plt.clf()

        row = 5
        col = 8

        ax1 = plt.subplot(row, col, 1)
        ax1.imshow(self.template)
        ax1.set_xticks([])
        ax1.set_yticks([])

        # ax1.scatter(self.cnt_in_frame[0]-self.pos[0], self.cnt_in_frame[1]-self.pos[1], c='r')
        ax1.scatter(self.cnt_of_target[0], self.cnt_of_target[1], c='b')

        ax2 = plt.subplot(row, col, 2)
        ax2.imshow(self.mask)
        # ax2.scatter(self.cnt_in_frame[0]-self.pos[0], self.cnt_in_frame[1]-self.pos[1], c='r')
        ax2.scatter(self.cnt_of_target[0], self.cnt_of_target[1], c='b')
        ax2.set_title("{:.2f}".format(self.depth))

        ax2.set_xticks([])
        ax2.set_yticks([])

        ax3 = plt.subplot(row, col, 3)
        ax3.imshow(frame)
        rect = patches.Rectangle((search_region[0], search_region[1]), search_region[2], search_region[3], edgecolor='b', facecolor='none')
        ax3.add_patch(rect)
        # ax3.scatter(self.cnt_in_frame[0], self.cnt_in_frame[1], c='r')


        rect = patches.Rectangle((gt_bbox[0], gt_bbox[1]), gt_bbox[2], gt_bbox[3], edgecolor='g', facecolor='none')
        ax3.add_patch(rect)

        ax3.set_xticks([])
        ax3.set_yticks([])

        ax4 = plt.subplot(row, col, 4)
        ax4.imshow(response)
        ax4.set_xticks([])
        ax4.set_yticks([])

        ax5 = plt.subplot(row, col, 5)
        ax5.imshow(search_img)

        ax5.set_xticks([])
        ax5.set_yticks([])

        for ii, pk in enumerate(peaks):
            ax4.scatter(pk[1], pk[0], c='r')
            ax5.scatter(pk[1], pk[0], c='r')

            ax4.text(pk[1], pk[0], str(peak_depths[ii]))

        ax6 = plt.subplot(row,col,6)
        bins = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
        ax6.plot(bins, self.hist)

        hist_peaks, _ = find_peaks(self.hist, height=0)
        ax6.plot(bins[hist_peaks], self.hist[hist_peaks], "x")

        ax6.plot(bins[hist_peaks[0]], 0, "x")

        ax6.scatter(self.depth, 100, c='b')

        min_dist = 1e18
        min_bbox = None
        min_bbox2 = None
        for kk in range(0, len(peaks)):
            ax = plt.subplot(row,col,9+kk*3)
            pk = peaks[kk]
            x0 = int(max(0, pk[1] - self.pos[2]/2))
            y0 = int(max(0, pk[0] - self.pos[3]/2))
            x1 = int(min(search_region[2], pk[1] + self.pos[2]/2))
            y1 = int(min(search_region[3], pk[0] + self.pos[3]/2))

            pd = peak_depths[kk]

            temp = search_img[y0:y1, x0:x1]

            temp_prob = scipy.stats.norm(pd, self.std/np.sqrt(2)).pdf(search_img)

            # temp_prob[temp_prob < np.median(temp_prob)] = 0
            temp_prob = (temp_prob - np.min(temp_prob)) / (np.max(temp_prob) - np.min(temp_prob))
            temp_prob = np.asarray(temp_prob*255, dtype=np.uint8)

            # Apply template Matching
            h, w = self.template.shape
            _, hog_temp_prob = hog(temp_prob, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)
            hog_temp_prob = exposure.rescale_intensity(hog_temp_prob, in_range=(0, 10))
            hog_temp_prob = np.asarray(hog_temp_prob, dtype=np.float32)

            _, hog_mask = hog(self.mask, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)
            hog_mask = exposure.rescale_intensity(hog_mask, in_range=(0, 10))
            hog_mask = np.asarray(hog_mask, dtype=np.float32)

            res = cv2.matchTemplate(hog_temp_prob,hog_mask, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)

            candidate = temp_prob[top_left[1]:top_left[1]+h, top_left[0]:top_left[0]+w]
            iou = np.sum(abs(self.mask - candidate)) / np.sum(self.mask + candidate)

            cand_mask = np.zeros_like(temp_prob, dtype=np.uint8)
            cand_mask[top_left[1]:top_left[1]+h, top_left[0]:top_left[0]+w] = candidate

            ax3 = plt.subplot(row, col, 9+kk*3+2)
            ax3.imshow(cand_mask)
            ax3.set_title("{:.2f}".format(iou))
            ax3.set_xticks([])
            ax3.set_yticks([])

            ax2 = plt.subplot(row, col, 9+kk*3+1)
            ax2.imshow(temp_prob)
            ax2.scatter(pk[1], pk[0], c='r')

            rect = patches.Rectangle((top_left[0], top_left[1]),w, h, edgecolor='b', facecolor='none')
            ax2.add_patch(rect)

            rect = patches.Rectangle((x0, y0), w, h, edgecolor='r', facecolor='none')
            ax2.add_patch(rect)

            ax2.set_xticks([])
            ax2.set_yticks([])

            d_pixels = temp.ravel()
            hist, bin_edges = np.histogram(d_pixels[np.nonzero(d_pixels)])
            temp_bins = (bin_edges[:-1] + bin_edges[1:]) / 2
            ax.plot(temp_bins, hist)


            hist_peaks, _ = find_peaks(hist, height=0)
            ax.plot(temp_bins[hist_peaks], hist[hist_peaks], "x")

            ax.plot(bins, self.hist)

            ax.set_xticks([])
            ax.set_yticks([])

            from scipy.stats import wasserstein_distance
            score = wasserstein_distance(self.hist,hist)
            ax.set_title("{:.2f}".format(score))

            if score < min_dist:
                min_dist = score
                min_bbox = [x0, y0, x1-x0, y1-y0]
                min_bbox2 = [top_left[0], top_left[1], w, h]

        try:
            rect = patches.Rectangle((min_bbox[0], min_bbox[1]), min_bbox[2], min_bbox[3], edgecolor='r', facecolor='none')
            ax5.add_patch(rect)

            rect = patches.Rectangle((min_bbox2[0], min_bbox2[1]), min_bbox2[2], min_bbox2[3], edgecolor='b', facecolor='none')
            ax5.add_patch(rect)
        except:
            print('no bbox')


        plt.show(block=False)
        plt.pause(0.1)

        return 0


if __name__ == '__main__':

    root_path = '/home/yan/Data2/DOT-results/CDTB-ST/sequences/'
    sequences = os.listdir(root_path)
    sequences.remove('list.txt')

    seq_id = random.randint(0, len(sequences))
    for seq in sequences[seq_id:]:
        print(seq)

        data_path = root_path + '%s/depth'%seq


        frame_id = 0
        init_frame = cv2.imread(os.path.join(data_path, '%08d.png'%(frame_id+1)), -1)
        init_frame = np.nan_to_num(init_frame)

        with open(root_path+'%s/groundtruth.txt'%seq, 'r') as fp:
            gt_bboxes = fp.readlines()
        gt_bboxes = [box.strip() for box in gt_bboxes]

        init_box = gt_bboxes[frame_id]
        init_box = [int(float(bb)) for bb in init_box.split(',')]

        tracker = NCC_depth(init_frame, init_box, visualize=True)
        tracker = tracker.cuda()

        print('after init')

        for frame_id in range(1, len(gt_bboxes)):
            print('frame : ', frame_id)

            gt_bbox = gt_bboxes[frame_id]
            if 'nan' in gt_bbox:
                gt_bbox = [0, 0, 0, 0]
            else:
                gt_bbox = [int(float(bb)) for bb in gt_bbox.split(',')]

            frame = cv2.imread(os.path.join(data_path, '%08d.png'%(frame_id+1)), -1)
            frame = np.nan_to_num(frame)
            # frame[frame > 10000] = 10000

            # try:
            region = tracker.track(frame, gt_bbox=gt_bbox)
            # except Exception as e:
                # print(e)
                # region = tracker.pos
