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

from scipy.stats import wasserstein_distance
from scipy.stats import ks_2samp

from sklearn.mixture import GaussianMixture
from scipy.stats import norm, multivariate_normal
from sklearn.metrics import pairwise_distances_argmin_min

import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

blue = lambda x: '\033[94m' + x + '\033[0m'


class NCC_depth(torch.nn.Module):

    def __init__(self, frame, xywh, visualize=False):
        super(NCC_depth, self).__init__()

        self.flag = 'init'

        x0, y0, w, h = xywh

        self.pos = xywh
        self.size = [w, h]
        self.template = np.asarray(frame[y0:y0+h, x0:x0+w], dtype=np.float32)
        self.hist, self.bins, self.depth = self.get_depth_by_hist(self.template)

        if not np.isscalar(self.depth):
            self.depth = self.depth[0]

        self.init_state = False
        if math.isnan(self.depth):
            self.init_state = False
        else:
            self.init_state = True

        self.std = np.std(self.template[np.nonzero(self.template)]) / 0.2 # np.sqrt(2)  # !!!!

        self.mask = self.get_mask_by_gaussian(self.template, self.depth, std=self.std)

        self.ncc = self.get_ncc(self.template, use_hog=False, use_gaussian=True)

        ''' construct a rotated template for ncc '''
        # self.rotated_template = self.rotate_image(self.template)
        # self.ncc_rotated = self.get_ncc(self.rotated_template, use_hog=False, use_gaussian=True)



    def get_ncc(self, X, use_hog=False, use_gaussian=False):

        X = np.asarray(X, dtype=np.float32)

        if use_gaussian:
            try:
                X = scipy.stats.norm(self.depth, self.std).pdf(X)
                X = (X - np.min(X))/ (np.max(X) - np.min(X))
                X = np.asarray(X*255, dtype=np.float32)
            except:
                print('use gaussian failed in get_ncc')

        if use_hog:
            _, X = hog(X, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)
            X = exposure.rescale_intensity(X, in_range=(0, 10))
            X = np.asarray(X, dtype=np.float32)

        ncc = NCC(torch.from_numpy(X[np.newaxis, ...]).cuda()).cuda()

        return ncc

    def get_mask_by_gaussian(self, image, mu, std=None):

        if std is None:
            std = self.std

        if not np.isscalar(mu):
            mu = mu[0]
        prob = scipy.stats.norm(mu, std).pdf(image)
        prob[prob < 0.7 * np.median(prob)] = 0
        prob = (prob - np.min(prob))/ (np.max(prob) - np.min(prob))
        # prob = np.asarray(prob*255, dtype=np.uint8)

        return prob


    def get_response(self, search_img, template, ncc_filter, use_hog=False, use_gaussian=True):

        search_img = np.asarray(search_img, dtype=np.float32)

        H, W = search_img.shape
        if H < template.shape[0] or W < template.shape[1]:
            search_img =  cv2.resize(search_img, template.shape, interpolation=cv2.INTER_AREA)

        if use_gaussian:
            search_img = scipy.stats.norm(self.depth, self.std).pdf(search_img)
            search_img = (search_img - np.min(search_img)) / (np.max(search_img) - np.min(search_img))
            search_img = np.asarray(search_img*255, dtype=np.float32)

        if use_hog:
            try:
                fd, search_img = hog(search_img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)
                search_img = exposure.rescale_intensity(search_img, in_range=(0, 10))
                search_img = np.asarray(search_img, dtype=np.float32)
            except:
                print('hog failed in get_response')

        search_img = torch.from_numpy(search_img[np.newaxis, np.newaxis, ...]).cuda()

        response = ncc_filter(search_img)
        response = response.detach().cpu().numpy().squeeze()
        response = (response - np.min(response)) / (np.max(response) - np.min(response))


        response = np.nan_to_num(response)
        if np.max(response) == 0:
            search_img = search_img.detach().cpu().numpy().squeeze()
            response = (search_img - np.min(search_img)) / (np.max(search_img) - np.min(search_img))

        if H < template.shape[0] or W < template.shape[1]:
            response =  cv2.resize(response, (H, W), interpolation=cv2.INTER_AREA)

        return response


    def get_depth_by_median(self, template, cnt=None):
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

        # std = np.std(template[np.nonzero(template)])

        return depth_seed # , std

    def get_depth_by_hist(self, proposal):
        ''' If multiple peaks in the histogram,
                - the last one maybe the background
                - the first one maybe the occluders
                - the middle one maybe the target
        '''

        proposal_pixels = proposal.ravel()
        proposal_pixels = np.nan_to_num(proposal_pixels)
        hist, bin_edges = np.histogram(proposal_pixels[np.nonzero(proposal_pixels)], bins=10)
        bins = (bin_edges[:-1] + bin_edges[1:]) / 2

        try:
            peak_idx, _ = find_peaks(hist, height=(np.sum(hist)/10, np.max(hist)+1))
            if len(peak_idx) > 0:
                if len(peak_idx) > 2:
                    ''' occludor, target and background '''
                    peak_idx = peakidx[1:-1]
                    peak_idx = peak_idx[len(peak_idx) // 2]

                elif len(peak_idx) == 2:
                    if abs(peak_idx[1] - peak_idx[0]) < 3:
                        ''' Partial occlusion '''
                        peak_idx = peak_idx[1]
                    else:
                        ''' background and target '''
                        peak_idx = peak_idx[0]
                else:
                    ''' only target '''
                    peak_idx = peak_idx[0]

                depth = bins[peak_idx]
            else:
                depth = self.get_depth_by_median(proposal)
        except Exception as e:
            # print(e)
            depth = self.get_depth_by_median(proposal)

        return hist, bins, depth


    def generate_search_region(self, frame, scale=None, cnt=None):

        H, W = frame.shape
        x0, y0, w, h = self.pos

        if scale is None:
            scale = 8 if self.flag in ['occlusion', 'fully_occlusion', 'not_found', 'uncertain'] else 4
            # scale = 8

        if cnt is not None:
            center_x, center_y = cnt
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


    def localize(self, response, n_clusters=5, n_peaks=1000, use_centroid=True, peak_size=15):
        ''' find the peak response in the response map of the search_image'''
        response = ndimage.maximum_filter(response, size=peak_size, mode='constant')
        peaks = feature.peak_local_max(response, min_distance=peak_size, num_peaks=n_peaks)

        if len(peaks) > 0:
            n_clusters = min(len(peaks), n_clusters)
            kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(peaks)
            if use_centroid:
                peaks = kmeans.cluster_centers_
            else:
                peaks_argmin, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, peaks)
                peaks = peaks[peaks_argmin, ...]

        return peaks

    def rotate_image(self, image):
        img_rotate_90_clockwise = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        return img_rotate_90_clockwise


    def track(self, frame, gt_bbox=None):

        tic = time.time()

        search_img, search_region = self.generate_search_region(frame, cnt=None)

        last_center = [self.pos[0]+self.pos[2]//2 - search_region[0], self.pos[1]+self.pos[3]//2 - search_region[1]]  # [X, y]

        response = self.get_response(search_img.copy(), self.template, self.ncc, use_gaussian=True)

        peaks = self.localize(response, n_clusters=5, n_peaks=1000, use_centroid=False, peak_size=15)  # [y, x]

        peaks = np.append(peaks, np.asarray([[last_center[1], last_center[0]]]), axis=0)

        ''' Proposals for each peak '''

        peak_iou = np.empty((len(peaks),), dtype=np.float32)
        peak_scale = np.empty((len(peaks),), dtype=np.float32)
        peak_cnt = np.empty((len(peaks), 2), dtype=np.float32)
        peak_hist = np.empty((len(peaks), 10), dtype=np.float32)
        peak_bins = np.empty((len(peaks), 10), dtype=np.float32)
        peak_wscore = np.empty((len(peaks), ), dtype=np.float32)
        peak_response = np.empty((len(peaks), int(self.size[1]), int(self.size[0])), dtype=np.float32)
        peak_masks = np.empty((len(peaks), int(self.size[1]), int(self.size[0])), dtype=np.uint8)
        peak_depths = np.empty((len(peaks),), dtype=np.float32)
        peak_xy_dist = np.empty((len(peaks),), dtype=np.float32)

        xy_threshold = 2*np.sqrt(self.size[0]*self.size[1])

        for ii, pk in enumerate(peaks):

            peak_xy_dist[ii] = np.sqrt((pk[1]-last_center[0])**2 + (pk[0]-last_center[1])**2) / xy_threshold

            cy, cx = pk
            w, h = int(self.size[0]), int(self.size[1])

            x0 = max(0, int(cx - self.size[0]//2))
            y0 = max(0, int(cy - self.size[1]//2))
            x1 = min(search_region[2], int(x0+self.size[0]))
            y1 = min(search_region[3], int(y0+self.size[1]))

            proposal = np.asarray(search_img[y0:y1, x0:x1], dtype=np.float32)


            ''' histogram of depth in proposal '''
            proposal = search_img[y0:y1, x0:x1]

            hist, bins, pd = self.get_depth_by_hist(proposal)

            peak_hist[ii] = hist
            peak_bins[ii] = bins
            peak_depths[ii] = pd


            '''
                - wasserstein_distance         : the higher the value, the lower the similarity
                    - the smallest distance to move
                - Kolmogorov-Smirnov statistic : the higher the pvalue, the higher the similarity

                - distance between self.hist and hist after aligning (moving) : the higher the value, the lower the similarity
            '''
            # _, p_value = ks_2samp(self.hist, hist)
            # p_value = 1 - p_value
            #
            w_score = wasserstein_distance(self.hist, hist) # how much bins moves

            move_bins = int(w_score / (bins[1] - bins[0]))
            aligned_proposal_hist = np.zeros((len(hist)+move_bins, ), dtype=np.float32)
            aligned_template_hist = np.zeros((len(hist)+move_bins, ), dtype=np.float32)
            if self.bins[0] > bins[0]:
                aligned_proposal_hist[move_bins:] = hist
                aligned_template_hist[:len(self.hist)] = self.hist
            else:
                aligned_template_hist[move_bins:] = self.hist
                aligned_proposal_hist[:len(hist)] = hist

            # hist_dist = np.linalg.norm(abs(self.hist-hist)) / np.sum(hist)
            hist_dist =  np.linalg.norm(abs(aligned_proposal_hist - aligned_template_hist)) / np.sum(hist)
            # wasserstein_score = w_score / (self.bins[-1] - self.bins[0])
            wasserstein_score = w_score / (bins[-1] - bins[0])

            peak_wscore[ii] = wasserstein_score + hist_dist

            ''' mask and size scale '''
            # proposal_mask = self.get_mask_by_gaussian(proposal, pd)
            # peak_scale[ii] = scipy.stats.norm(self.depth, self.std).pdf(pd) / scipy.stats.norm(self.depth, self.std).pdf(self.depth)

            if proposal.shape[0]>0 and proposal.shape[1]>0:
                proposal_ncc = self.get_ncc(proposal, use_hog=False, use_gaussian=True)

                proposal_response = self.get_response(self.template, proposal, proposal_ncc, use_gaussian=True)
                proposal_peaks = self.localize(proposal_response, n_clusters=1, n_peaks=100, use_centroid=False, peak_size=15)

                try:
                    cy, cx = proposal_peaks[0]
                except Exception as e:
                    # print(e)
                    cy, cx = 0, 0

                del proposal_ncc

            else:
                cy, cx = 0, 0
                proposal_response = np.zeros_like(proposal)

            try:
                peak_response[ii, :y1-y0, :x1-x0] = proposal_response[:y1-y0, :x1-x0]
                # peak_masks[ii, :y1-y0, :x1-x0] = np.asarray(proposal_mask[:y1-y0, :x1-x0]*255, dtype=np.uint8)
            except Exception as e:
                print(e)

            dist = np.sqrt((cx - w//2)**2 + (cy - h//2)**2) / np.sqrt(w*h)
            peak_iou[ii] = dist
            peak_cnt[ii, ...] = [cx+x0, cy+y0]


        ''' the weight for iou score should be lower !!!!
            for wasserstein_score should have high weight !

            the lower the weight, the higher score
        '''

        w_xy = 1
        w_iou = 1
        w_wscore = 0.5
        scores = np.asarray([max(0.01,(1- w_iou*iou))*max(0.01, (1- w_wscore*wscore))*max(0.01, (1-w_xy*pxyd)) for iou, wscore, pxyd in zip(peak_iou, peak_wscore, peak_xy_dist)])

        toc = time.time()


        row = 5
        col = max(5, len(peaks))

        plt.cla()
        plt.clf()

        ax = plt.subplot(row, col, 1)
        ax.imshow(self.template)
        ax.set_title('template')
        ax.set_xticks([])
        ax.set_yticks([])


        ax = plt.subplot(row, col, 2)
        ax.imshow(self.mask)
        ax.set_title('mask')
        ax.set_xticks([])
        ax.set_yticks([])

        ax2 = plt.subplot(row, col, 3)
        ax2.imshow(response)
        ax2.set_title('response')
        ax2.set_xticks([])
        ax2.set_yticks([])

        ax = plt.subplot(row, col, 4)
        search_img_mask = self.get_mask_by_gaussian(search_img, self.depth)
        ax.imshow(search_img_mask)
        # ax.set_title('search prob')
        ax.set_xticks([])
        ax.set_yticks([])

        try:
            opt_score = np.max(scores)
            ax.set_title('{:.2f}'.format(opt_score))
            optimal_cnt = peak_cnt[scores == opt_score][0]
            ax.scatter(optimal_cnt[0], optimal_cnt[1], c='b')

            pk = peaks[scores == opt_score][0]
            cy, cx = pk
            x0 = max(0, int(cx - self.size[0]//2))
            y0 = max(0, int(cy - self.size[1]//2))
            x1 = min(search_region[2], int(x0+self.size[0]))
            y1 = min(search_region[3], int(y0+self.size[1]))

            optimal_region = [x0, y0, x1-x0, y1-y0]

            rect = patches.Rectangle((x0, y0), x1-x0, y1-y0, linewidth=2, edgecolor='b', facecolor='none')
            ax.add_patch(rect)

            if gt_bbox is not None:
                gt_rect = patches.Rectangle((gt_bbox[0]-search_region[0], gt_bbox[1]-search_region[1]), gt_bbox[2], gt_bbox[3], linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(gt_rect)
        except Exception as e:
            print(e)
            opt_score = 0
            optimal_region = [self.pos - search_region[0], self.pos - search_region[1], self.pos[2], self.pos[3]]


        ax.set_xticks([])
        ax.set_yticks([])

        ax = plt.subplot(row, col, 5)
        ax.imshow(frame)
        if gt_bbox is not None:
            gt_rect = patches.Rectangle((gt_bbox[0], gt_bbox[1]), gt_bbox[2], gt_bbox[3], linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(gt_rect)

        region_in_frame = np.asarray(optimal_region) + np.asarray([search_region[0], search_region[1], 0, 0])

        rect = patches.Rectangle((region_in_frame[0], region_in_frame[1]), region_in_frame[2], region_in_frame[3], linewidth=2, edgecolor='b', facecolor='none')
        ax.add_patch(rect)

        # rect = patches.Rectangle((x0, y0), x1-x0, y1-y0, linewidth=2, edgecolor='b', facecolor='none')
        # ax.add_patch(rect)

        ax.set_xticks([])
        ax.set_yticks([])

        for ii, pk in enumerate(peaks):

            cy, cx = pk
            ax2.scatter(cx, cy, c='b')

            pd = peak_depths[ii]
            if not np.isscalar(pd): # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                pd = pd[0]

            pxyd = peak_xy_dist[ii]

            x0 = max(0, int(cx - self.size[0]//2))
            y0 = max(0, int(cy - self.size[1]//2))
            x1 = min(search_region[2], int(x0+self.size[0]))
            y1 = min(search_region[3], int(y0+self.size[1]))

            ax = plt.subplot(row, col, col+ii+1)
            search_img_mask = self.get_mask_by_gaussian(search_img, pd)
            ax.imshow(search_img_mask)
            rect = patches.Rectangle((x0, y0), x1-x0, y1-y0, edgecolor='b', facecolor='none')
            ax.add_patch(rect)

            ax.scatter(last_center[0], last_center[1], c='g')

            ax.scatter(pk[1], pk[0], c='b')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title('xy {:.2f}'.format(1-pxyd))

            ax = plt.subplot(row, col, col*2+ii+1)
            ax.imshow(response)
            ax.scatter(pk[1], pk[0], c='b')
            rect = patches.Rectangle((x0, y0), x1-x0, y1-y0, edgecolor='b', facecolor='none')
            ax.add_patch(rect)
            ax.set_xticks([])
            ax.set_yticks([])

            pr = peak_response[ii]
            p_iou = peak_iou[ii]
            try:
                cy, cx = np.where(pr == np.max(pr))
                cy, cx = cy[0], cx[0]

            except Exception as e:
                # print(e)
                cy, cx = 0, 0

            ax = plt.subplot(row, col, col*3+ii+1)
            ax.imshow(pr)
            ax.scatter(cx, cy, c='b')
            ax.set_xticks([])
            ax.set_yticks([])
            # ax.set_title('resp on template')
            ax.set_title('iou {:.2f}'.format(1-p_iou))

            hist = peak_hist[ii]
            bins = peak_bins[ii]

            ax = plt.subplot(row, col, col*4+ii+1)
            ax.plot(bins, hist, c='b')
            ax.plot(self.bins, self.hist, c='r')
            try:
                peak_idx, _ = find_peaks(hist, height=(np.sum(hist)/10, np.max(hist)+1))
                ax.plot(bins[peak_idx], hist[peak_idx], 'x')

                peak_idx, _ = find_peaks(self.hist, height=(np.sum(self.hist)/10, np.max(self.hist)+1))
                ax.plot(self.bins[peak_idx], self.hist[peak_idx], 'o')

                ax.plot(bins, np.ones_like(bins)*np.sum(hist)/10, c='k')
                ax.scatter(self.depth, 100, c='r')

                pd = peak_depths[ii]
                ax.scatter(pd, 100, c='b')

            except Exception as e:
                print(e)

            wasserstein_score = peak_wscore[ii]
            ax.set_title('wscore {:.02f}'.format(1-wasserstein_score))
            ax.set_xticks([])
            ax.set_yticks([])
            #
            # ax = plt.subplot(row, col, col*5+ii+1)
            # pm = peak_masks[ii]
            # ps = peak_scale[ii]
            # ax.imshow(pm)
            # ax.set_title('scale: {:.2f}'.format(ps))
            # ax.set_xticks([])
            # ax.set_yticks([])

        plt.show(block=False)
        plt.pause(0.1)

        region = [optimal_region[0]+search_region[0], optimal_region[1]+search_region[1], optimal_region[2], optimal_region[3]]
        if opt_score > 0.8:
            self.pos = region
            template = frame[self.pos[1]:self.pos[1]+self.pos[3], self.pos[0]:self.pos[0]+self.pos[2]]
            # self.depth, self.std = self.get_depth_by_median(template, cnt=None)
            self.hist, self.bins, self.depth = self.get_depth_by_hist(template)
            self.flag = 'normal'

        if opt_score < 0.4:
            self.flag = 'uncertain'
        if opt_score < 0.3:
            self.flag = 'not_found'

        return region, opt_score, toc-tic


if __name__ == '__main__':

    root_path = '/home/yan/Data2/DOT-results/CDTB-ST/sequences/'
    out_path = '/home/yan/Data2/DOT-results/CDTB-ST/results/NCC_D/'
    if not os.path.isdir(out_path):
        os.mkdir(out_path)

    if not os.path.isdir(os.path.join(out_path, 'rgbd-unsupervised')):
        os.mkdir(os.path.join(out_path, 'rgbd-unsupervised'))

    sequences = os.listdir(root_path)
    sequences.remove('list.txt')

    # seq_id = random.randint(0, len(sequences))
    seq_id = 0
    for seq in sequences[seq_id:]:

        if os.path.isfile(os.path.join(out_path, 'rgbd-unsupervised', seq, seq+'_1.txt')):
            print(seq+' already done')
        else:

            pred_bbox = []
            pred_score = []
            pred_time = []

            data_path = root_path + '%s/depth'%seq

            init_success = False

            with open(root_path+'%s/groundtruth.txt'%seq, 'r') as fp:
                gt_bboxes = fp.readlines()
            gt_bboxes = [box.strip() for box in gt_bboxes]


            # frame_id = 0

            for init_id in range(1, len(gt_bboxes)):

                tic = time.time()

                init_frame = cv2.imread(os.path.join(data_path, '%08d.png'%(init_id+1)), -1)
                # init_frame += 1
                init_frame = np.nan_to_num(init_frame)

                init_box = gt_bboxes[init_id]
                init_box = [int(float(bb)) for bb in init_box.split(',')]

                last_bbox = init_box

                tracker = NCC_depth(init_frame, init_box, visualize=True)

                toc = time.time()

                pred_bbox.append(init_box)
                pred_time.append(toc-tic)
                pred_score.append(1)

                if tracker.init_state:
                    break

            tracker = tracker.cuda()

            for frame_id in range(init_id+1, len(gt_bboxes)):

                gt_bbox = gt_bboxes[frame_id]
                if 'nan' in gt_bbox:
                    # gt_bbox = [0, 0, 0, 0]
                    gt_bbox = last_bbox
                else:
                    gt_bbox = [int(float(bb)) for bb in gt_bbox.split(',')]
                    last_bbox = gt_bbox

                frame = cv2.imread(os.path.join(data_path, '%08d.png'%(frame_id+1)), -1)
                # frame += 1
                frame = np.nan_to_num(frame)
                # frame[frame > 10000] = 10000

                # try:
                region, opt_score, t_time = tracker.track(frame, gt_bbox=gt_bbox)
                print(seq, ' frame : ', frame_id+1, '/ ', len(gt_bboxes), ' time : ', t_time)
                # except:
                #     region = tracker.pos
                #     opt_score = 0
                #     t_time = 0.2

                pred_bbox.append(region)
                pred_score.append(opt_score)
                pred_time.append(t_time)


            out_seq = os.path.join(out_path, 'rgbd-unsupervised', seq)
            if not os.path.isdir(out_seq):
                os.mkdir(out_seq)

            with open(os.path.join(out_seq, seq+'_1.txt'), 'w') as fp:
                for bb in pred_bbox:
                    fp.write('%f,%f,%f,%f\n'%(bb[0], bb[1], bb[2], bb[3]))

            with open(os.path.join(out_seq, seq+'_1_time.value'), 'w') as fp:
                for t in pred_time:
                    fp.write('%f\n'%(t))

            with open(os.path.join(out_seq, seq+'_1_confidence.value'), 'w') as fp:
                for t in pred_score:
                    fp.write('%f\n'%(t))
