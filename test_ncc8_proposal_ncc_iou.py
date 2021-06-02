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

        self.d_pixels = self.template.ravel()
        self.hist, self.bin_edges = np.histogram(self.d_pixels[np.nonzero(self.d_pixels)])
        self.bins = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2

        try:
            peak_idx, _ = find_peaks(self.hist, height=0)
            peak_idx = np.where(self.hist == np.max(self.hist[peak_idx]))[0]
            self.depth = self.bins[peak_idx]
        except:
            self.depth = np.median(self.template[np.nonzero(self.template)])

        self.std = np.std(self.template[np.nonzero(self.template)]) / 0.2 # np.sqrt(1)  # !!!!
        print(self.std)

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

        if math.isnan(np.max(response)):
            search_img = search_img.detach().cpu().numpy().squeeze()
            response = (search_img - np.min(search_img)) / (np.max(search_img) - np.min(search_img))

        if H < template.shape[0] or W < template.shape[1]:
            response =  cv2.resize(response, (H, W), interpolation=cv2.INTER_AREA)

        return response


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


    def generate_search_region(self, frame, scale=None, cnt=None):

        H, W = frame.shape
        x0, y0, w, h = self.pos

        if scale is None:
            scale = 8 if self.flag in ['occlusion', 'fully_occlusion', 'not_found'] else 3

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

        # if len(peaks) == 0:
        #     print('Not found peaks by NCC')
        # else:
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

        if gt_bbox is not None:
            last_center = [self.pos[0]+self.pos[2]//2, self.pos[1]+self.pos[3]//2]
            self.pos = gt_bbox
            template = frame[gt_bbox[1]:gt_bbox[1]+gt_bbox[3], gt_bbox[0]:gt_bbox[0]+gt_bbox[2]]
            self.depth, self.std = self.get_depth(template, cnt=None)

        search_img, search_region = self.generate_search_region(frame, cnt=last_center)
        response = self.get_response(search_img.copy(), self.template, self.ncc, use_gaussian=True)

        peaks = self.localize(response, n_clusters=5, n_peaks=1000, use_centroid=False, peak_size=15)
        peaks = np.append(peaks, np.asarray([[last_center[0]-search_region[0], last_center[1]-search_region[1]]]), axis=0)

        peak_depths = np.asarray([self.get_depth(search_img, cnt=[int(pk[1]), int(pk[0])]) for pk in peaks])
        depth_diff = np.asarray([abs(pd - self.depth) for pd in peak_depths])
        valid_idx = np.where(depth_diff < 500)[0]

        peaks = peaks[valid_idx]
        peak_depths = peak_depths[depth_diff < 500]

        peak_xy_dist = np.asarray([np.sqrt((pk[1]+search_region[0]-last_center[0])**2 + (pk[0]+search_region[1]-last_center[1])**2) for pk in peaks])
        valid_idx = np.where(peak_xy_dist < np.sqrt(self.template.shape[0] * self.template.shape[1]))[0]

        peaks = peaks[valid_idx]
        xy_threshold = np.sqrt(self.size[0]*self.size[1])
        peak_depths = peak_depths[peak_xy_dist < xy_threshold]
        peak_xy_dist =  peak_xy_dist[peak_xy_dist < xy_threshold]

        if len(peaks) == 0:
            peaks = np.asarray([[last_center[0]-search_region[0], last_center[1]-search_region[1]]])
            peak_depths = np.asarray([self.get_depth(search_img, cnt=[int(pk[1]), int(pk[0])]) for pk in peaks])
            peak_xy_dist = np.asarray([np.sqrt((pk[1]+search_region[0]-last_center[0])**2 + (pk[0]+search_region[1]-last_center[1])**2) for pk in peaks])

        peak_xy_dist = np.asarray([pxyd / np.sqrt(self.size[0]*self.size[1]) for pxyd in peak_xy_dist])
        ''' Proposals for each peak '''

        peak_iou = np.empty((len(peaks),), dtype=np.float32)
        peak_cnt = np.empty((len(peaks), 2), dtype=np.float32)
        peak_hist = np.empty((len(peaks), 10), dtype=np.float32)
        peak_bins = np.empty((len(peaks), 10), dtype=np.float32)
        peak_wscore = np.empty((len(peaks), ), dtype=np.float32)
        peak_response = np.empty((len(peaks), int(self.size[1]), int(self.size[0])), dtype=np.float32)

        for ii, pk in enumerate(peaks):
            cy, cx = pk
            w, h = int(self.size[0]), int(self.size[1])

            x0 = max(0, int(cx - self.size[0]//2))
            y0 = max(0, int(cy - self.size[1]//2))
            x1 = min(search_region[2], int(x0+self.size[0]))
            y1 = min(search_region[3], int(y0+self.size[1]))

            proposal = np.asarray(search_img[y0:y1, x0:x1], dtype=np.float32)

            if proposal.shape[0]>0 and proposal.shape[1]>0:
                proposal_ncc = self.get_ncc(proposal, use_hog=False, use_gaussian=True)

                proposal_response = self.get_response(self.template, proposal, proposal_ncc, use_gaussian=True)
                proposal_peaks = self.localize(proposal_response, n_clusters=1, n_peaks=100, use_centroid=False, peak_size=15)

                try:
                    cy, cx = proposal_peaks[0]
                except:
                    cy, cx = 0, 0

                del proposal_ncc

            else:
                cy, cx = 0, 0
                proposal_response = np.zeros_like(proposal)

            try:
                peak_response[ii, :y1-y0, :x1-x0] = proposal_response[:y1-y0, :x1-x0]
            except:
                pass


            dist = np.sqrt((cx - w//2)**2 + (cy - h//2)**2) / np.sqrt(w*h)
            peak_iou[ii] = dist
            peak_cnt[ii, ...] = [cx+x0, cy+y0]


            ''' histogram of depth in proposal '''
            proposal = search_img[y0:y1, x0:x1]
            proposal_pixels = proposal.ravel()
            proposal_pixels = np.nan_to_num(proposal_pixels)
            hist, bin_edges = np.histogram(proposal_pixels[np.nonzero(proposal_pixels)], bins=10)
            bins = (bin_edges[:-1] + bin_edges[1:]) / 2

            peak_hist[ii] = hist
            peak_bins[ii] = bins

            wasserstein_score = wasserstein_distance(self.hist, hist)
            # peak_wscore[ii] = wasserstein_score / np.sum(self.hist)
            peak_wscore[ii] = wasserstein_score / (self.bins[-1] - self.bins[0])
        # scores = peak_iou
        scores = np.asarray([(1-iou)*(1-wscore)*(1-pxyd) for iou, wscore, pxyd in zip(peak_iou, peak_wscore, peak_xy_dist)])



        toc = time.time()




        row = 5
        col = max(4, len(peaks))


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
            ax.scatter(optimal_cnt[0], optimal_cnt[1], c='r')

            pk = peaks[scores == opt_score][0]
            cy, cx = pk
            x0 = max(0, int(cx - self.size[0]//2))
            y0 = max(0, int(cy - self.size[1]//2))
            x1 = min(search_region[2], int(x0+self.size[0]))
            y1 = min(search_region[3], int(y0+self.size[1]))

            rect = patches.Rectangle((x0, y0), x1-x0, y1-y0, edgecolor='b', facecolor='none')
            ax.add_patch(rect)

            gt_rect = patches.Rectangle((gt_bbox[0]-search_region[0], gt_bbox[1]-search_region[1]), gt_bbox[2], gt_bbox[3], edgecolor='r', facecolor='none')
            ax.add_patch(gt_rect)
        except:
            print('No optimal score ....')


        ax.set_xticks([])
        ax.set_yticks([])


        for ii, pk in enumerate(peaks):

            cy, cx = pk
            ax2.scatter(cx, cy, c='r')

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

            ax.scatter(last_center[0]-search_region[0], last_center[1]-search_region[1], c='g')

            ax.scatter(pk[1], pk[0], c='r')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title('xy {:.2f}'.format(1-pxyd))

            ax = plt.subplot(row, col, col*2+ii+1)
            ax.imshow(response)
            ax.scatter(pk[1], pk[0], c='r')
            rect = patches.Rectangle((x0, y0), x1-x0, y1-y0, edgecolor='b', facecolor='none')
            ax.add_patch(rect)
            ax.set_xticks([])
            ax.set_yticks([])

            pr = peak_response[ii]
            p_iou = peak_iou[ii]
            try:
                cy, cx = np.where(pr == np.max(pr))
                cy, cx = cy[0], cx[0]

            except:
                cy, cx = 0, 0

            ax = plt.subplot(row, col, col*3+ii+1)
            ax.imshow(pr)
            ax.scatter(cx, cy, c='r')
            ax.set_xticks([])
            ax.set_yticks([])
            # ax.set_title('resp on template')
            ax.set_title('iou {:.2f}'.format(1-p_iou))

            hist = peak_hist[ii]
            bins = peak_bins[ii]

            ax = plt.subplot(row, col, col*4+ii+1)
            ax.plot(bins, hist, c='r')
            ax.plot(self.bins, self.hist, c='b')

            wasserstein_score = peak_wscore[ii]
            ax.set_title('wscore {:.02f}'.format(1-wasserstein_score))
            # ax.set_xticks([])
            ax.set_yticks([])

        plt.show(block=False)
        plt.pause(0.1)

        return toc-tic


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
        init_frame += 1
        init_frame = np.nan_to_num(init_frame)


        with open(root_path+'%s/groundtruth.txt'%seq, 'r') as fp:
            gt_bboxes = fp.readlines()
        gt_bboxes = [box.strip() for box in gt_bboxes]

        init_box = gt_bboxes[frame_id]
        init_box = [int(float(bb)) for bb in init_box.split(',')]

        last_bbox = init_box

        tracker = NCC_depth(init_frame, init_box, visualize=True)
        tracker = tracker.cuda()

        for frame_id in range(1, len(gt_bboxes)):


            gt_bbox = gt_bboxes[frame_id]
            if 'nan' in gt_bbox:
                # gt_bbox = [0, 0, 0, 0]
                gt_bbox = last_bbox
            else:
                gt_bbox = [int(float(bb)) for bb in gt_bbox.split(',')]
                last_bbox = gt_bbox

            frame = cv2.imread(os.path.join(data_path, '%08d.png'%(frame_id+1)), -1)
            frame += 1
            frame = np.nan_to_num(frame)
            # frame[frame > 10000] = 10000

            # try:
            t_time = tracker.track(frame, gt_bbox=gt_bbox)
            print('frame : ', frame_id, ' time : ', t_time)
            # except Exception as e:
                # print(e)
                # region = tracker.pos
