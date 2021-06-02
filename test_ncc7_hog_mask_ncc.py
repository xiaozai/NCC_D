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

        self.template =  frame[y0:y0+h, x0:x0+w]

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

        self.mask = self.get_mask_by_gaussian(self.template, self.depth, std=self.std/np.sqrt(2))

        self.ncc_template = self.get_ncc(self.template, type='template')


    def get_ncc(self, img, type='template'):

        if type == 'template':
            img = scipy.stats.norm(self.depth, self.std/np.sqrt(2)).pdf(img)
            img = np.asarray(img*255, dtype=np.float32)

        _, hog_image = hog(img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)
        hog_image = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        hog_image = np.asarray(hog_image, dtype=np.float32)

        self.hog_image = hog_image

        hog_image = torch.from_numpy(hog_image[np.newaxis, ...])
        hog_image = hog_image.cuda()

        ncc = NCC(hog_image).cuda()

        return ncc

    def get_mask_by_gaussian(self, image, mu, std=None):

        if std is None:
            std = self.std
        prob = scipy.stats.norm(mu, std).pdf(image)
        prob = (prob - np.min(prob))/ (np.max(prob) - np.min(prob))
        prob = np.asarray(prob*255, dtype=np.uint8)

        return prob

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

    def get_response(self, search_img):

        H, W = search_img.shape
        if H < self.template.shape[0] or W < self.template.shape[1]:
            search_img =  cv2.resize(search_img, self.template.shape, interpolation=cv2.INTER_AREA)

        search_img = scipy.stats.norm(self.depth, self.std/np.sqrt(2)).pdf(search_img)
        search_img = (search_img - np.min(search_img)) / (np.max(search_img) - np.min(search_img))
        search_img = np.asarray(search_img*255, dtype=np.float32)

        try:
            fd, hog_image = hog(search_img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)
            hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
            hog_image_rescaled = np.asarray(hog_image_rescaled, dtype=np.float32)
        except:
            hog_image_rescaled = search_img

        hog_image_rescaled = torch.from_numpy(hog_image_rescaled[np.newaxis, np.newaxis, ...])
        hog_image_rescaled = hog_image_rescaled.cuda()

        response = self.ncc_template(hog_image_rescaled)
        response = response.detach().cpu().numpy()
        response = np.squeeze(response)

        if H < self.template.shape[0] or W < self.template.shape[1]:
            response =  cv2.resize(response, (H, W), interpolation=cv2.INTER_AREA)


        return response, hog_image_rescaled.detach().cpu().numpy().squeeze()

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
        response, hog_search_img = self.get_response(search_img.copy())
        response = (response - np.min(response)) / (np.max(response) - np.min(response))

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


        ''' histogram of template depth '''
        template_hist_peaks, _ = find_peaks(self.hist, height=0)
        template_bins = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2

        ''' Perform NCC on the mask '''
        _, hog_template_mask = hog(self.mask, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)
        hog_template_mask = exposure.rescale_intensity(hog_template_mask, in_range=(0, 10))
        hog_template_mask = np.asarray(hog_template_mask, dtype=np.float32)


        plt.cla()
        plt.clf()

        row = 7
        col = 8

        ax1 = plt.subplot(row, col, 1)
        ax1.imshow(self.template)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_title('template')

        ax2 = plt.subplot(row, col, 2)
        ax2.imshow(self.mask)
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_title('mask')

        ax3 = plt.subplot(row, col, 3)
        ax3.imshow(self.hog_image)
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax3.set_title('hog template')

        ax4 = plt.subplot(row, col, 4)
        ax4.imshow(hog_template_mask)
        ax4.set_xticks([])
        ax4.set_yticks([])
        ax4.set_title('hog mask')

        ax5 = plt.subplot(row, col, 5)
        ax5.plot(template_bins, self.hist)
        ax5.plot(template_bins[template_hist_peaks], self.hist[template_hist_peaks], 'x')
        ax5.set_xticks([])
        ax5.set_yticks([])
        ax5.set_title('template depth hist')

        ax6 = plt.subplot(row, col, 6)
        ax6.imshow(search_img)
        ax6.set_xticks([])
        ax6.set_yticks([])
        ax6.set_title('search image')

        ax7 = plt.subplot(row, col, 7)
        ax7.imshow(response)
        ax7.set_xticks([])
        ax7.set_yticks([])
        ax7.set_title('ncc response')
        for pk in peaks:
            ax7.scatter(pk[1], pk[0], c='r')


        ax8 = plt.subplot(row, col, 8)
        ax8.imshow(hog_search_img)
        ax7.set_xticks([])
        ax7.set_yticks([])
        ax8.set_title('hog search_img')
        
        min_score = 1e18
        eps = 1e-18
        optimal_bbox = None
        # for kk in range(0, len(peaks)):
        for kk in range(0, min(row-1, len(peaks))):
            ''' Proposals by peaks

                - peak_proposal_bbox
                - peak_proposal           -> depth image
                - peak_proposal_hist
                - peak_proposal_hist_peaks
                - peak_proposal_score
                # - res_peak_proposals   -> no nesscessary
                - hog_peak_proposal
            '''
            pk = peaks[kk]
            x0 = int(max(0, pk[1] - self.pos[2]/2))
            y0 = int(max(0, pk[0] - self.pos[3]/2))
            x1 = int(min(search_region[2], pk[1] + self.pos[2]/2))
            y1 = int(min(search_region[3], pk[0] + self.pos[3]/2))

            peak_response_score = response[int(pk[0]), int(pk[1])]
            peak_proposal_bbox = [x0, y0, x1-x0, y1-y0]
            peak_proposal = search_img[y0:y1, x0:x1]

            hog_peak_proposal = scipy.stats.norm(self.depth, self.std/np.sqrt(2)).pdf(peak_proposal)
            hog_peak_proposal = (hog_peak_proposal - np.min(hog_peak_proposal)) / (np.max(hog_peak_proposal) - np.min(hog_peak_proposal))
            hog_peak_proposal = np.asarray(hog_peak_proposal*255, dtype=np.float32)
            _, hog_peak_proposal = hog(hog_peak_proposal, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)
            hog_peak_proposal = exposure.rescale_intensity(hog_peak_proposal, in_range=(0, 10))
            hog_peak_proposal = np.asarray(hog_peak_proposal, dtype=np.float32)
            # res_peak_proposals = cv2.matchTemplate(hog_peak_proposal, hog_mask, cv2.TM_CCOEFF_NORMED)


            ''' Hist of depth in peak proposal '''
            peak_proposal_pixels = peak_proposal.ravel()
            peak_proposal_hist, peak_proposal_edges = np.histogram(peak_proposal_pixels[np.nonzero(peak_proposal_pixels)])
            peak_proposal_bins = (peak_proposal_edges[:-1] + peak_proposal_edges[1:]) / 2
            peak_proposal_hist_peaks, _ = find_peaks(peak_proposal_hist, height=0)
            peak_proposal_wasserstein_score = wasserstein_distance(self.hist, peak_proposal_hist)


            ''' Proposals by the masks '''
            pd = peak_depths[kk]
            # scale = scipy.stats.norm(self.depth, self.std/np.sqrt(2)).pdf(pd)
            # print('scale : ', scale, self.depth, pd)

            peak_mask = scipy.stats.norm(pd, self.std/np.sqrt(2)).pdf(search_img)
            peak_mask = (peak_mask - np.min(peak_mask)) / (np.max(peak_mask) - np.min(peak_mask))
            peak_mask = np.asarray(peak_mask*255, dtype=np.uint8)

            ''' Find a new position based on the mask '''
            h, w = self.template.shape
            _, hog_proposal_mask = hog(peak_mask, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)
            hog_proposal_mask = exposure.rescale_intensity(hog_proposal_mask, in_range=(0, 10))
            hog_proposal_mask = np.asarray(hog_proposal_mask, dtype=np.float32)

            res_proposal_mask = cv2.matchTemplate(hog_proposal_mask, hog_template_mask, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res_proposal_mask)
            mask_proposal = search_img[max_loc[1]:max_loc[1]+h, max_loc[0]:max_loc[0]+w]

            mask_proposal_pixels = mask_proposal.ravel()
            mask_proposal_hist, mask_proposal_bin_edges = np.histogram(mask_proposal_pixels[np.nonzero(mask_proposal_pixels)])
            mask_proposal_bins = (mask_proposal_bin_edges[:-1] + mask_proposal_bin_edges[1:]) / 2
            mask_proposal_hist_peaks, _ = find_peaks(mask_proposal_hist, height=0)
            mask_proposal_wasserstein_score = wasserstein_distance(self.hist, mask_proposal_hist)
            mask_response_score = response[int(max_loc[1]+h//2), int(max_loc[0]+w//2)]

            peak_proposal_score = peak_proposal_wasserstein_score / (peak_response_score + eps)
            mask_proposal_score = mask_proposal_wasserstein_score / (mask_response_score + eps)

            if peak_proposal_score < min_score and peak_proposal_score < mask_proposal_score:
                min_score = peak_proposal_score
                optimal_bbox = peak_proposal_bbox
            elif mask_proposal_score < min_score and mask_proposal_score < peak_proposal_score:
                min_score = mask_proposal_score
                optimal_bbox = [max_loc[0], max_loc[1], w, h]



            ax = plt.subplot(row, col, col+kk*col+1)
            ax.imshow(peak_proposal)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(str(peak_proposal_score))


            ax = plt.subplot(row, col, col+kk*col+2)
            ax.imshow(hog_peak_proposal)
            ax.set_title(str(peak_response_score))
            ax.set_xticks([])
            ax.set_yticks([])

            ax = plt.subplot(row, col, col+kk*col+3)
            ax.plot(peak_proposal_bins, peak_proposal_hist, c='r')
            ax.plot(peak_proposal_bins[peak_proposal_hist_peaks], peak_proposal_hist[peak_proposal_hist_peaks], 'o')
            ax.plot(template_bins, self.hist, c='g')
            ax.plot(template_bins[template_hist_peaks], self.hist[template_hist_peaks], 'x')
            ax.set_title(str(peak_proposal_wasserstein_score))
            ax.set_xticks([])
            ax.set_yticks([])

            ax = plt.subplot(row, col, col+kk*col+4)
            ax.imshow(peak_mask)
            rect = patches.Rectangle((max_loc[0], max_loc[1]), w, h, edgecolor='b', facecolor='none')
            ax.add_patch(rect)
            ax.scatter(max_loc[0], max_loc[1], c='r')
            ax.scatter(max_loc[0]+w//2, max_loc[1]+h//2, c='b')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(str(mask_proposal_score))

            ax = plt.subplot(row, col, col+kk*col+5)
            ax.imshow(hog_proposal_mask)
            ax.scatter(max_loc[0], max_loc[1], c='r')
            ax.scatter(max_loc[0]+w//2, max_loc[1]+h//2, c='b')
            ax.set_xticks([])
            ax.set_yticks([])

            ax = plt.subplot(row, col, col+kk*col+6)
            ax.imshow(res_proposal_mask)
            ax.set_title(str(mask_response_score))
            ax.set_xticks([])
            ax.set_yticks([])

            ax = plt.subplot(row, col, col+kk*col+7)
            ax.imshow(mask_proposal)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title('mask proposal')

            ax = plt.subplot(row, col, col+kk*col+8)
            ax.plot(mask_proposal_bins, mask_proposal_hist, c='r')
            ax.plot(mask_proposal_bins[mask_proposal_hist_peaks], mask_proposal_hist[mask_proposal_hist_peaks], 'o')
            ax.plot(template_bins, self.hist, c='g')
            ax.plot(template_bins[template_hist_peaks], self.hist[template_hist_peaks], 'x')
            ax.set_title(str(mask_proposal_wasserstein_score))
            ax.set_xticks([])
            ax.set_yticks([])

        try:
            # rect = patches.Rectangle((min_bbox[0], min_bbox[1]), min_bbox[2], min_bbox[3], edgecolor='r', facecolor='none')
            # ax5.add_patch(rect)

            rect = patches.Rectangle((optimal_bbox[0], optimal_bbox[1]), optimal_bbox[2], optimal_bbox[3], edgecolor='b', facecolor='none')
            ax6.add_patch(rect)

            ax6.set_title(str(min_score))
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
        # init_frame += 10
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
            # frame += 10
            frame = np.nan_to_num(frame)
            # frame[frame > 10000] = 10000

            # try:
            region = tracker.track(frame, gt_bbox=gt_bbox)
            # except Exception as e:
                # print(e)
                # region = tracker.pos
