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
from sklearn.cluster import KMeans
import scipy
from scipy import ndimage, misc
from skimage import measure
from skimage import filters
from skimage import feature

blue = lambda x: '\033[94m' + x + '\033[0m'

class NCC_depth(torch.nn.Module):
    """docstring for ."""

    def __init__(self, template, pos):
        super(NCC_depth, self).__init__()

        self.H = template.shape[0]
        self.W = template.shape[1]
        self.pos = pos               # [x, y, w, h] in image

        self.template = template     # H, W
        self.std = np.std(template[np.nonzero(template)])
        print(self.std)

        self.flag = 'normal'

        self.depth = max(1, np.median(template))

        self.initialize(template.copy())

        self.template = (self.template - self.depth) * 1.0 / self.depth
        self.template = np.asarray(self.template, dtype=np.float32)
        self.template = torch.from_numpy(self.template[np.newaxis, ...])
        self.template = self.template.cuda()

        self.ncc = NCC(self.template)
        self.ncc = self.ncc.cuda()

    def initialize(self, template):
        prob = template
        prob[prob > self.depth+self.std//2] = 0
        prob[prob < self.depth-self.std//2] = 0
        prob = (prob - self.depth) / (self.std * 2/2)
        prob = ndimage.median_filter(prob, size=3)

        l = 256
        n = 12
        sigma = l / (4. * n)
        prob = filters.gaussian(prob, sigma=sigma)
        mask = prob > 0.7 * prob.mean()
        mask = np.asarray(mask, dtype=np.uint8)
        self.mask = mask


    def generate_search_region(self, frame, scale=4, center=None):

        H, W = frame.shape
        x0, y0, w, h = self.pos

        if center is not None:
            center_x, center_y = center[0], center[1]
            center_x = max(0, center_x)
            center_x = min(W, center_x)
            center_y = max(0, center_y)
            center_y = min(H, center_y)
        else:
            center_x, center_y = int(x0 + self.W//2), int(y0 + self.H//2)

        new_w, new_h = w*scale, h*scale

        new_x0 = int(max(0, center_x - new_w//2))
        new_y0 = int(max(0, center_y - new_h//2))
        new_x1 = int(min(W, center_x + new_w//2))
        new_y1 = int(min(H, center_y + new_h//2))

        if new_x0 == 0:
            new_x1 = int(min(W, new_x0+new_w))

        if new_y0 == 0:
            new_y1 = int(min(H, new_y0+new_w))

        search_region = [new_x0, new_y0, new_x1-new_x0, new_y1-new_y0]
        search_img = frame[new_y0:new_y1, new_x0:new_x1]

        search_img = (search_img - self.depth) * 1.0 / (self.depth+0.001)
        search_img = np.asarray(search_img, dtype=np.float32)
        search_img = torch.from_numpy(search_img[np.newaxis, np.newaxis, ...])

        return search_img, search_region

    def track(self, frame):

        if self.flag in ['occlusion', 'fully_occlusion', 'not_found']:
            search_img, search_region = self.generate_search_region(frame, scale=4)
        elif self.flag == 'uncertain':
            search_img, search_region = self.generate_search_region(frame, scale=2)
        else:
            search_img, search_region = self.generate_search_region(frame, scale=1.6)

        search_img = search_img.cuda()

        response = self.ncc(search_img)
        response = response.detach().cpu().numpy()
        response = np.squeeze(response)

        search_img = search_img.detach().cpu().numpy()
        search_img = np.squeeze(search_img)
        search_img = search_img * self.depth + self.depth

        k_candidates_box, k_candidates_im, k_candidates_response, k_candidates_center, peaks, peak_depth, peak_response, peaks_in_frame, search_img, search_region, response = self.localize(frame, search_img, search_region, response)

        return k_candidates_box, k_candidates_im, k_candidates_response, k_candidates_center, peaks, peak_depth, peak_response, peaks_in_frame, search_img, search_region, response

    def localize(self, frame, search_img, search_region, response):

        H, W = frame.shape
        search_x0, search_y0, _, _ = search_region

        ''' 1) find the peak response in the response map of the search_image'''

        response = ndimage.maximum_filter(response, size=10, mode='constant')
        peaks = feature.peak_local_max(response, min_distance=10)

        center_x, center_y, peak_depth, peak_response = [], [], [], []

        for p_xy in peaks:
            center_x.append(p_xy[1])
            center_y.append(p_xy[0])
            peak_depth.append(search_img[p_xy[0], p_xy[1]])
            peak_response.append(response[p_xy[0], p_xy[1]])

        peaks = np.c_[center_x, center_y]
        peak_depth = np.asarray(peak_depth)
        peak_response = np.asarray(peak_response)


        ''' 2) find the candidate boxes in the frame '''
        cand_H = int(self.H*1.01)
        cand_W = int(self.W*1.01)

        peaks_in_frame = np.asarray([[p_xy[0]+search_region[0], p_xy[1]+search_region[1]] for p_xy in peaks])
        # candidate_boxes_in_frame = np.zeros((len(peaks_in_frame), 4), dtype=np.int32)
        candidate_boxes_in_frame = []
        for ii, p_xy in enumerate(peaks_in_frame):
            x0 = max(0, p_xy[0]-cand_W//2)
            y0 = max(0, p_xy[1]-cand_H//2)
            x1 = min(W, p_xy[0]+cand_W//2)
            y1 = min(H, p_xy[1]+cand_H//2)

            ''' remove the outliers '''
            xy_diff = np.linalg.norm(np.asarray([p_xy[0] - self.pos[0], p_xy[1]-self.pos[1]]))
            d_diff = max(frame[p_xy[1], p_xy[0]], self.depth) / (min(frame[p_xy[1], p_xy[0]], self.depth) + 0.0001)

            if xy_diff < 1.5*np.sqrt(self.H*self.W) and d_diff < 1.5:
                candidate_boxes_in_frame.append([x0, y0, x1-x0, y1-y0])

        candidate_boxes_in_frame = np.asarray(candidate_boxes_in_frame, dtype=np.int32)



        ''' 3) get the response map for each candidate box '''

        candidates = np.zeros((len(candidate_boxes_in_frame), cand_H, cand_W), dtype=np.float32)
        candidates_response =  np.zeros((len(candidate_boxes_in_frame), cand_H, cand_W), dtype=np.float32)
        max_candidate_response = []

        for ii, box in enumerate(candidate_boxes_in_frame):
            cand = np.zeros((cand_H, cand_W), dtype=np.float32)

            cand_im = frame[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
            cand[:box[3], :box[2]] = cand_im
            cand = (cand - self.depth) * 1.0 / (self.depth+0.001)
            cand = torch.from_numpy(cand[np.newaxis, np.newaxis, ...])
            cand = cand.cuda()
            cand_response = self.ncc(cand)
            cand_response = cand_response.detach().cpu().numpy().squeeze()
            cand = cand.detach().cpu().numpy().squeeze()
            cand = cand * self.depth + self.depth

            candidates[ii] = cand[:cand_H, :cand_W]
            candidates_response[ii] = cand_response[:cand_H, :cand_W]
            max_candidate_response.append(np.max(cand_response[:cand_H, :cand_W]))

        ''' 4) find the top K maximum response position '''
        K = 5
        if len(candidates) < K:
            k_candidates_im = candidates
            k_candidates_response = candidates_response
            k_candidates_box = candidate_boxes_in_frame
            k_candidates_center = peaks_in_frame
        else:
            k_candidates_im = np.zeros((K, cand_H, cand_W), dtype=np.float32)
            k_candidates_response = np.zeros((K, cand_H, cand_W), dtype=np.float32)
            k_candidates_box = np.zeros((K, 4), dtype=np.int32)
            k_candidates_center = np.zeros((K, 2), dtype=np.int32)

            topK_response = max_candidate_response
            topK_response.sort(reverse=True)
            topK_response = topK_response[:K]

            for k, k_resp in enumerate(topK_response):
                k_idx = np.where(max_candidate_response == k_resp)[0]

                k_candidates_im[k] = candidates[k_idx]
                k_candidates_response[k] = candidates_response[k_idx]
                k_candidates_box[k] = candidate_boxes_in_frame[k_idx]
                k_candidates_center[k] = peaks_in_frame[k_idx]

        return k_candidates_box, k_candidates_im, k_candidates_response, k_candidates_center, peaks, peak_depth, peak_response, peaks_in_frame, search_img, search_region, response

    def refine_box(self,frame, candidates_box, candidate_im, candidates_response, candidate_center):
        ''' all coordinates are in the frame

        '''

        return 0


if __name__ == '__main__':

    # seq = 'humans_shirts_room_occ_1_A' # !!!!!!
    # seq = 'backpack_blue'   3 !!!!!
    # seq = 'bicycle_outside' # !!!!
    # seq = 'XMG_outside'
    # seq = 'backpack_robotarm_lab_occ'
    # seq = 'bottle_box' # !!!!?
    # seq = 'box_darkroom_noocc_10' # !!!!!!!!!!!!!!!!!!!!!!!!! multi depth seed
    # seq = 'box_darkroom_noocc_9'
    seq = 'box_darkroom_noocc_8'
    data_path = '/home/sgn/Data1/yan/Datasets/CDTB/%s/depth/'%seq

    frame_id = 0

    init_frame = cv2.imread(os.path.join(data_path, '%08d.png'%(frame_id+1)), -1)

    with open('/home/sgn/Data1/yan/Datasets/CDTB/%s/groundtruth.txt'%seq, 'r') as fp:
        gt_bboxes = fp.readlines()
    gt_bboxes = [box.strip() for box in gt_bboxes]


    init_box = gt_bboxes[frame_id]
    init_box = [int(float(bb)) for bb in init_box.split(',')]

    template = init_frame[init_box[1]:init_box[1]+init_box[3], init_box[0]:init_box[0]+init_box[2]]

    tracker = NCC_depth(template, init_box)
    tracker = tracker.cuda()

    for frame_id in range(1, len(gt_bboxes)):

        tic_track = time.time()
        frame = cv2.imread(os.path.join(data_path, '%08d.png'%(frame_id+1)), -1)
        k_candidates_box, k_candidates_im, k_candidates_response, k_candidates_center, peaks, peak_depth, peak_response, peaks_in_frame, search_img, search_region, response = tracker.track(frame)

        toc_track = time.time()
        print('Track a frame time : ', toc_track - tic_track)



        plt.clf()

        M = 11
        N = 5

        ax1 = plt.subplot(M,N,1)

        template = np.squeeze(template)
        ax1.imshow(template)
        ax1.set_title('template')

        ax5 = plt.subplot(M,N,2)
        ax5.imshow(tracker.mask)
        ax5.set_title('tracker.mask')

        ax4 = plt.subplot(M,N,3)
        ax4.imshow(search_img)
        for p_xy in peaks:
            ax4.scatter(p_xy[0], p_xy[1], linewidth=1, c='r')
        ax4.set_title('search_img')


        ax2 = plt.subplot(M,N,4)
        ax2.imshow(frame)

        # for p_xy in peaks_in_frame:
        #     ax2.scatter(p_xy[0], p_xy[1], linewidth=1, c='r')


        for box in k_candidates_box:
            x0, y0, w, h = box
            rect = patches.Rectangle((x0, y0), w, h, linewidth=1, edgecolor='r', facecolor='none')
            ax2.add_patch(rect)

        x0, y0, w, h = search_region
        rect = patches.Rectangle((x0, y0), w, h, linewidth=1, edgecolor='g', facecolor='none')
        ax2.add_patch(rect)
        ax2.set_title('frame')


        ax3 = plt.subplot(M,N,5)
        ax3.imshow(np.exp(response))


        for ii, p_xy in enumerate(peaks):
            ax3.scatter(p_xy[0], p_xy[1], linewidth=1, c='r')

            # ax3.text(p_xy[0]-1, p_xy[1]-1, str(peak_response[ii]))

        for ii, p_xy in enumerate(k_candidates_center):
            ax3.scatter(p_xy[0]-search_region[0], p_xy[1]-search_region[1], linewidth=1, c='g')
            # ax3.text(p_xy[0]-search_region[0]+3, p_xy[1]-search_region[1]+3, str(np.max(k_candidates_response[ii, ...])), color='white')

        ax3.set_title('Max response : ' + str(np.max(response)))



        for ii in range(0, len(k_candidates_im)):

            ax11 = plt.subplot(M, N, ii*N+6)
            ax11.imshow(frame)
            box = k_candidates_box[ii, ...]
            x0, y0, w, h = box
            rect = patches.Rectangle((x0, y0), w, h, linewidth=1, edgecolor='r', facecolor='none')
            ax11.add_patch(rect)

            ax22 = plt.subplot(M, N, ii*N+6+1)
            ax22.imshow(k_candidates_im[ii, ...])


            ax33 = plt.subplot(M,N,ii*N+6+2)
            ax33.imshow(k_candidates_response[ii, ...])

            max_resp = np.max(k_candidates_response[ii, ...])
            max_resp_xy = np.where(k_candidates_response[ii, ...] == max_resp)
            my, mx = max_resp_xy
            my, mx = my[0], mx[0]

            depth = k_candidates_im[ii, my, mx]
            ax22.scatter(mx, my, linewidth=2, c='r')
            ax22.text(mx+1, my+1, str(depth), color='white')

            ax33.set_title('Max response : ' + str(max_resp))
            ax33.scatter(mx, my, linewidth=2, c='r')
            ax33.text(mx+1, my+1, str(max_resp), color='white')

            print(k_candidates_box[ii, ...], max_resp)

        plt.show(block=False)
        plt.pause(3)
