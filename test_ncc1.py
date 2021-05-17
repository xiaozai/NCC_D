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

        self.pos = pos               # [x, y, w, h] in image
        self.flag = 'init'
        self.std = np.std(template[np.nonzero(template)])
        self.depth = max(1, np.median(template[np.nonzero(template)]))
        self.mask = self.get_mask(template, self.depth)
        self.ncc = self.initalize_ncc(template.copy())
        self.template = template

    def initalize_ncc(self, template):
        template = (template - self.depth) * 1.0 / self.depth
        template = np.asarray(template, dtype=np.float32)
        template = torch.from_numpy(template[np.newaxis, ...])
        template = template.cuda()

        ncc = NCC(template)
        ncc = ncc.cuda()

        return ncc

    def generate_search_region(self, frame, scale=None, center=None):

        H, W = frame.shape
        x0, y0, w, h = self.pos

        if scale is None:
            scale = 4 if self.flag in ['occlusion', 'fully_occlusion', 'not_found'] else 2

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
            new_y1 = int(min(H, new_y0+new_w))

        search_region = [new_x0, new_y0, new_x1-new_x0, new_y1-new_y0]
        search_img = frame[new_y0:new_y1, new_x0:new_x1]

        return search_img, search_region

    def get_response(self, search_img):

        search_img = (search_img - self.depth) * 1.0 / (self.depth+0.001)
        search_img = np.asarray(search_img, dtype=np.float32)
        search_img = torch.from_numpy(search_img[np.newaxis, np.newaxis, ...])
        search_img = search_img.cuda()

        response = self.ncc(search_img)
        response = response.detach().cpu().numpy()
        response = np.squeeze(response)

        search_img = search_img.detach().cpu().numpy()
        search_img = np.squeeze(search_img)
        search_img = search_img * (self.depth+0.001) + self.depth

        return response, search_img

    def track(self, frame):

        search_img, search_region = self.generate_search_region(frame)
        response, search_img = self.get_response(search_img)
        peaks, peak_depths, peak_scales, peak_responses, peak_masks, distance, matching_res, matching_box, new_masks = self.localize(frame, search_img, search_region, response)

        return peaks, peak_depths, peak_responses, peak_masks, search_img, response, distance, matching_res, matching_box, new_masks

    def localize(self, frame, search_img, search_region, response):

        H, W = frame.shape
        search_x0, search_y0, search_w, search_h = search_region
        last_center = [self.pos[0]+self.pos[2]//2, self.pos[1]+self.pos[3]//2]
        xy_threshold = np.sqrt(self.pos[2]*self.pos[3]) + 0.000001

        ''' 1) find the peak response in the response map of the search_image'''
        response = ndimage.maximum_filter(response, size=15, mode='constant')
        peaks = feature.peak_local_max(response, min_distance=15)

        center_x, center_y, peak_depths, peak_scales, peak_responses, peak_masks, distance = [], [], [], [], [], [], []

        for p_xy in peaks:
            py, px = p_xy

            px_in_frame, py_in_frame = px+search_x0, py+search_y0
            xy_dist = np.sqrt((px_in_frame - last_center[0])**2 + (py_in_frame - last_center[1])**2)

            temp_area = search_img[max(0, py-20):min(search_h, py+20), max(0, px-20):min(search_w, px+20)]
            pd = np.median(temp_area[np.nonzero(temp_area)])
            d_diff = abs(pd - self.depth) / self.depth

            if xy_dist < xy_threshold and d_diff < 0.5:

                center_x.append(px)
                center_y.append(py)
                peak_depths.append(pd)
                peak_responses.append(response[py, px])
                peak_masks.append(self.get_mask(search_img, pd))
                peak_scales.append(self.depth / (pd +0.000001))

                xy_diff = xy_dist / xy_threshold
                xyd_diff = xy_diff * d_diff
                distance.append(xyd_diff)

        peaks = np.c_[center_x, center_y]
        peak_depths = np.asarray(peak_depths)
        peak_responses = np.asarray(peak_responses)
        peak_masks = np.asarray(peak_masks)
        peak_scales = np.asarray(peak_scales)

        ''' Minimum xy*d '''
        distance = np.asarray(distance)
        sorted_dist = distance.copy()
        sorted_dist.sort()
        min_distance = sorted_dist[:3]
        min_idx = [np.where(distance == md)[0][0] for md in min_distance]

        peaks = peaks[min_idx]
        peak_depths = peak_depths[min_idx]
        peak_responses = peak_responses[min_idx]
        peak_masks = peak_masks[min_idx]
        peak_scales = peak_scales[min_idx]

        matching_res, matching_box, new_masks = self.mask_matching(search_img, peak_masks, peak_scales)

        return peaks, peak_depths, peak_scales, peak_responses, peak_masks, distance, matching_res, matching_box, new_masks

    def mask_matching(self, search_img, masks, scales):

        ''' Template matching
            If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum otherwise maximum
        '''
        template = self.mask
        template = np.asarray(template, dtype=np.uint8)
        w, h = template.shape[::-1]
        results = []
        boxes = []
        new_boxes = []
        centers = []
        aligned_masks = []
        new_masks = []
        new_imgs = []
        kpt_template = []
        kpt_search = []
        kpt_matches = []

        for pm, s in zip(masks, scales):
            pm = np.asarray(pm, dtype=np.uint8)

            ''' scaling based on the depth '''
            scaled_template = cv2.resize(template, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)

            ''' find the center location based on the Template Matching '''
            res = cv2.matchTemplate(pm, scaled_template, cv2.TM_SQDIFF) # W-w+, H-h+1
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            box = [min_loc[0], min_loc[1], int(w*s), int(h*s)]
            cnt = [min_loc[0]+int(w*s/2), min_loc[1]+int(h*s/2)]

            ''' Find the correct mask '''
            # new_pm = np.zeros_like(pm)
            # new_pm[box[1]:box[1]+box[3], box[0]:box[0]+box[2]] = scaled_template
            # template_hist = cv2.calcHist([scaled_template], [0], None, [256], [0, 256])
            # template_hist = cv2.normalize(template_hist, template_hist).flatten()
            #
            # pm_hist = cv2.calcHist([pm], [0], None, [256], [0, 256])
            # pm_hist = cv2.normalize(pm_hist, pm_hist).flatten()
            #
            # d = cv2.compareHist(template_hist, pm_hist, cv2.HISTCMP_CORREL)

            ''' Find the pixels belong to target, e.g. the closest ones ???? '''
            new_im = search_img.copy()
            new_im = np.multiply(new_im, pm)



            ''' Find the best rotation that -> min err = || R*Temp - T0 || '''
            #
            # def get_gradient(im) :
            #     # Calculate the x and y gradients using Sobel operator
            #     grad_x = cv2.Sobel(im,cv2.CV_32F,1,0,ksize=3)
            #     grad_y = cv2.Sobel(im,cv2.CV_32F,0,1,ksize=3)
            #     # Combine the two gradients
            #     grad = cv2.addWeighted(np.absolute(grad_x), 0.5, np.absolute(grad_y), 0.5, 0)
            #     return grad
            #
            #
            # new_pm = np.zeros_like(pm, dtype=np.uint8)
            # new_pm[box[1]:box[1]+box[3], box[0]:box[0]+box[2]] = scaled_template
            #
            #
            # warp_mode = cv2.MOTION_TRANSLATION
            # warp_matrix = np.eye(2, 3, dtype=np.float32)
            # number_of_iterations = 50000
            # termination_eps = 1e-10
            # criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
            # (cc, warp_matrix) = cv2.findTransformECC (get_gradient(pm), get_gradient(new_pm), warp_matrix, warp_mode, criteria)
            # aligned_pm = cv2.warpAffine(new_pm, warp_matrix, pm.shape, flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            #
            # new_xy = np.nonzero(aligned_pm)
            #
            # new_x0 = np.min(new_xy[1])
            # new_x1 = np.max(new_xy[1])
            # new_y0 = np.min(new_xy[0])
            # new_y1 = np.max(new_xy[0])
            #
            # new_box = [new_x0, new_y0, new_x1-new_x0, new_y1-new_y0]

            ''' Find the rotated box ? mask '''
            # new_pm = np.zeros_like(pm, dtype=np.uint8)
            # new_pm[box[1]:box[1]+box[3], box[0]:box[0]+box[2]] = scaled_template
            # keypoints_template, keypoints_search, matches = self.SIFT_matching(new_pm, pm)


            results.append(res)
            boxes.append(box)
            centers.append(cnt)
            # aligned_masks.append(aligned_pm)
            # new_boxes.append(new_box)
            new_imgs.append(new_im)

            # new_masks.append(new_pm)
            # new_masks.append(d)
            # kpt_template.append(keypoints_template)
            # kpt_search.append(keypoints_search)
            # kpt_matches.append(matches)

        results = np.asarray(results)
        boxes = np.asarray(boxes)
        centers = np.asarray(centers)
        new_imgs = np.asarray(new_imgs)
        # aligned_masks = np.asarray(aligned_masks)
        # new_boxes = np.asarray(new_boxes)
        # new_masks = np.asarray(new_masks)
        # kpt_template = np.asarray(kpt_template)
        # kpt_search = np.asarray(kpt_search)
        # kpt_matches = np.asarray(kpt_matches)

        return results, boxes, new_imgs # , new_masks # , aligned_masks, new_boxes  #  , kpt_template, kpt_search, kpt_matches

    def SIFT_matching(self, template, search_img):
        sift = cv2.xfeatures2d.SIFT_create()

        template_gray = cv2.normalize(template, None, 0, 1, cv2.NORM_MINMAX)
        template_gray = np.asarray(template_gray*255, dtype=np.uint8)
        keypoints_template, descriptors_template = sift.detectAndCompute(template_gray, None)

        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

        search_img_gray = search_img.copy()
        search_img_gray = cv2.normalize(search_img_gray, None, 0, 1, cv2.NORM_MINMAX)
        search_img_gray = np.asarray(search_img_gray*255, dtype=np.uint8)
        keypoints_search, descriptors_search = sift.detectAndCompute(search_img_gray, None)

        matches = bf.match(descriptors_template, descriptors_search)
        matches = sorted(matches, key=lambda x:x.distance)

        # (x,y)
        keypoints_template = [kp.pt for kp in keypoints_template]
        keypoints_search = [kp.pt for kp in keypoints_search]

        # m.queryIdx  in the template
        # m.trainIdx  in the search_img
        matches_idx_template = [m.queryIdx for m in matches]
        matches_idx_search = [m.trainIdx for m in matches]


        return keypoints_template, keypoints_search, matches


    def get_mask(self, image, depth):
        mask = image.copy()
        mask = mask.copy()
        mask[mask > depth+self.std//2] = 0
        mask[mask < depth-self.std//2] = 0
        mask = (mask - depth) / (self.std * 2/2)
        mask = ndimage.median_filter(mask, size=3)

        l = 256
        n = 12
        sigma = l / (4. * n)
        mask = filters.gaussian(mask, sigma=sigma)
        mask = mask > 0.7 * mask.mean()
        mask = np.asarray(mask, dtype=np.uint8)

        return mask

    def refine_box(self, candidate_boxes):

        return candidate_boxes


if __name__ == '__main__':

    # seq = 'humans_shirts_room_occ_1_A' # !!!!!!
    # seq = 'backpack_blue'    #!!!!!
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
        peaks, peak_depths, peak_responses, peak_masks, search_img, response, distance, matching_res, matching_box, new_masks = tracker.track(frame)

        toc_track = time.time()
        print('Track a frame time : ', toc_track - tic_track)



        plt.clf()

        M = 4
        N = 4

        plt.axis('off')

        ax1 = plt.subplot(M,N,1)

        template = np.squeeze(template)
        ax1.imshow(template)
        ax1.set_title('template')
        ax1.set_xticks([])
        ax1.set_yticks([])

        ax2 = plt.subplot(M,N,2)
        ax2.imshow(tracker.mask)
        ax2.set_title('tracker.mask')
        ax2.set_xticks([])
        ax2.set_yticks([])

        ax3 = plt.subplot(M,N,3)
        ax3.imshow(search_img)

        ax3.set_title('search_img')
        ax3.set_xticks([])
        ax3.set_yticks([])

        ax4 = plt.subplot(M, N, 4)
        ax4.imshow(response)

        ax4.set_xticks([])
        ax4.set_yticks([])

        for p_xy in peaks:
            ax3.scatter(p_xy[0], p_xy[1], linewidth=1, c='r')
            ax4.scatter(p_xy[0], p_xy[1], linewidth=1, c='r')

        for ii, pm in enumerate(peak_masks):
            ax = plt.subplot(M, N, 5+ii)

            p_xy = peaks[ii]
            ax.imshow(pm)
            ax.scatter(p_xy[0], p_xy[1], linewidth=1, c='r')

            box = matching_box[ii]
            center = [box[0]+box[2]//2, box[1]+box[3]//2]
            rect = patches.Rectangle((box[0], box[1]), box[2], box[3], edgecolor='b', facecolor='none')
            ax.add_patch(rect)


            # box = new_boxes[ii]
            # rect = patches.Rectangle((box[0], box[1]), box[2], box[3], edgecolor='g', facecolor='none')
            # ax.add_patch(rect)

            ax.scatter(center[0], center[1], linewidth=1, c='b')

            ax.set_title(str(distance[ii]))
            #
            # kpts = kpt_search[ii]
            # kpts = np.asarray(kpts)
            # for xy in kpts:
            #     ax.scatter(xy[0], xy[1], linewidth=1, c='r')
            #
            # match = kpt_matches[ii]
            # match_idx = [m.trainIdx for m in match]
            # match_idx = np.asarray(match_idx)
            # match_pts = kpts[match_idx]
            # for mpt in match_pts:
            #     ax.scatter(mpt[0], mpt[1], linewidth=1, c='g')

        for ii, ms in enumerate(matching_res):

            ax = plt.subplot(M, N, 9+ii)

            # ms = ms+pm
            # ms = cv2.normalize(ms, None, 0, 1, cv2.NORM_MINMAX)
            ax.imshow(ms)

        # for ii, ams in enumerate(aligned_masks):
        #
        #     pm = peak_masks[ii]
        #
        #     ax = plt.subplot(M, N, 13+ii)
        #     ax.imshow(ams)
        #
        #     box = new_boxes[ii]
        #     rect = patches.Rectangle((box[0], box[1]), box[2], box[3], edgecolor='g', facecolor='none')
        #     ax.add_patch(rect)

        # new_masks, kpt_template, kpt_search, kpt_matches
        # # m.queryIdx  in the template
        # # m.trainIdx  in the search_img
        # matches_idx_template = [m.queryIdx for m in kpt_matches]
        # matches_idx_search = [m.trainIdx for m in kpt_matches]

        for ii, nm in enumerate(new_masks):
            ax = plt.subplot(M, N, 13+ii)
            ax.imshow(nm)

            # match_pts = kpt_template[ii]
            # for xy in match_pts:
            #     ax.scatter(xy[0], xy[1], linewidth=1, c='r')


        plt.show(block=False)
        plt.pause(3)
