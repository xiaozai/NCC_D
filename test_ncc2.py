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
from sklearn.cluster import KMeans
import scipy
from scipy import ndimage, misc
from skimage import measure
from skimage import filters
from skimage import feature


# from icp.icp import icp
# from icp.icp_python import icp

from sklearn.neighbors import NearestNeighbors

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
        self.angle = 5
        self.visualize = True

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

        tic = time.time()
        search_img, search_region = self.generate_search_region(frame)
        toc = time.time()
        print('generate search region time : ', toc-tic)

        tic = time.time()
        response, search_img = self.get_response(search_img)
        toc = time.time()
        print('get response time : ', toc - tic)

        tic = time.time()
        target_box, target_mask, peaks, peak_depths, peak_scales, peak_responses, peak_masks, distance, matching_res, matching_box, aligned_template, boxes_iou = self.localize(frame, search_img, search_region, response)
        toc = time.time()
        print('matching mask time : ', toc - tic )

        target_box_in_frame = target_box.copy()
        target_box_in_frame[0] = target_box_in_frame[0] + search_region[0]
        target_box_in_frame[1] = target_box_in_frame[1] + search_region[1]


        tic = time.time()
        self.update(target_box_in_frame, target_mask, frame)
        toc = time.time()
        print('updating time : ', toc-tic)

        return target_box, target_mask, peaks, peak_depths, peak_responses, peak_masks, search_img, response, distance, matching_res, matching_box, aligned_template, boxes_iou


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

        matching_res, matching_box, aligned_template, boxes_iou = self.mask_matching(search_img, peak_masks, peak_scales)

        target_idx = np.where(boxes_iou == np.max(boxes_iou))[0][0]

        target_box = matching_box[target_idx]
        target_mask = peak_masks[target_idx]
        target_mask = target_mask[target_box[1]:target_box[1]+target_box[3], target_box[0]:target_box[0]+target_box[2]]

        return target_box, target_mask, peaks, peak_depths, peak_scales, peak_responses, peak_masks, distance, matching_res, matching_box, aligned_template, boxes_iou

    def rotate_image(self, image, image_center, angle):
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

    def mask_matching(self, search_img, masks, scales):

        ''' Template matching
            If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum otherwise maximum
        '''

        results = []
        boxes = []
        centers = []
        boxes_iou = []

        overlap_template = np.multiply(self.template, self.mask)
        overlap_template = np.asarray(overlap_template, dtype=np.float32)
        w, h = overlap_template.shape[::-1]

        aligned_template = []
        for pm, s in zip(masks, scales):

            ''' find the center location based on the Template Matching '''
            overlap_pm = np.multiply(search_img, pm)
            overlap_pm = np.asarray(overlap_pm, dtype=np.float32)

            res_template_match = cv2.matchTemplate(overlap_pm, overlap_template, cv2.TM_SQDIFF) # W-w+, H-h+1
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res_template_match)
            box = [min_loc[0], min_loc[1], int(w*s), int(h*s)]
            cnt = [min_loc[0]+int(w*s/2), min_loc[1]+int(h*s/2)]


            ''' Find the optimal rotation and scales to get minimal IoU '''

            from scipy.optimize import minimize
            params = np.asarray([self.angle, 1.0, 0, 0])

            def rosen(params):
                scaled_mask = cv2.resize(self.mask, (int(w*params[1]), int(h*params[1])), interpolation=cv2.INTER_AREA)
                hh, ww = scaled_mask.shape
                template_img = np.zeros_like(pm, dtype=np.uint8)

                center = [int(cnt[0]+params[2]), int(cnt[1]+params[3])]

                x0 = center[0] - int(ww)//2
                y0 = center[1] - int(hh)//2
                x1 = x0 + ww
                y1 = y0 + hh

                template_img[y0:y1, x0:x1] = scaled_mask


                rotate_template = self.rotate_image(template_img, center, params[0])

                overlap_imgs = rotate_template + pm
                iou = np.sum(overlap_imgs>1) / np.sum(overlap_imgs>0)

                return -1 * iou

            tic = time.time()
            res = minimize(rosen, params, method='nelder-mead', options={'xatol':1e-8, 'disp':False})
            optimal_angle, optimal_scale, x_trans, y_trans = res.x
            max_iou = -1.0*res.fun
            toc = time.time()

            if self.visualize:
                print('iou : ', max_iou, '  angle : ', optimal_angle, '  scale : ', optimal_scale)
            scaled_mask = cv2.resize(self.mask, (int(w*optimal_scale), int(h*optimal_scale)), interpolation=cv2.INTER_AREA)
            template_img = np.zeros_like(pm, dtype=np.uint8)
            hh, ww = scaled_mask.shape
            center = [int(cnt[0]+x_trans), int(cnt[1]+y_trans)]

            x0 = center[0] - int(ww)//2
            y0 = center[1] - int(hh)//2
            x1 = x0 + ww
            y1 = y0 + hh
            template_img[y0:y1, x0:x1] = scaled_mask

            rotate_template = self.rotate_image(template_img, center, optimal_angle)

            overlap_tp = rotate_template + pm
            overlap_tp[overlap_tp<2] = 0

            opt_y, opt_x = np.nonzero(overlap_tp)

            opt_x0 = np.min(opt_x)
            opt_y0 = np.min(opt_y)
            opt_x1 = np.max(opt_x)
            opt_y1 = np.max(opt_y)

            opt_boxes = [opt_x0, opt_y0, opt_x1-opt_x0, opt_y1-opt_y0]

            rotate_template = rotate_template + pm
            aligned_template.append(rotate_template)

            results.append(res_template_match)
            boxes.append(opt_boxes)
            centers.append(cnt)
            boxes_iou.append(max_iou)


        aligned_template = np.asarray(aligned_template)
        results = np.asarray(results)
        boxes = np.asarray(boxes)
        centers = np.asarray(centers)
        boxes_iou = np.asarray(boxes_iou)

        return results, boxes, aligned_template, boxes_iou

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

    def update(self, prediction_box, prediction_mask, frame):
        pred_area = prediction_box[2]*prediction_box[3]
        pred_center = [prediction_box[0]+prediction_box[2]//2, prediction_box[1]+prediction_box[3]//2]


        gt_area = self.pos[2]*self.pos[3]
        gt_center = [self.pos[0]+self.pos[2]//2, self.pos[1]+self.pos[3]//2]

        area_diff = abs(pred_area - gt_area) / gt_area
        cnt_diff = np.sqrt((pred_center[0]-gt_center[0])**2 + (pred_center[1]-gt_center[1])**2)

        distance_threshold = np.sqrt(self.pos[2]*self.pos[3])

        if area_diff < 0.3 and cnt_diff < distance_threshold:

            print('Update ....')
            self.pos = prediction_box
            self.mask = prediction_mask

            self.template = frame[prediction_box[1]:prediction_box[1]+prediction_box[3], prediction_box[0]:prediction_box[0]+prediction_box[2]]
            self.depth = np.median(self.template[np.nonzero(self.template)])
            self.std = np.std(self.template[np.nonzero(self.template)])




if __name__ == '__main__':

    root_path = '/home/yan/Data2/vot-workspace-CDTB/sequences/'

    # seq = 'humans_shirts_room_occ_1_A' # !!!!!!
    # seq = 'backpack_blue'    #!!!!!
    # seq = 'bicycle_outside' # !!!!
    # seq = 'XMG_outside'
    # seq = 'backpack_robotarm_lab_occ'
    # seq = 'bottle_box' # !!!!?
    # seq = 'box_darkroom_noocc_10' # !!!!!!!!!!!!!!!!!!!!!!!!! multi depth seed
    # seq = 'box_darkroom_noocc_9'
    seq = 'box_darkroom_noocc_8'
    # data_path = '/home/sgn/Data1/yan/Datasets/CDTB/%s/depth/'%seq
    data_path = root_path + '%s/depth'%seq

    frame_id = 0

    init_frame = cv2.imread(os.path.join(data_path, '%08d.png'%(frame_id+1)), -1)

    with open(root_path+'%s/groundtruth.txt'%seq, 'r') as fp:
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
        target_box, target_mask, peaks, peak_depths, peak_responses, peak_masks, search_img, response, distance, matching_res, matching_box, aligned_template, boxes_iou = tracker.track(frame)

        toc_track = time.time()
        print('Track a frame time : ', toc_track - tic_track)


        fig = plt.figure(1)

        plt.clf()

        M = 4
        N = 4

        plt.axis('off')

        ax1 = fig.add_subplot(M,N,1)

        template = np.squeeze(template)
        ax1.imshow(template)
        ax1.set_title('template')


        ax2 = fig.add_subplot(M,N,2)
        ax2.imshow(tracker.mask)
        ax2.set_title('tracker.mask')

        ax3 = fig.add_subplot(M,N,3)
        ax3.imshow(search_img)
        rect = patches.Rectangle((target_box[0], target_box[1]), target_box[2], target_box[3], edgecolor='b', facecolor='none')
        ax3.add_patch(rect)

        ax3.set_title('search_img')

        ax4 = fig.add_subplot(M, N, 4)
        ax4.imshow(response)


        for p_xy in peaks:
            ax3.scatter(p_xy[0], p_xy[1], linewidth=1, c='r')
            ax4.scatter(p_xy[0], p_xy[1], linewidth=1, c='r')

        for ii, pm in enumerate(peak_masks):
            ax = fig.add_subplot(M, N, 5+ii)

            p_xy = peaks[ii]
            ax.imshow(pm)
            ax.scatter(p_xy[0], p_xy[1], linewidth=1, c='r')
            box = matching_box[ii, ...]
            center = [box[0]+box[2]//2, box[1]+box[3]//2]
            rect = patches.Rectangle((box[0], box[1]), box[2], box[3], edgecolor='b', facecolor='none')
            ax.add_patch(rect)
            ax.scatter(center[0], center[1], linewidth=1, c='b')
            ax.set_title(str(distance[ii]))

        for ii, ms in enumerate(matching_res):

            ax = fig.add_subplot(M, N, 9+ii)
            ax.imshow(ms)

        for ii, at in enumerate(aligned_template):
            ax = fig.add_subplot(M, N, 13+ii)
            ax.imshow(at)

            box = matching_box[ii, ...]
            center = [box[0]+box[2]//2, box[1]+box[3]//2]
            rect = patches.Rectangle((box[0], box[1]), box[2], box[3], edgecolor='b', facecolor='none')
            ax.add_patch(rect)
            ax.scatter(center[0], center[1], linewidth=1, c='b')

            iou = boxes_iou[ii]
            ax.set_title('iou :'+str(iou))

        ax = fig.add_subplot(M, N, 16)
        ax.imshow(target_mask)

        plt.show(block=False)
        plt.pause(0.1)
