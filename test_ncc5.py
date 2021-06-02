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


from sklearn.neighbors import NearestNeighbors
from scipy.optimize import minimize

blue = lambda x: '\033[94m' + x + '\033[0m'

class NCC_depth(torch.nn.Module):

    def __init__(self, frame, xywh, visualize=False):
        super(NCC_depth, self).__init__()

        self.pos = xywh         # [x, y, w, h] in image
        self.flag = 'init'

        # self.template, self.mask, self.depth, self.std = self.get_mask_by_grabcut(frame, xywh)
        self.template = frame[xywh[1]:xywh[1]+xywh[3], xywh[0]:xywh[0]+xywh[2]]


        cnt = [xywh[0] - xywh[2]//2, xywh[1] - xywh[3]//2]
        self.depth, self.std = self.get_depth(self.template, cnt=cnt)

        self.mask = self.get_mask_by_gaussian(self.template, self.depth, std=self.std)
        print(self.mask.shape, self.template.shape)

        self.ncc = self.initalize_ncc(self.template)

        self.angle = 5
        self.visualize = visualize

    def get_mask_by_grabcut(self, img, xywh):

        try:
            x0, y0, w, h = xywh
            H, W = img.shape

            n_bg_pixels = 30
            new_x0 = max(0, x0-n_bg_pixels)
            new_y0 = max(0, y0-n_bg_pixels)
            new_x1 = min(W, x0+w+n_bg_pixels)
            new_y1 = min(H, y0+h+n_bg_pixels)

            img02 = img02[new_y0:new_y1, new_x0:new_x1]
            box_in_img = [x0-new_x0, y0-new_y0, w, h]

            mask = np.zeros(img02.shape[:2],np.uint8)
            bgdModel = np.zeros((1,65),np.float64)
            fgdModel = np.zeros((1,65),np.float64)
            rect = (box_in_img[0], box_in_img[1], box_in_img[0]+box_in_img[2], box_in_img[1]+box_in_img[3])

            colormap = cv2.normalize(img02, None, 0, 255, cv2.NORM_MINMAX)
            colormap = np.asarray(colormap, dtype=np.uint8)
            colormap = cv2.applyColorMap(colormap, cv2.COLORMAP_JET)
            cv2.grabCut(colormap,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
            mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')

            img02 = img02 * mask2[:, :]
            template = img02[box_in_img[1]:box_in_img[1]+box_in_img[3], box_in_img[0]:box_in_img[0]+box_in_img[2]]

            template_depth = np.median(template[np.nonzero(template)])
            template_std = np.std(template[np.nonzero(template)])

            Y, X = np.nonzero(template)
            x0 = np.min(X)
            x1 = np.max(X)
            y0 = np.min(Y)
            y1 = np.max(Y)

            if (y1-y0>0) and (x1-x0>0):
                template = template[y0:y1, x0:x1]
                template_mask = template > 0
                if np.sum(template_mask) < 0.1*(y1-y0)*(x1-x0):
                    template_mask = np.ones_like(template)
            else:
                template_mask = np.ones_like(template)

        except:
            print('Warning : failed in Grabcut !!!!!!!')
            x0, y0, w, h = xywh
            print(xywh)
            template = img[y0:y0+h, x0:x0+w]
            print(template.shape)


            template_depth = np.median(template[np.nonzero(template)])
            template_std = np.std(template[np.nonzero(template)])
            print(template_depth, template_std)
            template_mask = self.get_mask_by_gaussian(template, template_depth, cnt=None, std=template_std)

        return template, np.asarray(template_mask, dtype=np.uint8), template_depth, template_std

    def initalize_ncc(self, template):
        template = (template - self.depth) * 1.0 / self.depth
        template = np.asarray(template, dtype=np.float32)
        template = torch.from_numpy(template[np.newaxis, ...])
        template = template.cuda()

        ncc = NCC(template)
        ncc = ncc.cuda()

        return ncc

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
            new_y1 = int(min(H, new_y0+new_w))

        search_region = [new_x0, new_y0, new_x1-new_x0, new_y1-new_y0]
        search_img = frame[new_y0:new_y1, new_x0:new_x1]

        return search_img, search_region

    def get_mask_by_gaussian(self, image, depth, cnt=None, std=None):

        import scipy
        import scipy.stats

        if std is None:
            std = self.std
        prob = scipy.stats.norm(depth, std).pdf(image)
        prob = prob > 0.7*np.mean(prob[np.nonzero(prob)])
        mask = prob

        return mask

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

        target_box, target_mask, peaks, peak_masks, matching_box, aligned_template, boxes_iou, optimal_angle = self.localize(frame, search_img, search_region, response)

        target_box_in_frame = target_box.copy()
        target_box_in_frame[0] = target_box_in_frame[0] + search_region[0]
        target_box_in_frame[1] = target_box_in_frame[1] + search_region[1]

        self.update(target_box_in_frame, target_mask, frame, optimal_angle)

        return target_box_in_frame, target_mask, peaks, peak_masks, matching_box, aligned_template, boxes_iou, search_img, response, search_region


    def localize(self, frame, search_img, search_region, response):

        ''' Select the suitable peaks

            for each peak:
                - xy
                - depth
                - response

            based on the peaks:
                - prediction depth
                - possible mask
                - possible area    --> compare with template
                - possible center  --> relationship with neighbour pixels
                - possible rotation and scales --> optimization --> IoU



        '''
        H, W = frame.shape
        h, w = self.template.shape
        search_x0, search_y0, search_w, search_h = search_region
        last_center = [self.pos[0]+self.pos[2]//2, self.pos[1]+self.pos[3]//2]
        xy_threshold = np.sqrt(self.pos[2]*self.pos[3]) + 0.000001

        ''' find the peak response in the response map of the search_image'''
        response = ndimage.maximum_filter(response, size=10, mode='constant')
        peaks = feature.peak_local_max(response, min_distance=7, num_peaks=50)

        if len(peaks) > 0:

            ''' Maximum Response for peaks '''
            peak_response = np.asarray([response[pt[1], pt[0]] for pt in peaks])
            high_response_idx = np.where(peak_response > np.median(peak_response))[0]
            peaks = peaks[high_response_idx]
            peak_response = peak_response[high_response_idx]

        if len(peaks) > 0:

            ''' xy distance '''
            peak_xydist = np.asarray([np.sqrt((pt[0]+search_x0-last_center[0])**2 + (pt[1]+search_y0-last_center[1])**2) for pt in peaks])
            close_xy_idx = np.where(peak_xydist < 1.5*xy_threshold)[0]
            peaks = peaks[close_xy_idx]
            peak_response = peak_response[close_xy_idx]
            peak_xydist = peak_xydist[close_xy_idx]

        if len(peaks) > 0:
            ''' depth change '''
            peak_depths = []
            for pxy in peaks:
                px, py = pxy
                temp_area = search_img[max(0, py-h//2):min(search_h, py+h//2), max(0, px-w//2):min(search_w, px+w//2)]
                pd = np.median(temp_area[np.nonzero(temp_area)])
                peak_depths.append(pd)
            peak_depths = np.asarray(peak_depths)

            d_diff = np.asarray([abs(pd - self.depth)/self.depth for pd in peak_depths])
            small_depthchange_idx = np.where(d_diff < 0.5)[0]

            peaks = peaks[small_depthchange_idx]
            peak_depths = peak_depths[small_depthchange_idx]
            peak_response = peak_response[small_depthchange_idx]
            peak_xydist = peak_xydist[small_depthchange_idx]
            peak_ddiff = d_diff[small_depthchange_idx]

        if len(peaks) > 0:
            ''' Remove similar peaks'''

            unique_idx = []
            ii, jj = 0, 1
            while ii < len(peaks) and jj < len(peaks):
                cur_pt = peaks[ii]
                next_pt = peaks[jj]
                dist = np.sqrt((cur_pt[0] - next_pt[0])**2 + (cur_pt[1] - next_pt[1])**2)

                if dist > 4:
                    unique_idx.append(ii)
                    ii = jj
                    jj += 1
                else:
                    jj += 1
            unique_idx.append(ii)

            unique_idx = np.asarray(unique_idx)


            peaks = peaks[unique_idx]
            peak_depths = peak_depths[unique_idx]
            peak_response = peak_response[unique_idx]
            peak_xydist = peak_xydist[unique_idx]
            peak_ddiff = peak_ddiff[unique_idx]

            ''' Minimum xy*d '''
            xyd_distance = np.asarray([xy_d / xy_threshold * d_d for xy_d, d_d in zip(peak_xydist, peak_ddiff)])
            sorted_dist = xyd_distance.copy()
            sorted_dist.sort()
            min_distance = sorted_dist[:30]
            min_idx = list(set([np.where(xyd_distance == md)[0][0] for md in min_distance]))
            min_idx = min_idx[::5] # choose the peak every 5
            min_idx = min_idx[:3]  # choose the top 3

            if len(min_idx) > 0:
                peaks = peaks[min_idx]
                peak_depths = peak_depths[min_idx]
            else:
                peaks = np.asarray([[last_center[0]-search_x0, last_center[1]-search_y0]])
                peak_depths = np.asarray([self.depth])
            peak_masks = [self.get_mask_by_gaussian(search_img, pd, cnt) for pd, cnt in zip(peak_depths, peaks)]

        else:
            peaks = np.asarray([last_center])
            peak_depths = np.asarray([self.depth])
            peak_masks = [self.get_mask_by_gaussian(search_img, pd, cnt) for pd, cnt in zip(peak_depths, peaks)]

        # ax1 = plt.subplot(1,4,1)
        # ax1.imshow(frame)
        # rect = patches.Rectangle((search_region[0], search_region[1]), search_region[2], search_region[3], edgecolor='b', facecolor='none')
        # ax1.add_patch(rect)
        #
        # ax2 = plt.subplot(1,4,2)
        # ax2.imshow(search_img)
        #
        # ax3 = plt.subplot(1,4,3)
        # ax3.imshow(response)
        #
        # for pts in peaks:
        #     ax3.scatter(pts[0],pts[1], c='r')
        #     ax2.scatter(pts[0],pts[1], c='r')
        #
        # ax4 = plt.subplot(1,4,4)
        # ax4.imshow(peak_masks[0])
        # plt.show()


        matching_box, aligned_template, boxes_iou, optimal_angles = self.mask_matching(search_img, peak_masks, peaks)

        if len(boxes_iou) > 0:
            target_idx = np.where(boxes_iou == np.max(boxes_iou))[0][0]

            target_box = matching_box[target_idx]

            target_mask = peak_masks[target_idx]
            target_mask = target_mask[target_box[1]:target_box[1]+target_box[3], target_box[0]:target_box[0]+target_box[2]]

            optimal_angle = optimal_angles[target_idx]
        else:
            print('Not found !!')
            self.flag = 'not_found'
            target_box = self.pos
            target_box[0] = target_box[0] - search_x0
            target_box[1] = target_box[1] - search_y0

            target_mask = np.zeros_like(search_img, dtype=np.uint8)
            target_mask = target_mask[target_box[1]:target_box[1]+target_box[3], target_box[0]:target_box[0]+target_box[2]]

            optimal_angle = self.angle




        return target_box, target_mask, peaks, peak_masks, matching_box, aligned_template, boxes_iou, optimal_angle

    def rotate_image(self, image, image_center, angle):
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

    def mask_matching(self, search_img, peak_masks, peaks):

        ''' Template matching
            If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum otherwise maximum
        '''

        results, boxes, centers, boxes_iou, aligned_template, optimal_angles = [], [], [], [], [], []

        overlap_template = np.asarray(np.multiply(self.template, self.mask),  dtype=np.float32)
        w, h = overlap_template.shape[::-1]

        print(overlap_template.shape)


        for ii, pm in enumerate(peak_masks):
            ''' find the center location based on the Template Matching '''
            overlap_pm = np.multiply(search_img, pm)
            overlap_pm = np.asarray(overlap_pm, dtype=np.float32)
            print(overlap_pm.shape)

            HH, WW = pm.shape

            ''' Redetect the center ????'''

            res_template_match = cv2.matchTemplate(overlap_pm, overlap_template, cv2.TM_CCOEFF_NORMED) # W-w+, H-h+1
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res_template_match)                     # Top-Left Point
            cnt = [max_loc[0]+int(w/2), max_loc[1]+int(h/2)]                                           # center in the search_img

            ''' remove outliers '''
            roi_x0 = max(0, cnt[0]-w)
            roi_y0 = max(0, cnt[1]-h)
            roi_x1 = min(WW, cnt[0]+w)
            roi_y1 = min(HH, cnt[1]+h)

            roi_region = np.zeros_like(pm, dtype=np.uint8)
            roi_region[roi_y0:roi_y1, roi_x0:roi_x1] = 1

            #
            # ax = plt.subplot(1,3,1)
            # ax.imshow(roi_region)
            #
            # ax2 = plt.subplot(1,3,2)
            # ax2.imshow(pm)


            pm = np.logical_and(pm, roi_region)

            # ax3 = plt.subplot(1,3,3)
            # ax3.imshow(pm)
            # plt.show()


            # cnt = peaks[ii]

            ''' Find the optimal rotation and scales to get minimal IoU '''

            # fig = plt.figure()

            def iou_loss(params):
                ''' Scale '''
                h, w = self.mask.shape
                scaled_mask = cv2.resize(self.mask, (int(w*params[1]), int(h*params[1])), interpolation=cv2.INTER_AREA)

                ''' Translation '''
                center = [int(cnt[0]+params[2]), int(cnt[1]+params[3])]             # center in the search_img

                ''' Padding '''
                template_img = np.zeros_like(pm, dtype=np.uint8)

                hh, ww = scaled_mask.shape

                x0, y0 = max(0, center[0] - int(ww)//2), max(0, center[1] - int(hh)//2)
                x1, y1 = min(WW, x0 + ww), min(HH, y0 + hh)
                template_img[y0:y1, x0:x1] = scaled_mask[:(y1-y0), :(x1-x0)]

                ''' Rotation around the center of target '''
                rotate_template = self.rotate_image(template_img, center, params[0])

                ''' IoU between rotated template and target '''
                overlap_imgs = rotate_template + pm
                iou = np.sum(overlap_imgs>1) / np.sum(overlap_imgs>0)

                return -1 * iou

            params = np.asarray([self.angle, 1.0, 0, 0])
            res = minimize(iou_loss, params, method='nelder-mead', options={'xatol':1e-2, 'disp':False})
            # xy_trans_threshold = np.sqrt(w*h)
            # xy_trans_bounds = (-1.0*xy_trans_threshold, 1.0*xy_trans_threshold)
            # res = minimize(iou_loss, params, method='BFGS', options={'xatol':1e-2, 'disp':False}, bounds=[(-45, 45), (0, 2.0), xy_trans_bounds, xy_trans_bounds])
            # res = minimize(iou_loss, params, method='nelder-mead', options={'xatol':1e-2, 'disp':False}, bounds=[(-45, 45), (0, 2.0), xy_trans_bounds, xy_trans_bounds])
            optimal_angle, optimal_scale, x_trans, y_trans = res.x
            max_iou = -1.0*res.fun


            scaled_mask = cv2.resize(self.mask, (int(w*optimal_scale), int(h*optimal_scale)), interpolation=cv2.INTER_AREA)
            template_img = np.zeros_like(pm, dtype=np.uint8)
            HH, WW = template_img.shape
            hh, ww = scaled_mask.shape
            center = [int(cnt[0]+x_trans), int(cnt[1]+y_trans)]

            x0 = max(0, center[0] - int(ww)//2)
            y0 = max(0, center[1] - int(hh)//2)
            x1 = min(WW, x0 + ww)
            y1 = min(HH, y0 + hh)
            template_img[y0:y1, x0:x1] = scaled_mask[:(y1-y0), :(x1-x0)]

            rotate_template = self.rotate_image(template_img, center, optimal_angle)

            overlap_tp = rotate_template + pm
            overlap_tp[overlap_tp<2] = 0

            try:
                opt_y, opt_x = np.nonzero(overlap_tp)

                opt_x0 = np.min(opt_x)
                opt_y0 = np.min(opt_y)
                opt_x1 = np.max(opt_x)
                opt_y1 = np.max(opt_y)

                opt_boxes = [opt_x0, opt_y0, opt_x1-opt_x0, opt_y1-opt_y0]
            except:
                opt_boxes = [0, 0, overlap_tp.shape[1], overlap_tp.shape[0]]

            rotate_template = rotate_template + pm
            aligned_template.append(rotate_template)

            # results.append(res_template_match)
            boxes.append(opt_boxes)
            centers.append(cnt)
            boxes_iou.append(max_iou)
            optimal_angles.append(optimal_angle)


        aligned_template = np.asarray(aligned_template)
        boxes = np.asarray(boxes)
        centers = np.asarray(centers)
        boxes_iou = np.asarray(boxes_iou)
        optimal_angles = np.asarray(optimal_angles)


        return boxes, aligned_template, boxes_iou, optimal_angles



    def update(self, prediction_box, prediction_mask, frame, optimal_angle):
        H, W = frame.shape
        x0, y0, w, h = prediction_box
        x1 = min(W, x0+w)
        y1 = min(H, y0+h)


        pred_area = w*h
        pred_center = [x0+w//2, y0+h//2]


        gt_area = self.pos[2]*self.pos[3]
        gt_center = [self.pos[0]+self.pos[2]//2, self.pos[1]+self.pos[3]//2]

        area_diff = abs(pred_area - gt_area) / gt_area
        cnt_diff = np.sqrt((pred_center[0]-gt_center[0])**2 + (pred_center[1]-gt_center[1])**2)

        distance_threshold = np.sqrt(self.pos[2]*self.pos[3])


        pred_ = frame[y0:y1, x0:x1]
        pred_depth = np.median(pred_[np.nonzero(pred_)])
        depth_diff = max(pred_depth, self.depth) / (min(pred_depth, self.depth) + 1.0)

        print('area diff :', area_diff, ' cnt_diff : ', cnt_diff, 'depth_diff : ', depth_diff)
        if area_diff < 0.3 and cnt_diff < distance_threshold and depth_diff < 1.3:

            print('Update ....')
            self.pos = prediction_box
            # self.mask = prediction_mask
            #
            self.angle = optimal_angle
            # self.template = frame[prediction_box[1]:prediction_box[1]+prediction_box[3], prediction_box[0]:prediction_box[0]+prediction_box[2]]
            # self.depth = np.median(self.template[np.nonzero(self.template)])
            self.depth = pred_depth
            # self.std = np.std(self.template[np.nonzero(self.template)])


def visualize(template, template_mask, frame, search_region, target_box, target_mask, response, peaks, peak_masks, matching_box, aligned_template):
    fig = plt.figure(1)

    plt.clf()

    M = 4
    N = 3
    #
    ax1 = fig.add_subplot(M,N,1)

    template = np.squeeze(template)
    ax1.imshow(template)
    ax1.set_title('template')


    ax2 = fig.add_subplot(M,N,2)
    ax2.imshow(template_mask)
    ax2.set_title('tracker.mask')

    ax = fig.add_subplot(M, N, 3)
    overlap_template = np.multiply(template, template_mask)
    ax.imshow(overlap_template)
    ax.set_title('Template overlap')



    ax3 = fig.add_subplot(M,N,4)
    ax3.imshow(frame)
    rect = patches.Rectangle((target_box[0], target_box[1]), target_box[2], target_box[3], edgecolor='b', facecolor='none')
    ax3.add_patch(rect)
    ax3.set_title('frame')

    ax4 = fig.add_subplot(M, N, 5)
    ax4.imshow(response)
    ax4.set_title('NCC Response')

    search_x0, search_y0, search_w, search_h = search_region
    rect = patches.Rectangle((search_x0, search_y0), search_w, search_h, edgecolor='r', facecolor='none')
    ax3.add_patch(rect)


    for p_xy in peaks:
        ax3.scatter(p_xy[0]+search_x0, p_xy[1]+search_y0, linewidth=1, c='r')
        ax4.scatter(p_xy[0], p_xy[1], linewidth=1, c='r')


    ax = fig.add_subplot(M, N, 6)
    ax.imshow(target_mask)
    ax.set_title('Pred Mask')

    for ii, pm in enumerate(peak_masks):
        ax = fig.add_subplot(M, N, 7+ii)

        p_xy = peaks[ii]
        ax.imshow(pm)
        ax.scatter(p_xy[0], p_xy[1], linewidth=1, c='r')
        box = matching_box[ii, ...]
        center = [box[0]+box[2]//2, box[1]+box[3]//2]
        rect = patches.Rectangle((box[0], box[1]), box[2], box[3], edgecolor='b', facecolor='none')
        ax.add_patch(rect)
        ax.scatter(center[0], center[1], linewidth=1, c='b')
        ax.set_yticks([])
        ax.set_xticks([])

    for ii, at in enumerate(aligned_template):
        ax = fig.add_subplot(M, N, 10+ii)
        ax.imshow(at)

        box = matching_box[ii, ...]
        center = [box[0]+box[2]//2, box[1]+box[3]//2]
        rect = patches.Rectangle((box[0], box[1]), box[2], box[3], edgecolor='b', facecolor='none')
        ax.add_patch(rect)
        ax.scatter(center[0], center[1], linewidth=1, c='b')

        iou = boxes_iou[ii]
        ax.set_title('iou :'+str(iou))
        ax.set_yticks([])
        ax.set_xticks([])


    plt.show(block=False)
    plt.pause(0.1)


if __name__ == '__main__':

    root_path = '/home/yan/Data2/DOT-results/CDTB-ST/sequences/'


    save_results = False

    if save_results:
        out_path = '/home/yan/Data2/DOT-results/CDTB-ST/results/NCC_D/'
        if not os.path.isdir(out_path):
            os.mkdir(out_path)
        out_path = out_path + 'rgbd-unsupervised/'
        if not os.path.isdir(out_path):
            os.mkdir(out_path)

    # seq = 'humans_shirts_room_occ_1_A' # !!!!!!
    # seq = 'backpack_blue'    #!!!!!
    # seq = 'bicycle_outside' # !!!!
    # seq = 'XMG_outside'
    # seq = 'backpack_robotarm_lab_occ'
    # seq = 'bottle_box' # !!!!?
    # seq = 'box_darkroom_noocc_10' # !!!!!!!!!!!!!!!!!!!!!!!!! multi depth seed
    # seq = 'box_darkroom_noocc_9'
    # seq = 'box_darkroom_noocc_8'
    # data_path = '/home/sgn/Data1/yan/Datasets/CDTB/%s/depth/'%seq

    sequences = os.listdir(root_path)
    sequences.remove('list.txt')

    # sequences = ['backpack_blue']


    for seq in sequences[2:]:
        print(seq)

        data_path = root_path + '%s/depth'%seq

        pred_boxes, pred_times, pred_confidences = [], [], []


        ''' ------  Initialize ------------------------------------------------'''
        tic = time.time()

        frame_id = 0
        init_frame = cv2.imread(os.path.join(data_path, '%08d.png'%(frame_id+1)), -1)

        with open(root_path+'%s/groundtruth.txt'%seq, 'r') as fp:
            gt_bboxes = fp.readlines()
        gt_bboxes = [box.strip() for box in gt_bboxes]

        init_box = gt_bboxes[frame_id]
        init_box = [int(float(bb)) for bb in init_box.split(',')]

        # template = init_frame[init_box[1]:init_box[1]+init_box[3], init_box[0]:init_box[0]+init_box[2]]

        tracker = NCC_depth(init_frame, init_box, visualize=True)
        tracker = tracker.cuda()

        toc = time.time()

        pred_times.append(toc-tic)
        pred_boxes.append(init_box)
        pred_confidences.append(1)

        for frame_id in range(1, len(gt_bboxes)):

            tic_track = time.time()
            frame = cv2.imread(os.path.join(data_path, '%08d.png'%(frame_id+1)), -1)
            target_box, target_mask, peaks, peak_masks, matching_box, aligned_template, boxes_iou, search_img, response, search_region = tracker.track(frame)

            if tracker.visualize:
                visualize(tracker.template, tracker.mask, frame, search_region, target_box, target_mask, response, peaks, peak_masks, matching_box, aligned_template)

            toc_track = time.time()
            print('Track a frame time : ', toc_track - tic_track)

            pred_boxes.append(target_box)
            pred_times.append(toc-tic)

        if save_results:
            out_seq = out_path + seq + '/'
            if not os.path.isdir(out_seq):
                os.mkdir(out_seq)

            with open(os.path.join(out_seq, '%s_001.txt'%seq), 'w') as fp:
                for pb in pred_boxes:
                    fp.write('%d,%d,%d,%d\n'%(int(pb[0]), int(pb[1]), int(pb[2]), int(pb[2])))

            with open(os.path.join(out_seq, '%s_001_time.value'%seq), 'w') as fp:
                for pt in pred_times:
                    fp.write('%f\n'%(pt))

            with open(os.path.join(out_seq, '%s_001_confidence.value'%seq), 'w') as fp:
                for pt in pred_times:
                    fp.write('1\n')
