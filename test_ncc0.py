from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
# from pointnet.dataset import ShapeNetDataset, ModelNetDataset
# from pointnet.model import PointNetCls, feature_transform_regularizer, PointNetfeat, PointNetCls_test
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
        # prob[prob > 0] = 1

        l = 256
        n = 12
        sigma = l / (4. * n)
        prob = filters.gaussian(prob, sigma=sigma)
        mask = prob > 0.7 * prob.mean()
        mask = np.asarray(mask, dtype=np.uint8)
        self.mask = mask
        # self.template = np.multiply(self.template, mask)


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

        search_img = (search_img - self.depth) * 1.0 / self.depth
        search_img = np.asarray(search_img, dtype=np.float32)
        search_img = torch.from_numpy(search_img[np.newaxis, np.newaxis, ...])

        return search_img, search_region

    def track(self, frame):
        print('std : ', self.std)
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

        center, target_depth, score, flag, target_xy, coordinates, depth_area_box = self.localize(search_img, search_region, response)
        location, mask, score, flag, search_region, prob = self.update(frame, search_img, search_region, center, score, target_depth, flag)

        location[2] = frame.shape[1] - location[0] if frame.shape[1] < location[0]+location[2] else location[2]
        location[3] = frame.shape[0] - location[1] if frame.shape[0] < location[1]+location[3] else location[3]

        self.flag = flag
        if flag == 'fully_occlusion':
            mask = np.zeros_like(search_img)

        return location, mask, score, response, flag, search_region, target_depth, target_xy, coordinates, depth_area_box, prob

    def localize(self, search_img, search_region, response):

        search_x0, search_y0, _, _ = search_region

        depth_area_box = [0, 0, 0, 0]
        if np.max(response) < 0.05:
            flag = 'not_found'
            max_response = 0

            center = [self.pos[0] - self.pos[2]//2 - search_x0, self.pos[1] - self.pos[3]//2 - search_y0]
            target_depth = self.depth
            target_xy = [center]
            coordinates = [center]
        else:
            response = ndimage.maximum_filter(response, size=10, mode='constant')
            coordinates = feature.peak_local_max(response, min_distance=10)
            center_x, center_y, topK_depth = [], [], []
            topK_response = []

            for tk_xy in coordinates:
                if tk_xy[0] > 0 and tk_xy[1]> 0 :
                    center_x.append(tk_xy[1])
                    center_y.append(tk_xy[0])
                    topK_depth.append(search_img[tk_xy[0], tk_xy[1]])
                    topK_response.append(response[tk_xy[0], tk_xy[1]])

            topK_depth = np.asarray(topK_depth)
            topK_response = np.asarray(topK_response)

            if len(center_x) == 0:
                flag = 'not_found'
                max_response = 0
                center = [self.pos[0] - self.pos[2]//2 - search_x0, self.pos[1] - self.pos[3]//2 - search_y0]
                target_depth = self.depth
                target_xy = [center]
                topK_xy = [center]

            else:
                center = np.asarray([self.pos[0] - self.pos[2]//2 - search_x0, self.pos[1] - self.pos[3]//2 - search_y0])

                topK_xy = np.c_[center_x, center_y]
                topK_depth = np.asarray(topK_depth)

                xy_dist = np.linalg.norm((topK_xy - center), axis=1, ord=np.inf)
                d_dist = np.abs((topK_depth - self.depth) / self.depth)

                dist = np.multiply(xy_dist, d_dist)

                dist2 = dist.copy()
                dist2.sort()
                min_dists = dist2[-5:][::-1]

                centers = []
                max_responses = []
                depth_boxes = []
                target_depths = []
                for md in min_dists:
                    m_idx = np.where(dist == md)
                    cx, cy = topK_xy[m_idx]

                    depth_area_x0 = max(0, cx - 20)
                    depth_area_y0 = max(0, cy - 20)
                    depth_area_x1 = min(search_img.shape[1], cx + 20)
                    depth_area_y1 = min(search_img.shape[0], cy + 20)
                    depth_area = search_img[depth_area_y0:depth_area_y1, depth_area_x0:depth_area_x1]
                    depth_area = np.nan_to_num(depth_area)
                    target_depth = np.median(depth_area[np.nonzero(depth_area)])

                    depth_area_box = [depth_area_x0, depth_area_y0, depth_area_x1-depth_area_x0, depth_area_y1-depth_area_y0]

                    # max_response = response[cy, cx]

                    center = [cx, cy]

                    centers.append(center)
                    max_responses.append(response[cy, cx])
                    depth_boxes.append(depth_area_box)
                    target_depths.append(target_depth)
                # if max_response < 0.3:
                #     if max_response < 0.05:
                #         flag = 'not_found'
                #     else:
                #         flag = 'uncertain'
                # else:
                #     flag = 'normal'
                if len(centers) == 0:
                    flag = 'not_found'

        # topK_xy = np.asarray([[tc[0]+search_x0, tc[1]+search_y0] for tc in topK_xy])
        # topK_xy = np.asarray([[x, y] for x, y in topK_xy if x >= 0 and y >= 0])
        # positive_topK_xy = []
        # for x, y in topK_xy:
        #     if x >= 0 and y >= 0:
        #         positive_topK_xy.append([x, y])
        #     else:
        #         print(x, y)
        # positive_topK_xy = np.asarray(positive_topK_xy)

        # depth_area_box = np.asarray([depth_area_box[0]+search_x0, depth_area_box[1]+search_y0, depth_area_box[2], depth_area_box[3]])


        # return center, target_depth, max_response, flag, positive_topK_xy, coordinates, depth_area_box
        return centers, max_responses, depth_boxes, target_depths, flag

    def update(self, frame, search_img, search_region, center, max_response, pred_depth, flag):

        if flag == 'not_found':
            center, pred_depth, max_response, flag, search_region, search_img = self.redet(frame)

        if flag != 'not_found':

            search_x0, search_y0, search_w, search_h = search_region

            ''' Mask '''
            prob = search_img
            prob[prob > pred_depth+self.std//2] = 0
            prob[prob < pred_depth-self.std//2] = 0
            prob = (prob - pred_depth) / (self.std * 2/2)
            prob = ndimage.median_filter(prob, size=3)
            # prob[prob > 0] = 1

            l = 256
            n = 12
            sigma = l / (4. * n)
            prob = filters.gaussian(prob, sigma=sigma)
            mask = prob > 0.7 * prob.mean()
            prob[prob>0] = 1
            mask = np.asarray(mask, dtype=np.uint8)

            non_zeros = np.nonzero(mask)
            nonzeros_Y = non_zeros[0]
            nonzeros_X = non_zeros[1]

            try:
                x0, y0 = np.min(nonzeros_X), np.min(nonzeros_Y)
                x1, y1 = np.max(nonzeros_X), np.max(nonzeros_Y)
                w, h = x1 - x0, y1 - y0
                x0_in_frame = x0 + search_x0
                y0_in_frame = y0 + search_y0
                box = [x0_in_frame, y0_in_frame, w, h]
            except:
                # box = self.pos
                box = [center[0]+self.W//2+search_x0, center[1]+self.H//2+search_y0, self.W, self.H]

            target_area = np.count_nonzero(mask)
            # last_area = int(self.pos[2]*self.pos[3])
            last_area = int(self.H*self.W)
            area_diff = max(target_area, last_area) * 1.0 / (min(target_area, last_area)+1.0)
            print('target_area , last_area : ', target_area, last_area, area_diff)

            dist = [p - v for p, v in zip(box, self.pos)]
            dist = np.linalg.norm(dist)

            dist_threshold = np.sqrt(self.H * self.W)
            depth_diff = max(pred_depth, self.depth) * 1.0 / (min(pred_depth, self.depth) + 1)

            if area_diff > 1.8 and depth_diff > 1.2:
                if area_diff > 3:
                    flag = 'fully_occlusion'
                else:
                    flag = 'occlusion'

            if dist < 3*dist_threshold and depth_diff < 1.2 and area_diff < 1.8 and flag not in ['occlusion', 'fully_occlusion', 'not_found'] : #  and area_diff > 0.8 and area_diff < 1.2:
                print('Updating self.pos .....')
                self.pos = box
                self.depth = pred_depth

            else:
                print('info dist, depth, area, flag : ', dist/(dist_threshold+1.0), depth_diff, area_diff, flag)
                box = self.pos
                pred_depth = self.depth

                if flag == 'norm':
                    flag = 'uncertain'

        else:
            box = self.pos
            mask = np.zeros_like(search_img)
            prob = mask
            flag = 'not_found'

        return box, mask, max_response, flag, search_region, prob

    def redet(self, frame):
        print('Redetecting ... ')

        # x0, y0, _, _ = self.pos
        # cx, cy = x0 - self.W//2, y0 - self.H//2

        max_response = 0
        search_img = frame
        response = None
        search_region = [0, 0, frame.shape[1], frame.shape[0]]

        for scale in range(1, 5):
            cand_search_img, cand_search_region = self.generate_search_region(frame, scale=scale, center=None)

            cand_search_img = cand_search_img.cuda()

            cand_response = self.ncc(cand_search_img)
            cand_response = cand_response.detach().cpu().numpy()
            cand_response = np.squeeze(cand_response)

            cand_max_response = np.max(cand_response)

            if max_response < cand_max_response:
                max_response = cand_max_response
                search_img = cand_search_img
                search_region = cand_search_region
                response = cand_response
            else:
                del cand_search_img
                del cand_search_region
                del cand_response

        try:
            search_img = search_img.detach().cpu().numpy().squeeze()
        except:
            search_img = np.squeeze(search_img)
        search_img = search_img * self.depth + self.depth

        if max_response < 0.05:
            flag = 'not_found'
            search_region = [0, 0, frame.shape[1], frame.shape[0]]
            search_img = frame
            center = [self.pos[0], self.pos[1]]
            target_depth = self.depth
        else:
            center, target_depth, max_response, flag, topK_xy, coordinates, depth_area_box = self.localize(search_img, search_region, response)

        if flag != 'not_found':
            x0 = max(0, center[0] - self.W // 2)
            y0 = max(0, center[1] - self.H // 2)

        return center, target_depth, max_response, flag, search_region, search_img


if __name__ == '__main__':

    # seq = 'humans_shirts_room_occ_1_A' # !!!!!!
    # seq = 'backpack_blue'   3 !!!!!
    # seq = 'bicycle_outside' # !!!!
    # seq = 'XMG_outside'
    seq = 'backpack_robotarm_lab_occ'
    # seq = 'bottle_box' # !!!!?
    # seq = 'box_darkroom_noocc_10' # !!!!!!!!!!!!!!!!!!!!!!!!! multi depth seed
    # seq = 'box_darkroom_noocc_9'
    # seq = 'box_darkroom_noocc_8'
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
        # frame[frame<10000] = 10000
        location, mask, score, response, flag, search_region, target_depth, topK_xy, coordinates, depth_area_box, prob = tracker.track(frame)

        toc_track = time.time()
        print('Track a frame time : ', toc_track - tic_track)



        plt.clf()

        N = 3
        ax1 = plt.subplot(2,N,1)

        template = np.squeeze(template)
        ax1.imshow(template)

        ax2 = plt.subplot(2,N,2)
        ax2.imshow(frame)
        if location[0]:

            x0, y0, w, h = location
            rect = patches.Rectangle((x0, y0), w, h, linewidth=1, edgecolor='r', facecolor='none')
            ax2.add_patch(rect)

            x0, y0, w, h = search_region
            rect = patches.Rectangle((x0, y0), w, h, linewidth=1, edgecolor='g', facecolor='none')
            ax2.add_patch(rect)

            x0, y0, w, h = depth_area_box
            rect = patches.Rectangle((x0, y0), w, h, linewidth=1, edgecolor='k', facecolor='none')
            ax2.add_patch(rect)

            # if topK_xy[0].any():
            if len(topK_xy) > 0:
                print('num of possible center : ', len(topK_xy))
                for i, t_xy in enumerate(topK_xy):

                    cx, cy = t_xy[0], t_xy[1]
                    if cx > 0 and cy > 0:
                        ax2.scatter(cx, cy, linewidth=1, c='b')

            ax2.set_title('Depth : '+ str(target_depth))

        ax3 = plt.subplot(2,N,3)
        ax3.imshow(np.exp(response))
        x0, y0, w, h = location
        rect = patches.Rectangle((x0-search_region[0], y0-search_region[1]), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax3.add_patch(rect)

        # if topK_xy[0].any():
        if len(topK_xy) > 0:
            for t_xy in topK_xy:
                cx, cy = t_xy[0]-search_region[0], t_xy[1]-search_region[1]
                if cx > 0 and cy > 0:
                    ax3.scatter(cx, cy, linewidth=1, c='b')

        # if coordinates[0].any():
        if len(coordinates) > 0:
            for xy in coordinates:
                ax3.scatter(xy[1], xy[0], linewidth=1, c='r')

        ax3.set_title('Max respose : ' + str(score))

        ax4 = plt.subplot(2,N,4)
        ax4.imshow(mask)
        x0, y0, w, h = location
        rect = patches.Rectangle((x0-search_region[0], y0-search_region[1]), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax4.add_patch(rect)


        ax4.set_title('flag : ' + flag)

        ax5 = plt.subplot(2, N, 5)
        ax5.imshow(prob)
        ax5.set_title('prob')

        ax6 = plt.subplot(2,N,6)
        ax6.imshow(tracker.mask)
        ax6.set_title('tracker.mask')


        plt.show(block=False)
        plt.pause(0.01)
