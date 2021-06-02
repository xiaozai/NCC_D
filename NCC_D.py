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

class NCCTracker(object):

    def __init__(self, image, region):
        self.window = max(region.width, region.height) * 2

        left = max(region.x, 0)
        top = max(region.y, 0)

        right = min(region.x + region.width, image.shape[1] - 1)
        bottom = min(region.y + region.height, image.shape[0] - 1)

        self.template = image[int(top):int(bottom), int(left):int(right)]
        self.position = (region.x + region.width / 2, region.y + region.height / 2)
        self.size = (region.width, region.height)

        self.depth = np.median(self.template[np.nonzero(self.template)])
        self.std = np.std(self.template[np.nonzero(self.template)]) + 0.0000000001

        template = self.template
        template = (template - self.depth) / self.std
        template = np.asarray(template, dtype=np.float32)
        template = torch.from_numpy(template[np.newaxis, ...])
        template = template.cuda()

        self.ncc = NCC(template)

        self.ncc = self.ncc.cuda()

    def track(self, image):

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
        if math.isnan(np.max(response)):
            max_loc = self.position
        else:
            max_loc_y, max_loc_x = np.where(response == np.max(response))
            print(max_loc_y, max_loc_x)
            max_loc = [max_loc_x[0], max_loc_y[0]]

        max_val = np.max(response)

        self.position = (left + max_loc[0] + float(self.size[0]) / 2, top + max_loc[1] + float(self.size[1]) / 2)

        return vot.Rectangle(left + max_loc[0], top + max_loc[1], self.size[0], self.size[1])

# handle = vot.VOT("rectangle")
handle = vot.VOT("rectangle",'rgbd')
selection = handle.region()

imagefile = handle.frame()
imagefile_rgb=imagefile[0]
imagefile_d=imagefile[1]
assert imagefile_rgb.find("color")>=0
assert imagefile_d.find("depth")>=0

# if not imagefile:
if not imagefile_d:
    sys.exit(0)

# image = cv2.imread(imagefile, cv2.IMREAD_GRAYSCALE)
image = cv2.imread(imagefile_d, -1)
# image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
image = np.asarray(image, dtype=np.float32)
tracker = NCCTracker(image, selection)
while True:
    imagefile = handle.frame()
    imagefile_d=imagefile[1]
    # if not imagefile:
    if not imagefile_d:
        break

    # image = cv2.imread(imagefile, cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(imagefile_d, -1)
    # image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    image = np.asarray(image, dtype=np.float32)
    # region, confidence = tracker.track(image)
    region = tracker.track(image)
    print(region)
    confidence = 1
    handle.report(region, confidence)
