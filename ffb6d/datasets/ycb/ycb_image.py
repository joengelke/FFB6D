#!/usr/bin/env python3
import os
import sys
import cv2
import torch
import os.path
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from common import Config
import pickle as pkl
from utils.basic_utils import Basic_Utils
import scipy.io as scio
import scipy.misc

from cv2 import imshow, waitKey
import normalSpeed
from models.RandLA.helper_tool import DataProcessing as DP


config = Config(ds_name='ycb')
bs_utils = Basic_Utils(config)
image_size = (480,640)


class YCB_IMAGE_PREPROC():

    def __init__(self, image_dir, image_name, camera='YCB', image_type='png'):
        self.xmap = np.array([[j for i in range(640)] for j in range(480)])
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])

        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.224])

        self.image_dir = image_dir
        self.image_name = image_name
        self.image_type = image_type
        self.camera = camera

    def dpt_2_pcld(self, dpt, cam_scale, K):
        if len(dpt.shape) > 2:
            dpt = dpt[:, :, 0]
        dpt = dpt.astype(np.float32) / cam_scale
        msk = (dpt > 1e-8).astype(np.float32)
        row = (self.ymap - K[0][2]) * dpt / K[0][0]
        col = (self.xmap - K[1][2]) * dpt / K[1][1]
        dpt_3d = np.concatenate(
            (row[..., None], col[..., None], dpt[..., None]), axis=2
        )
        dpt_3d = dpt_3d * msk[:, :, None]
        return dpt_3d

    def resizeKeepingAspectRation(self, img):
        new_height = int(img.shape[0] * (640/img.shape[1]))
        img = cv2.resize(img, (640, new_height))

        height, width = img.shape[:2]
        blank_image = np.zeros([image_size[0], image_size[1], img.shape[2]], np.uint8)

        l_img = blank_image.copy()
        y_offset = int((image_size[0] - new_height) / 2)
        # Here, y_offset+height <= blank_image.shape[0] and x_offset+width <= blank_image.shape[1]
        l_img[y_offset:y_offset+height, :width] = img.copy()
        return l_img

    def read_rgb_image(self):
        with Image.open(os.path.join(self.image_dir, self.image_name+'-color.'+self.image_type)) as ci:
            rgb = np.array(ci)[:, :, :3]
        if rgb.shape[:2] != (480,640):
            return self.resizeKeepingAspectRation(rgb)
        else:
            return rgb

    def read_depth_image(self):
        if self.camera == 'intel_l515':
            with Image.open(os.path.join(self.image_dir, self.image_name+'-depth.'+self.image_type)) as di:
                return np.array(di)
        else:
            with Image.open(os.path.join(self.image_dir, self.image_name+'-depth.'+self.image_type)) as di:
                return np.array(di)

    def get_item(self):
        #Load Images
        rgb = self.read_rgb_image()
        dpt_um = self.read_depth_image()

        #DEBUG REMOVE
        imshow("rgb", cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        waitKey(0)
        imshow("depth", dpt_um)
        waitKey(0)

        #Load Camera Params
        K = config.intrinsic_matrix['intel_l515']
        cam_scale = config.factor_depth['intel_l515']

        #Dont Know Yet
        msk_dp = dpt_um > 1e-6
        dpt_um = bs_utils.fill_missing(dpt_um, cam_scale, 1)
        msk_dp = dpt_um > 1e-6
        dpt_mm = (dpt_um.copy()/10).astype(np.uint16)
        nrm_map = normalSpeed.depth_normal(
            dpt_mm, K[0][0], K[1][1], 5, 2000, 20, False
        )

        #DEBUG REMOVE
        show_nrm_map = ((nrm_map + 1.0) * 127).astype(np.uint8)
        imshow("nrm_map", show_nrm_map)

        #Dont Know Yet
        dpt_m = dpt_um.astype(np.float32) / cam_scale
        dpt_xyz = self.dpt_2_pcld(dpt_m, 1.0, K)

        #Dont Know Yet
        choose = msk_dp.flatten().nonzero()[0].astype(np.uint32)
        if len(choose) < 400:
            return None
        choose_2 = np.array([i for i in range(len(choose))])
        if len(choose_2) < 400:
            return None
        if len(choose_2) > config.n_sample_points:
            c_mask = np.zeros(len(choose_2), dtype=int)
            c_mask[:config.n_sample_points] = 1
            np.random.shuffle(c_mask)
            choose_2 = choose_2[c_mask.nonzero()]
        else:
            choose_2 = np.pad(choose_2, (0, config.n_sample_points-len(choose_2)), 'wrap')
        choose = np.array(choose)[choose_2]

        sf_idx = np.arange(choose.shape[0])
        np.random.shuffle(sf_idx)
        choose = choose[sf_idx]

        cld = dpt_xyz.reshape(-1, 3)[choose, :]
        rgb_pt = rgb.reshape(-1, 3)[choose, :].astype(np.float32)
        nrm_pt = nrm_map[:, :, :3].reshape(-1, 3)[choose, :]

        choose = np.array([choose])
        cld_rgb_nrm = np.concatenate((cld, rgb_pt, nrm_pt), axis=1).transpose(1, 0)

        rgb = np.transpose(rgb, (2, 0, 1)) # hwc2chw

        xyz_lst = [dpt_xyz.transpose(2, 0, 1)] # c, h, w
        msk_lst = [dpt_xyz[2, :, :] > 1e-8]

        for i in range(3):
            msk_lst.append(xyz_lst[-1][2, :, :] > 1e-8)
        sr2dptxyz = {
            pow(2, ii): item.reshape(3, -1).transpose(1, 0) for ii, item in enumerate(xyz_lst)
        }

        rgb_ds_sr = [4, 8, 8, 8]
        n_ds_layers = 4
        pcld_sub_s_r = [4, 4, 4, 4]
        inputs = {}
        # DownSample stage
        for i in range(n_ds_layers):
            nei_idx = DP.knn_search(
                cld[None, ...], cld[None, ...], 16
            ).astype(np.int32).squeeze(0)
            sub_pts = cld[:cld.shape[0] // pcld_sub_s_r[i], :]
            pool_i = nei_idx[:cld.shape[0] // pcld_sub_s_r[i], :]
            up_i = DP.knn_search(
                sub_pts[None, ...], cld[None, ...], 1
            ).astype(np.int32).squeeze(0)
            inputs['cld_xyz%d'%i] = cld.astype(np.float32).copy()
            inputs['cld_nei_idx%d'%i] = nei_idx.astype(np.int32).copy()
            inputs['cld_sub_idx%d'%i] = pool_i.astype(np.int32).copy()
            inputs['cld_interp_idx%d'%i] = up_i.astype(np.int32).copy()
            nei_r2p = DP.knn_search(
                sr2dptxyz[rgb_ds_sr[i]][None, ...], sub_pts[None, ...], 16
            ).astype(np.int32).squeeze(0)
            inputs['r2p_ds_nei_idx%d'%i] = nei_r2p.copy()
            nei_p2r = DP.knn_search(
                sub_pts[None, ...], sr2dptxyz[rgb_ds_sr[i]][None, ...], 1
            ).astype(np.int32).squeeze(0)
            inputs['p2r_ds_nei_idx%d'%i] = nei_p2r.copy()
            cld = sub_pts

        item_dict = dict(
            rgb=rgb.astype(np.uint8),  # [c, h, w]
            cld_rgb_nrm=cld_rgb_nrm.astype(np.float32),  # [9, npts]
            choose=choose.astype(np.int32),  # [1, npts]
        )
        item_dict.update(inputs)
        return item_dict


    def __len__(self):
        return 1

    def __getitem__(self, idx=0):
            return self.get_item()
