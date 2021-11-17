#!/usr/bin/env python3
from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import os
import torch
import argparse
import numpy as np
import pickle as pkl
from PIL import Image
from common import Config, ConfigRandLA
from models.ffb6d import FFB6D
from datasets.ycb.ycb_image import YCB_IMAGE_PREPROC as YCB_Image
from utils.pvn3d_eval_utils_kpls import cal_frame_poses, cal_frame_poses_lm
from utils.basic_utils import Basic_Utils
from cv2 import imshow, waitKey

def load_checkpoint(model=None, optimizer=None, filename="checkpoint"):
    filename = "{}.pth.tar".format(filename)

    assert os.path.isfile(filename), "==> Checkpoint '{}' not found".format(filename)
    print("==> Loading from checkpoint '{}'".format(filename))
    try:
        checkpoint = torch.load(filename)
    except Exception:
        checkpoint = pkl.load(open(filename, "rb"))
    epoch = checkpoint.get("epoch", 0)
    it = checkpoint.get("it", 0.0)
    best_prec = checkpoint.get("best_prec", None)
    if model is not None and checkpoint["model_state"] is not None:
        ck_st = checkpoint['model_state']
        if 'module' in list(ck_st.keys())[0]:
            tmp_ck_st = {}
            for k, v in ck_st.items():
                tmp_ck_st[k.replace("module.", "")] = v
            ck_st = tmp_ck_st
        model.load_state_dict(ck_st)
    if optimizer is not None and checkpoint["optimizer_state"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    print("==> Done")
    return it, epoch, best_prec

def load_data(rgb_path, depth_path, camera):
    rgb_image = read_image(rgb_path)
    print("rgb_image", rgb_path, rgb_image.dtype)
    depth_image = read_image(depth_path)
    print("depth_image type",depth_path, depth_image.dtype)
    assert (rgb_image is not None) and (depth_image is not None), "Failed to read in images!"

    ds = YCB_Image(rgb_image=rgb_image, depth_image=depth_image, camera=camera)
    loader = torch.utils.data.DataLoader(ds, batch_size=1)
    return next(iter(loader))

def predict_poses(model, data, config):
    model.eval()
    with torch.set_grad_enabled(False):
        cu_dt = {}
        for key in data.keys():
            if data[key].dtype in [np.float32, np.uint8]:
                cu_dt[key] = torch.from_numpy(data[key].astype(np.float32)).cuda()
            elif data[key].dtype in [np.int32, np.uint32, np.uint16]:
                cu_dt[key] = torch.LongTensor(data[key].astype(np.int32)).cuda()
            elif data[key].dtype in [torch.uint8, torch.float32]:
                cu_dt[key] = data[key].float().cuda()
            elif data[key].dtype in [torch.int32, torch.int16]:
                cu_dt[key] = data[key].long().cuda()
        end_points = model(cu_dt)
        _, classes_rgbd = torch.max(end_points['pred_rgbd_segs'], 1)

        pcld = cu_dt['cld_rgb_nrm'][:, :3, :].permute(0, 2, 1).contiguous()
        predicted_ids, predicted_poses, _ = cal_frame_poses(
            pcld[0], classes_rgbd[0], end_points['pred_ctr_ofs'][0],
            end_points['pred_kp_ofs'][0], True, config.n_objects, True,
            None, None
        )

        rgb_image = cu_dt['rgb'].cpu().numpy().astype("uint8")[0].transpose(1, 2, 0).copy()

        return predicted_ids, predicted_poses, rgb_image

def show_predicted_poses(config, predicted_ids, predicted_poses, rgb_image, camera):

    bs_utils = Basic_Utils(config)

    np_rgb = rgb_image[:, :, ::-1].copy()

    for i, id in enumerate(predicted_ids):
        pose = predicted_poses[i]
        obj_id = int(id)
        mesh_pts = bs_utils.get_pointxyz(obj_id, ds_type="ycb").copy()
        mesh_pts = np.dot(mesh_pts, pose[:, :3].T) + pose[:, 3]

        if camera == 'ycb':
            K = config.intrinsic_matrix['ycb_K1']
        else:
            K = config.intrinsic_matrix[camera]

        mesh_p2ds = bs_utils.project_p3d(mesh_pts, 1.0, K)
        color = bs_utils.get_label_color(obj_id, n_obj=22, mode=2)
        np_rgb = bs_utils.draw_p2ds(np_rgb, mesh_p2ds, color=color)
    bgr = np_rgb

    imshow("projected_pose_rgb", bgr)
    waitKey()

def read_image(path):
    with Image.open(path) as im:
        return np.array(im)

def main():
    global DEBUG

    parser = argparse.ArgumentParser(description="Arg parser")
    parser.add_argument(
        "-checkpoint", type=str, default=None, help="Checkpoint to eval"
    )
    parser.add_argument(
        "-rgb", type=str,
        help="RGB Image Path"
    )
    parser.add_argument(
        "-depth", type=str,
        help="Depth Image Path"
    )
    parser.add_argument(
        "-camera", default='ycb', help="Camera Type from ['ycb','intel_l515']"
    )
    args = parser.parse_args()

    config = Config(ds_name="ycb")
    data = load_data(args.rgb, args.depth, args.camera)

    rndla_cfg = ConfigRandLA
    model = FFB6D(
        n_classes=config.n_objects, n_pts=config.n_sample_points, rndla_cfg=rndla_cfg,
        n_kps=config.n_keypoints
    )
    model.cuda()

    #load status from checkpoint
    if args.checkpoint is not None:
        load_checkpoint(
            model, None, filename=args.checkpoint[:-8]
        )

    predicted_ids, predicted_poses, rgb_image = predict_poses(model, data, config)
    print("poses:", predicted_poses)
    bs_utils = Basic_Utils(config)
    print("ids:",[bs_utils.get_cls_name(c.item(), "ycb") for c in predicted_ids])
    show_predicted_poses(config, predicted_ids, predicted_poses, rgb_image, args.camera)


if __name__ == "__main__":
    main()
