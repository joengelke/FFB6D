#!/bin/bash
base=/home/lruegeme/projects/6dof/tiago-object-recognition-citk/images/
tst_mdl=train_log/ycb/checkpoints/FFB6D_best.pth.tar
python3 -m single_ycb_inference -checkpoint $tst_mdl -rgb $base"xtion/rgb/image_rect_color0000.png" -depth $base"xtion/depth_registered/image0000.png" -camera ycb
python3 -m single_ycb_inference -checkpoint $tst_mdl -rgb $base"/rs/color/image_rect_color0000.png" -depth $base"/rs/depth/aligned_raw0000.png" -camera intel_l515
