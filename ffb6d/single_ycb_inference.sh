#!/bin/bash
base=/home/bv6dof/Desktop/GitLab/2022-6dof-object/images/
tst_mdl=train_log/ycb/checkpoints/FFB6D_best.pth.tar

#python3 -m single_ycb_inference -checkpoint $tst_mdl -rgb $base"v4/ycb/000010-color.png" -depth $base"v4/ycb/000010-depth.png" -camera ycb
printf '\n\n##########\n\n'
python3 -m single_ycb_inference -checkpoint $tst_mdl -rgb $base"v2/xtion/rgb/image_raw0000.png" -depth $base"v2/xtion/depth/image_raw0000.png" -camera xtion

#printf "\n\n##########\n\n"
#python3 -m single_ycb_inference -checkpoint $tst_mdl -rgb $base"/rs/color/image_rect_color0000_converted.png" -depth $base"/rs/depth/aligned_raw0000.png" -camera intel_l515
#printf '\n\n##########\n\n'
#python3 -m single_ycb_inference -checkpoint $tst_mdl -rgb $base"xtion/rgb/image_rect_color0000_converted.png" -depth $base"xtion/depth_registered/image0000.png" -camera xtion
