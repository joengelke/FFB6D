#!/bin/bash
base=/home/lruegeme/projects/6dof/tiago-object-recognition-citk/images/
tst_mdl=train_log/ycb/checkpoints/FFB6D_best.pth.tar

python3 -m single_ycb_inference -checkpoint $tst_mdl -rgb $base"/ycb/000001-color.png" -depth $base"/ycb/000001-depth.png" -camera ycb
printf '\n\n##########\n\n'
python3 -m single_ycb_inference -checkpoint $tst_mdl -rgb $base"xtion/rgb/image_rect_color0000_converted.png" -depth $base"xtion/depth_registered/hw_registered-image_rect0000.png" -camera ycb
printf '\n\n##########\n\n'
python3 -m single_ycb_inference -checkpoint $tst_mdl -rgb $base"xtion/rgb/image_rect_color0000_converted.png" -depth $base"xtion/depth_registered/image0000_converted.png" -camera ycb
printf "\n\n##########\n\n"
python3 -m single_ycb_inference -checkpoint $tst_mdl -rgb $base"/rs/color/image_rect_color0000_converted.png" -depth $base"/rs/depth/aligned_raw0000.png" -camera intel_l515
echo -e "\n\n##########\n\n"
python3 -m single_ycb_inference -checkpoint $tst_mdl -rgb $base"/rs/color/image_rect_color0000_converted.png" -depth $base"/rs/depth/aligned_raw0000_converted.png" -camera intel_l515
printf '\n\n##########\n\n'
python3 -m single_ycb_inference -checkpoint $tst_mdl -rgb $base"xtion/rgb/image_rect_color0000_converted.png" -depth $base"xtion/depth_registered/image0000_converted.png" -camera ycb

