#!/bin/bash
tst_mdl=train_log/ycb/checkpoints/FFB6D_best.pth.tar
python3 -m single_ycb_inference -checkpoint $tst_mdl -rgb "/home/tmarkmann/depth_image/2-color.png" -depth "/home/tmarkmann/depth_image/2-depth.png" -camera ycb
