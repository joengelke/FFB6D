#!/bin/bash
tst_mdl=train_log/ycb/checkpoints/FFB6D_best.pth.tar
python3 -m singleInference -checkpoint $tst_mdl -dataset ycb
