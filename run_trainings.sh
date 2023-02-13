#!/usr/bin/env bash

base_config="/home/daniel/PycharmProjects/AdelaiDet_gaze/configs/BoxInst/MS_R_50_1x_VOC2012_gaze.yaml"
no_empty_mask="True"
baseline="0.0 0.1 0.3 0.5"
iters="10000 15000 20000"

for em in $no_empty_mask; do
  for bl in $baseline; do
    for it in $iters; do
      echo "python" /home/daniel/PycharmProjects/AdelaiDet_gaze/tools/train_net.py --config-file  $base_config --num-gpus 2 OUTPUT_DIR /home/daniel/PycharmProjects/AdelaiDet_gaze/training_dir/loss_conf_sweep_2/BoxInst_empty_mask_"$em"_baseline_"$bl"_iters_"$it"/ MODEL.GAZEINST.GAZE_LOSS_NO_EMPTY_MASK $em MODEL.GAZEINST.GAZE_LOSS_LABEL crf MODEL.GAZEINST.GAZE_LOSS_COOLDOWN_BASELINE $bl MODEL.GAZEINST.GAZE_LOSS_COOLDOWN_ITERS $it MODEL.GAZEINST.GAZE_LOSS_LABEL dice
    done;
 done;
done;
