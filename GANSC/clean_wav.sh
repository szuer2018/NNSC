#!/bin/bash


# guia file containing pointers to files to clean up
:<<EOF
if [ $# -lt 1 ]; then
    echo 'ERROR: at least wavname must be provided!'
    echo "Usage: $0 <guia_file> [optional:save_path]"
    echo "If no save_path is specified, clean file is saved in current dir"
    exit 1
fi

NOISY_WAVNAME="$1"
SAVE_PATH="."
if [ $# -gt 1 ]; then
  SAVE_PATH="$2"
fi
EOF

#l1_adv_loss or wassserstein

NOISY_WAVNAME="/home/szuer/interspeech2019/bl_test_16k/8000kHz/"
SAVE_PATH="/home/szuer/interspeech2019/bl_test_16k/enhanced20190824_adv/8000kHz"
mode='stage1'
model_save="/home/szuer/interspeech2019/saved_models_0322/interspeech2019/models_saved/BWE_lsgan_models"
echo "INPUT NOISY WAV: $NOISY_WAVNAME"
echo "SAVE PATH: $SAVE_PATH"
mkdir -p $SAVE_PATH
#l1 150 70700 wasser 100 88100
CUDA_VISIBLE_DEVICES=1 python main.py --init_noise_std 0. --save_path $model_save \
               --batch_size 100 --g_nl prelu --weights SEGAN-160000 \
               --preemph 0.97 --bias_deconv True \
               --bias_downconv True --bias_D_conv True \
               --test_wav $NOISY_WAVNAME --save_clean_path $SAVE_PATH \
               --loss_type l1_adv_loss
