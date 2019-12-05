#!/bin/bash

# Place the CUDA_VISIBLE_DEVICES="xxxx" required before the python call
# e.g. to specify the first two GPUs in your system: CUDA_VISIBLE_DEVICES="0,1" python ...


# Apply pre-emphasis AND apply biases to all conv layers (best SEGAN atm) wasserstein 100, l1_loss 150
mode='stage2'
save_path="bwe_wavform100/l1_adv_loss/$mode/segan_allbiased_preemph"
#batchsize for wasserstein is 85 l1_adv_loss
CUDA_VISIBLE_DEVICES=1 python main.py --init_noise_std 0. --save_path $save_path \
                                          --init_l1_weight 100. --batch_size 100 --g_nl prelu \
                                          --save_freq 5000 --preemph 0.97 --epoch 86 --bias_deconv True \
                                          --bias_downconv True --bias_D_conv True --loss_type 'l1_adv_loss'
