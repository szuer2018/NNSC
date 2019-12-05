#!/bin/bash
#Author Yupeng Shi
#Time 2019.03,14
#Modified by Yupeng Shi
# this script modified for interspeech 2019 band width extension
#./runse.sh | tee -a ./**_results.log

echo ============================================================================
echo "             Speech Enhancement with neural networks（CNN | DNN | RNN）                 "
echo ============================================================================

cat ./README.txt

TF_SET='spectrogram' # 1 presents log-spectra features, 0 presents time-domain features(spectrogram, timedomain)
#NOISE_MODE='bwe'
NOISE_STYLE="ds" #stationary fluctuating

stage=6




#if [ $stage -le -1 ]; then
homedir=$(pwd)

if [ $TF_SET = spectrogram ]; then
  WORKSPACE="$homedir/$NOISE_STYLE/$TF_SET"
  if [ -e $WORKSPACE ]; then
	echo "$WORKSPACE already exist" 
  else
	echo "$WORKSPACE no exist, now creating it"
	mkdir -p $WORKSPACE
	#echo "please check carefully and make sure to remove all files in $WORKSPACE"
      	#rm -rf $WORKSPACE && mkdir $WORKSPACE     
	#echo "$WORKSPACE already removed and created again" 	
  fi
  
  SPEECH_DIR='/home/szuer/CWGAN/bwe/icassp2020/raw_wav/thch30'
#  TR_NOISY_DIR="/home/szuer/CWGAN/bwe/icassp2020/raw_wav/ds/train_label"
  #TE_SPEECH_DIR=$TE_VAD_OUPUT
  #TE_NOISE_DIR="/home/szuer/speech_enhance/journal2019_noises/16kHz/mixed"
  echo "Using $TF_SET features. "
else
  WORKSPACE="$homedir/workspace_$train_utterances/$TF_SET/$NOISE_STYLE"
  if [ -e $WORKSPACE ]; then
	echo "$WORKSPACE already exist" 
  else
	echo "$WORKSPACE no exist, now creating it"
	mkdir -p $WORKSPACE
	#echo "please check carefully and make sure to remove all files in $WORKSPACE"
      	#rm -rf $WORKSPACE && mkdir $WORKSPACE     
	#echo "$WORKSPACE already removed and created again" 	
  fi
  TR_SPEECH_DIR="/home/szuer/journal2018/speechdata1000_16kHz/train"
  TR_NOISE_DIR="/home/szuer/speech_enhance/ADF_TEST/Nonspeech/stationary"
  TE_SPEECH_DIR="/home/szuer/journal2019/speechdata1000_16kHz/test"
  TE_NOISE_DIR="/home/szuer/speech_enhance/ADF_TEST/Nonspeech/stationary"
  echo "Using $TF_SET features. "
fi

#fi



if [ $stage -le 1 ]; then
# Calculate mixture features.
echo ===================================================================================================
echo "         Calculate $TF features for wideband and narrowband audio           "
echo ===================================================================================================
python prepare_data.py calculate_features --TF=$TF_SET --workspace=$WORKSPACE --speech_dir=$SPEECH_DIR --data_type=train
python prepare_data.py calculate_features --TF=$TF_SET --workspace=$WORKSPACE --speech_dir=$SPEECH_DIR --data_type=val
fi

# Pack features. 
N_CONCAT=9
N_HOP=4
#WORKSPACE="/home/szuer/SE_CNN_DNN/timedomain/workspace"
if [ $stage -le 2 ]; then
echo ============================================================================
echo "             Load all features, apply log and conver to 3D tensor                 "
echo ============================================================================ 
python prepare_data.py pack_features --TF=$TF_SET --workspace=$WORKSPACE --data_type=train --n_concat=$N_CONCAT --n_hop=$N_HOP
python prepare_data.py pack_features --TF=$TF_SET --workspace=$WORKSPACE --data_type=val --n_concat=$N_CONCAT --n_hop=$N_HOP
fi

if [ $stage -le 3 ]; then
# Compute scaler. 
echo ============================================================================
echo "            Whitening the features (using the mean and var of training data to normalize train dataset)                 "
echo ============================================================================ 
python prepare_data.py compute_scaler --workspace=$WORKSPACE --data_type=train
fi
# Train. 
#LEARNING_RATE=1e-4
LEARNING_RATE=0.0002
#model_name='sdnn1'
#WORKSPACE="/home/szuer/SE_CNN_DNN/workspace_babble"
#CUDA_VISIBLE_DEVICES=0 
if [ $stage -le 5 ]; then
echo ============================================================================
echo "             Training DNN for several epochs                 "
echo ============================================================================ 

  for m in 'dnn' ; do
	echo "training the $m model..."
	CUDA_VISIBLE_DEVICES=0 python main_dnn.py train --model_name=$m --workspace=$WORKSPACE --lr=$LEARNING_RATE
  done

fi

if [ $stage -le -2 ]; then
echo ============================================================================
echo "             Plot the train | test loss graph                 "
echo ============================================================================ 
# Plot training stat. 
python evaluate.py plot_training_stat --workspace=$WORKSPACE --tr_snr=$TR_SNR --bgn_iter=0 --fin_iter=10001 --interval_iter=1000
fi
# Inference, enhanced wavs will be created. 
#ITERATION=10000
#WORKSPACE="/home/szuer/SE_CNN_DNN/workspace2"
if [ $stage -le 6 ]; then
echo ============================================================================
echo "             Test the system                 "
echo ============================================================================ 
#CUDA_VISIBLE_DEVICES=3
#  python prepare_data.py calculate_features --TF=$TF_SET --workspace=$WORKSPACE --speech_dir=$SPEECH_DIR --data_type=test
  for m in 'dnn'; do
	echo "test the $m model..."
	CUDA_VISIBLE_DEVICES=0 python main_dnn.py inference --TF=$TF_SET --model_name=$m --workspace=$WORKSPACE --n_concat=$N_CONCAT
  done
fi

te_enh='enhance'
if [ $stage -le -7 ]; then
# Calculate PESQ of all enhanced speech. 
python evaluate.py calculate_pesq --workspace=$WORKSPACE --speech_dir=$TE_SPEECH_DIR --model_name=$model_name --te_enh=$te_enh
fi

if [ $stage -le -8 ]; then

fileName='/home/szuer/journal2018/PESQ_2018_new/score.m'
ref_speech_dir=$TE_SPEECH_DIR
enh_speech_dir="$WORKSPACE/enh_wavs/test/$model_name"
echo "scoring for PESQ, LSD, STOI and segSNR, please wait..."
matlab -nodesktop -nosplash -nojvm -r "run $fileName;quit;"
#matlab -nodesktop -nosplash -nojvm -r "score('$ref_speech_dir', ‘$enh_speech_dir’); quit;"
#matlab -nodesktop -nosplash -r  "enh_speech_dir='$enh_speech_dir',ref_speech_dir='$ref_speech_dir'"
echo "scoring done successfully!"

fi

score_results_dir="/home/szuer/journal2018/workspace_16kHz_1000/spectrogram/noise8/stationary/score_results"
if [ $stage -le -9 ]; then
# Calculate overall stats. 
#python evaluate.py get_stats >> $score_results_dir/scores_results.txt
python evaluate.py get_standars >> $score_results_dir/standar_scores_results.txt
echo "results were saved to $score_results_dir!"
# >> $pesq_results_dir/pesq_sdnn1_results.txt
fi


#EOF
exit 0;
