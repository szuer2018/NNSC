#!/bin/bash
# -*- coding: utf-8 -*-

# DOWNLOAD THE DATASET
#mkdir -p data  # create a dir ./data
#pushd data

data_dir='/media/szuer/f501afb9-d15c-408e-a0c8-eef55f887738/szuer/SE_DATASOURCE/DS_10283_2791'
cd $data_dir

mkdir -p $data_dir/data
#:<< EOF
cd $data_dir/data

if [ ! -d clean_trainset_56spk_wav_16k ]; then
    # Clean utterances
    if [ ! -f clean_trainset_56spk_wav.zip ]; then
        echo 'DOWNLOADING CLEAN DATASET...'
        wget http://datashare.is.ed.ac.uk/bitstream/handle/10283/1942/clean_trainset_wav.zip
    fi
    if [ ! -d clean_trainset_56spk_wav ]; then
        echo 'INFLATING CLEAN TRAINSET ZIP...'
        unzip -q clean_trainset_56spk_wav.zip 
    fi
    if [ ! -d clean_trainset_56spk_wav_16k ]; then
        echo 'CONVERTING CLEAN WAVS TO 16K...'
        mkdir -p clean_trainset_56spk_wav_16k

        #pushd clean_trainset_wav
	cd ./clean_trainset_56spk_wav

        ls *.wav | while read name; do
            sox $name -r 16k ../clean_trainset_56spk_wav_16k/$name
        done
        #popd
	cd ..
    fi
fi
if [ ! -d noisy_trainset_56spk_wav_16k ]; then
    # Noisy utterances
    if [ ! -f noisy_trainset_56spk_wav.zip ]; then
        echo 'DOWNLOADING NOISY DATASET...'
        wget http://datashare.is.ed.ac.uk/bitstream/handle/10283/1942/noisy_trainset_wav.zip
    fi
    if [ ! -d noisy_trainset_56spk_wav ]; then
        echo 'INFLATING NOISY TRAINSET ZIP...'
        unzip -q noisy_trainset_56spk_wav.zip 
    fi
    if [ ! -d noisy_trainset_56spk_wav_16k ]; then
        echo 'CONVERTING NOISY WAVS TO 16K...'
        mkdir -p noisy_trainset_56spk_wav_16k
        #pushd noisy_trainset_wav
	cd ./noisy_trainset_56spk_wav
        ls *.wav | while read name; do
            sox $name -r 16k ../noisy_trainset_56spk_wav_16k/$name
        done
        #popd
	cd ..
    fi
fi
#popd
:<<EOF
cd /home/szuer/SEGAN2018

echo 'PREPARING TRAINING DATA...'
python make_tfrecords.py --force-gen --cfg cfg/e2e_maker.cfg

cd $data_dir/data/clean_trainset_wav


        ls *.wav | while read name; do
	    echo "Down-sample $name to 16kHz..."
            sox $name -r 16k ../clean_trainset_wav_16k/$name
        done

cd $data_dir/data/noisy_trainset_wav
        ls *.wav | while read name; do
	    echo "Down-sample $name to 16kHz..."
            sox $name -r 16k ../noisy_trainset_wav_16k/$name
        done
EOF

#EOF
