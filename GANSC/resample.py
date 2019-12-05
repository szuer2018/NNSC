#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import librosa
import fnmatch
import numpy as np
import os
import soundfile as sf

import shutil


def find_files(directory, pattern=['*.wav', '*.WAV']):
    '''find files in the directory'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern[0]):
            files.append(os.path.join(root, filename))
        for filename in fnmatch.filter(filenames, pattern[1]):
            files.append(os.path.join(root, filename))
    return files


file_path = '/home/szuer/segan2017/data/bone_test/女声2_R.wav'
out_path = '/home/szuer/segan2017/data/bone_test/clean_test16k_bone'

print("starting to resample the utterances...")

file_name = file_path.split("/")[-1]
out_audio_file = os.path.join(out_path, file_name)
noise_44k, sr = librosa.load(file_path, sr=None)
noise_org = librosa.resample(noise_44k, sr, 16000)
sf.write(out_audio_file, noise_org, 16000, subtype='PCM_16')


print 'resampling done'