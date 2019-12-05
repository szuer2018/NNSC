#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 18:37:33 2018

@author: szuer
summary: this script used to simulate the bone transform, below 2kHz
"""

import librosa
import fnmatch
import numpy as np
import os
import soundfile as sf
import obspy.signal.filter
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


def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)


noise_dir = '/media/szuer/8e1f3765-286d-4d57-a4d7-d64aa2a176e0/home/amax/boneloss_rawdata/clean_trainset_56spk_wav_16k'
clean_dir = '/media/szuer/Elements/clean_testset16k'
# noise_len='/home/szuer/speech_enhance/noise_audio_60/1-9841-A-13.wav'
# oise_13, sr=librosa.load(noise_len, sr=None)
out_file_path = '/home/szuer/PLGAN/rawdata/trainset_56spk_freqloss'
txt_source_dir = '/home/szuer/PLGAN/rawdata/trainset_56spk_txt'
txt_dest_dir = '/home/szuer/PLGAN/boneloss_data/train_text'
# create_folder(txt_dest_path)
noisefiles = find_files(noise_dir)
N_noise_files = len(noisefiles)
fc = [1000,2000,3000]
rs = np.random.RandomState()
# pass_fc = rs.choice(fc, size=1, replace=False)[0]
i = 0
'''
for f in fc:
    for files in noisefiles:
        # pass_fc = rs.choice(fc, size=1, replace=False)[0]
        file_name = files.split("/")[-1]
        out_audio_file = os.path.join(out_file_path, "%dkHz" % f, file_name)
        create_folder(os.path.dirname(out_audio_file))
        y, sr = librosa.load(files, sr=None)
        print("apply low_pass {}kHz to {}".format(f, file_name))
        noise_org = obspy.signal.filter.lowpass(y, f, sr, corners=8)
        sf.write(out_audio_file, noise_org, sr, subtype='PCM_16')
'''
for files in noisefiles:
    pass_fc = rs.choice(fc, size=1, replace=False)[0]
    file_name = files.split("/")[-1]
    out_audio_file = os.path.join(out_file_path, file_name)
    create_folder(os.path.dirname(out_audio_file))
    y, sr = librosa.load(files, sr=None)
    print("apply low_pass {}kHz to {}".format(pass_fc, file_name))
    noise_org = obspy.signal.filter.lowpass(y, pass_fc, sr, corners=8)
    sf.write(out_audio_file, noise_org, sr, subtype='PCM_16')

    txt_name = file_name.split(".")[0] + ".txt"
    out_txt_file = os.path.join(txt_dest_dir, '%dkHz' % pass_fc)
    create_folder(out_txt_file)

    txt_source_file = os.path.join(txt_source_dir, txt_name)
    print("copy text file {} from {} to {}".format(txt_name, txt_source_dir, out_txt_file))
    shutil.copy(txt_source_file, out_txt_file)

    #out_clean_file = os.path.join(clean_dir, '%dkHz' % pass_fc)
    #create_folder(out_clean_file)
    #shutil.copy(files, out_clean_file)

    i = i + 1
print("processing {} utterances".format(i))
print 'done successfully'

