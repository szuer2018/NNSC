#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 16:24:20 2018

@author: Yupeng Shi

Summary:  Recover spectrogram/timedomain frames to wave.
"""

import numpy as np
import librosa
 
def time_recover_wav(pd_abs_x, n_window, n_hop, winfunc):
    
    IN_sk2 = np.fft.rfft(pd_abs_x).astype(np.complex64)

    in_sk = librosa.core.istft(IN_sk2.T, hop_length=n_hop, win_length=n_window, window=winfunc, center=False)
    
    in_sk /= abs(in_sk).max()
    #if max(abs(in_sk)) > 1:
     #   in_sk = in_sk*0.9/max(abs(in_sk))
        
    return in_sk*0.9

def spectra_to_wav(pd_abs_x, gt_x, n_window, n_hop, winfunc):
    x = real_to_complex(pd_abs_x, gt_x)
    frames = librosa.core.istft(x.T, hop_length=n_hop, win_length=n_window, window=winfunc, center=False)
    
    #if max(abs(frames)) > 1:
    #    frames = frames*0.9/max(abs(frames))
    frames /= abs(frames).max()
    
    return frames*0.45
    

def real_to_complex(pd_abs_x, gt_x):
    
    theta = np.angle(gt_x)
    theta_new = -1*np.flipud((theta.T)[0:140, :])
    theta[:, 140:280] = theta_new.T
    # pd_abs_x[:,0:140] = np.abs(gt_x)[:,0:140]
    cmplx = pd_abs_x * np.exp(1j * theta)
    return cmplx