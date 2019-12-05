"""
Summary:  Config file. 
Author:   Yupeng Shi
Created:  2018.07.20
Modified: -
"""

import numpy as np
#sample_rate = 16000
sample_rate=16000
TF = 'spectrogram'   #(spectrogram, timedomain, fftmagnitude)
SNR = [-5., 0., 5., 10.] # -6., -3., 0., 3., 6., 9., 12. -10., -5., 0., 5., 10.

if TF ==  'spectrogram':
    
    if sample_rate == 8000:
        n_window = 256      # windows size for each frame 32ms
        n_overlap = int(0.75*n_window)     # overlap between frames
    elif sample_rate == 16000:
        n_window = 640      # windows size for each frame 40ms
        n_overlap = int(0.5*n_window)     # overlap between frames
    elif sample_rate == 44100:
        n_window = 1024      # windows size for each frame about 23ms
        n_overlap = int(0.5*n_window)  

elif TF == 'fftmagnitude':
    #n_window = 256      # windows size for each frame 32ms
    #n_overlap = int(0.75*n_window)     # overlap between frames
    if sample_rate == 8000:
        n_window = 256      # windows size for each frame 32ms
        n_overlap = int(0.75*n_window)     # overlap between frames
    elif sample_rate == 16000:
        n_window = 512      # windows size for each frame 32ms
        n_overlap = int(0.75*n_window)     # overlap between frames
    
elif TF == 'timedomain':
    
    #n_window = 160      # windows size for each frame 20ms
    #n_overlap = int(0.75*n_window)     # overlap between frames
    if sample_rate == 8000:
        n_window = 256      # windows size for each frame 32ms
        n_overlap = int(0.75*n_window)     # overlap between frames
    elif sample_rate == 16000:
        n_window = 512      # windows size for each frame 32ms
        n_overlap = int(0.75*n_window)     # overlap between frames
'''
SNR = [-6., -3., 0., 3., 6., 9., 12.]
rs = np.random.RandomState()
selected_snr = rs.choice(SNR, size=1, replace=False)

'''