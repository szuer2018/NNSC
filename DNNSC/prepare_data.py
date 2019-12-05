# -*- coding: utf-8 -*-
"""
Summary:  Prepare data. 
Modified: Yupeng Shi
time:     2019.09.15
for icassp 2020 as a baseline system
"""
import os
import soundfile as sf
import numpy as np
from numpy.lib import stride_tricks
import argparse
import csv
import time
import matplotlib.pyplot as plt
from scipy import signal
import pickle
import cPickle

from sklearn import preprocessing
import librosa
from pydub import AudioSegment
import prepare_data as pp_data
import config as cfg
import fnmatch
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import h5py
import scipy
import scipy.io.wavfile as wav


def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)
        
def read_audio(path, target_fs=None):
    (audio, fs) = sf.read(path)
    #transfer multi-channel to single-channel
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    #down-sample the audio to target sample rate, 8kHz
    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
    return audio, fs
    
def write_audio(path, audio, sample_rate):
    sf.write(file=path, data=audio, samplerate=sample_rate)
    #wav.write(path, sample_rate, np.int16(audio * 32767))
    
def find_files(directory, pattern=['*.wav', '*.WAV']):
    '''find files in the directory'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern[0]):
            files.append(os.path.join(root, filename))
        for filename in fnmatch.filter(filenames, pattern[1]):
            files.append(os.path.join(root, filename))
    return files

###
    
###
def calculate_features(args):
    """
    Calculate spectrogram or time-domain features for mixed, speech and noise audio. Then write the 
    features to disk. 
    
    Args:
      workspace: str, path of workspace. 
      speech_dir: str, path of speech data. 

      data_type: str, 'train' | 'test'. 
      snr: float, signal to noise ratio to be mixed. 
    """
    workspace = args.workspace
    speech_dir = args.speech_dir
    # noisy_dir = args.noisy_dir
    data_type = args.data_type


    TF = args.TF
    cwav_path = os.path.join(speech_dir, data_type+'_label')
    nwav_path = os.path.join(speech_dir, data_type)
    wavfiles = os.listdir(cwav_path)
    N_wav_files = len(wavfiles)
    print("dectectin {} utterances in {}".format(N_wav_files, cwav_path))
    
    t1 = time.time()
    cnt = 0
    for files in wavfiles:

        # file_name = files.split("/")[-1]
        out_bare_na = files.split(".")[0]
        cwav, sr = librosa.load(os.path.join(cwav_path, files), None)
        nwav, sr = librosa.load(os.path.join(nwav_path, files), None)
        if sr != 16000:
            raise ValueError("The sample rate expected to be 16kHz!")
        # check the wav channels
        channels1 = len(nwav.shape)
        channels2 = len(cwav.shape)

        if channels1 != 1 or channels2 != 1:
            raise ValueError("Expected to be mono-channel!")
            # print('WARNING: stereo to mono: ' + data_folder + wav_lst[snt_id_arr[i]])
            # signal = signal[:, 0]

        if len(nwav) != len(cwav):
            # print("the ground truth and the corrupted speech mismatch in length!!")
            # print("force alignment operation...")
            min_len = len(nwav)
            if len(cwav) < min_len:
                min_len = len(cwav)
            nwav = nwav[0:min_len]
            cwav = cwav[0:min_len]
        # print("min ok")

        assert len(nwav) == len(cwav)

        if TF == "spectrogram":
            # Extract spectrogram. 
            # print("Extracting log-spectra frames of the speech data--{}".format(os.path.join(cwav_path, files)))
            #speech_x = calc_sp(y_clean, mode='magnitude')
            cwav_complx = calc_sp(cwav, mode='complex')
            # print("processing the narrow speech--{}".format(os.path.join(nwav_path, files)))
            nwav_complx = calc_sp(nwav, mode='complex')
            #noise_x = calc_sp(noise_audio, mode='magnitude')

        else:
            raise Exception("TF must be spectrogram, timedomain or fftmagnitude!")
            

        # Write out features. 

        out_feat_path = os.path.join(workspace, "features",
                                         data_type, "%s.p" % out_bare_na)
        
        create_folder(os.path.dirname(out_feat_path))
        data = [nwav_complx, cwav_complx]
        cPickle.dump(data, open(out_feat_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)
        
        # Print. 
        if cnt % 1000 == 0:
            print("processed {} utterances and remaining {} utternaces" .format(cnt, N_wav_files-cnt))
            
        cnt += 1

    print("Extracting feature time: %f s" % (time.time() - t1))
    

    
def calc_sp(audio, mode):
    """Calculate spectrogram. 
    
    Args:
      audio: 1D array. 
      mode: string, 'magnitude' | 'complex'
    
    Returns:
      spectrogram: 2darray, (n_time, n_freq). 
    """
    n_window = cfg.n_window
    n_overlap = cfg.n_overlap
    #ham_win = np.hamming(n_window)
    ham_win = scipy.signal.hamming(n_window)
    [f, t, x] = signal.spectral.spectrogram(
                    audio, 
                    window=ham_win,
                    nperseg=n_window, 
                    noverlap=n_overlap, 
                    detrend=False, 
                    return_onesided=True, 
                    mode=mode) 
    x = x.T
    if mode == 'magnitude':
        x = x.astype(np.float32)
    elif mode == 'complex':
        x = x.astype(np.complex64)
    else:
        raise Exception("Incorrect mode!")
    return x
###

def frame_func(sig, frameSize, overLap, window=np.hamming):
    """ short time fourier transform of audio signal """
    win = window(frameSize)
    hopSize = frameSize-overLap
    #samples = np.array(sig, dtype='float64')
    #cols = int(np.ceil((len(sig) - frameSize) / float(hopSize)))
    
    cols = int(np.floor((len(sig) - frameSize) / float(hopSize))) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(sig, np.zeros(frameSize))
    frames = stride_tricks.as_strided(
        samples,
        shape=(cols, frameSize),
        strides=(samples.strides[0] * int(hopSize), samples.strides[0])).copy()
    frames *= win
    
    return frames
    
###
def pack_features(args):
    """Load all features, apply log and conver to 3D tensor, write out to .h5 file. 
    
    Args:
      workspace: str, path of workspace. 
      data_type: str, 'train' | 'test'. 
      snr: float, signal to noise ratio to be mixed. 
      n_concat: int, number of frames to be concatenated. 
      n_hop: int, hop frames(frame shift). 
    """
    workspace = args.workspace
    data_type = args.data_type
    n_concat = args.n_concat
    n_hop = args.n_hop
    TF = args.TF
    
    x_all = []  # (n_segs, n_concat, n_freq)

    y_all = []  # (n_segs, n_freq)

    
    cnt = 0
    t1 = time.time()
    
    # Load all features. data = [mixed_complx_x, speech_x, noise_x, alpha, out_bare_na]

    feat_dir = os.path.join(workspace, "features", data_type)
            
    #feat_dir = os.path.join(workspace, "features", data_type)
    names = os.listdir(feat_dir)
    for na in names:
        # Load feature. 
        feat_path = os.path.join(feat_dir, na)
        data = cPickle.load(open(feat_path, 'rb'))
        [mixed_complx_x, speech_complx_x] = data
        
        n_pad = (n_concat - 1) / 2
        speech_x = pad_with_border(np.abs(speech_complx_x), n_pad)
        # speech_phase = pad_with_border(np.angle(speech_complx_x), n_pad) #phase = np.exp(1j * np.angle(D))
        '''
        mixed_x = np.abs(mixed_complx_x)

        # Pad start and finish of the spectrogram with boarder values. 
        # mixed_x.shape=frames x framesize
        n_pad = (n_concat - 1) / 2
        mixed_x = pad_with_border(mixed_x, n_pad)
        speech_x = pad_with_border(speech_x, n_pad)
    
        # Cut input spectrogram to 3D segments with n_concat. 
        mixed_x_3d = mat_2d_to_3d(mixed_x, agg_num=n_concat, hop=n_hop)
        #x_all.append(mixed_x_3d)
        '''
        
        if TF == "spectrogram":
            mixed_x = np.abs(mixed_complx_x)
            # mixed_phase = np.angle(mixed_complx_x)

            # Pad start and finish of the spectrogram with boarder values. 
            # mixed_x.shape=frames x framesize
            #n_pad = (n_concat - 1) / 2
            mixed_x = pad_with_border(mixed_x, n_pad)
            # mixed_phase = pad_with_border(mixed_phase, n_pad)
            #speech_x = pad_with_border(speech_x, n_pad)
    
            # Cut input spectrogram to 3D segments with n_concat. 
            mixed_x_3d = mat_2d_to_3d(mixed_x, agg_num=n_concat, hop=n_hop)
            # mixed_phase_3d = mat_2d_to_3d(mixed_phase, agg_num=n_concat, hop=n_hop)
            log_x = log_sp(mixed_x_3d.copy())
            x_all.append(log_x)
            # x_phase.append(mixed_phase_3d)

        else:
            raise Exception("TF must be spectrogram, timedomain or fftmagnitude!")
        # Cut target spectrogram and take the center frame of each 3D segment. 
        speech_x_3d = mat_2d_to_3d(speech_x, agg_num=n_concat, hop=n_hop)
        # speech_phase_3d = mat_2d_to_3d(speech_phase, agg_num=n_concat, hop=n_hop)
        y = speech_x_3d[:, (n_concat - 1) / 2, :]
        # y_pha = speech_phase_3d[:, (n_concat - 1) / 2, :]
        if TF == "spectrogram":
            print("convert to log-spectra features")
            log_y = log_sp(y.copy())
            y_all.append(log_y)
            # y_phase.append(y_pha)

            
        else:
            raise Exception("TF must be spectrogram, timedomain or fftmagnitude!")
    
        # Print. 
        if cnt % 1000 == 0:
            print("Packing all the {} utterances to 3D tensor, remaining {} utterances".format(cnt, len(names)-cnt))
            
        # if cnt == 3: break
        cnt += 1
        
    nwav_all = np.concatenate(x_all, axis=0)   # transfer to array(n_segs, n_concat, n_freq)
    cwav_all = np.concatenate(y_all, axis=0)   # (n_segs, n_freq)
    
    # x_phase = np.concatenate(x_phase, axis=0)   # transfer to array(n_segs, n_concat, n_freq)
    # y_phase = np.concatenate(y_phase, axis=0)
    
    nwav_all = nwav_all.astype(np.float32)
    cwav_all = cwav_all.astype(np.float32)
    # x_phase = x_phase.astype(np.float32)
    # y_phase = y_phase.astype(np.float32)
    '''
    if TF == "spectrogram":
        x_all = log_sp(x_all).astype(np.float32)
        y_all = log_sp(y_all).astype(np.float32)
    elif TF == "timedomain":
        x_all = x_all.astype(np.float32)
        y_all = y_all.astype(np.float32)
    else:
        raise Exception("TF must be spectrogram or timedomain!")
    '''
    # Write out data to .h5 file. 

    mag_out_path = os.path.join(workspace, "packed_features", data_type, "mag.h5")
    # phase_out_path = os.path.join(workspace, "packed_features", data_type, "phase.h5")
            
    #out_path = os.path.join(workspace, "packed_features", data_type, "data.h5")
    create_folder(os.path.dirname(mag_out_path))
    with h5py.File(mag_out_path, 'w') as hf:
        hf.create_dataset('x', data=nwav_all)
        hf.create_dataset('y', data=cwav_all)
    
    print("Write out to %s" % mag_out_path)

    print("Pack features finished! %f s" % (time.time() - t1))
    
def log_sp(x):
    return 10*np.log10(abs(x) + 1e-10)
    #return np.log(x + 1e-10)
'''
def mat_2d_to_3d(x, agg_num, hop):
    """Segment 2D array to 3D segments. 
    """
    # Pad to at least one block. 
    #len_x is total frames of each utterance, n_in is frame length
    len_x, n_in = x.shape
    if (len_x < agg_num):
        x = np.concatenate((x, np.zeros((agg_num - len_x, n_in))))
        
    # Segment 2d to 3d. 
    len_x = len(x)
    #i1 = 0
    i1 = hop
    i2 = 0
    x3d = []
    while (i2 <= len_x):
        x3d.append(x[i1 - 3 : i1 + 4])
        i1 += 1
        i2 = 1
    return np.array(x3d)
'''

def mat_2d_to_3d(x, agg_num, hop):
    """Segment 2D array to 3D segments. 
    """
    # Pad to at least one block. 
    #len_x is total frames of each utterance, n_in is frame length
    len_x, n_in = x.shape
    if (len_x < agg_num):
        x = np.concatenate((x, np.zeros((agg_num - len_x, n_in))))
        
    # Segment 2d to 3d. 
    len_x = len(x)
    i1 = 0
    x3d = []
    while (i1 + agg_num <= len_x):
        x3d.append(x[i1 : i1 + agg_num])
        i1 += hop
    return np.array(x3d)

def pad_with_border(x, n_pad):
    """Pad the begin and finish of spectrogram with border frame value. 
    """
    x_pad_list = [x[0:1]] * n_pad + [x] + [x[-1:]] * n_pad
    return np.concatenate(x_pad_list, axis=0)

###
def compute_scaler(args):
    """Compute and write out scaler of data. 
    """
    workspace = args.workspace
    data_type = args.data_type
    #snr = args.snr
    
    # Load data. 
    t1 = time.time()
    hdf5_path = os.path.join(workspace, "packed_features", data_type, "mag.h5")
    with h5py.File(hdf5_path, 'r') as hf:
        x = hf.get('x')     
        x = np.array(x)     # (n_segs, n_concat, n_freq)
    
    # Compute scaler. 
    (n_segs, n_concat, n_freq) = x.shape
    x2d = x.reshape((n_segs * n_concat, n_freq))
    scaler = preprocessing.StandardScaler(with_mean=True, with_std=True).fit(x2d)
    # print(scaler.mean_)
    # print(scaler.scale_)
    
    # Write out scaler. 
    out_path = os.path.join(workspace, "packed_features", data_type, "mag_scaler.p")
    create_folder(os.path.dirname(out_path))
    pickle.dump(scaler, open(out_path, 'wb'))
    
    print("Save scaler to %s" % out_path)
    print("Compute scaler finished! %f s" % (time.time() - t1))
    
def scale_on_2d(x2d, scaler):
    """Scale 2D array data. 
    """
    return scaler.transform(x2d)
    
def scale_on_3d(x3d, scaler):
    """Scale 3D array data. 
    """
    (n_segs, n_concat, n_freq) = x3d.shape
    x2d = x3d.reshape((n_segs * n_concat, n_freq))
    x2d = scaler.transform(x2d)
    x3d = x2d.reshape((n_segs, n_concat, n_freq))
    return x3d
    
def inverse_scale_on_2d(x2d, scaler):
    """Inverse scale 2D array data. 
    """
    return x2d * scaler.scale_[None, :] + scaler.mean_[None, :]
    
###
def load_hdf5(hdf5_path):
    """Load hdf5 data. 
    """
    with h5py.File(hdf5_path, 'r') as hf:
        x = hf.get('x')
        y = hf.get('y')
        x = np.array(x)     # (n_segs, n_concat, n_freq)
        y = np.array(y)     # (n_segs, n_freq)        
    return x, y

def np_mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true))




    
###
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')
    

    parser_calculate_features = subparsers.add_parser('calculate_features')
    parser_calculate_features.add_argument('--workspace', type=str, required=True)
    parser_calculate_features.add_argument('--speech_dir', type=str, required=True)
    parser_calculate_features.add_argument('--data_type', type=str, required=True)
    parser_calculate_features.add_argument('--TF', type=str, required=True)
    
    parser_pack_features = subparsers.add_parser('pack_features')
    parser_pack_features.add_argument('--workspace', type=str, required=True)
    parser_pack_features.add_argument('--data_type', type=str, required=True)
    #parser_pack_features.add_argument('--snr', type=float, required=True)
    parser_pack_features.add_argument('--n_concat', type=int, required=True)
    parser_pack_features.add_argument('--n_hop', type=int, required=True)
    parser_pack_features.add_argument('--TF', type=str, required=True)
    
    parser_compute_scaler = subparsers.add_parser('compute_scaler')
    parser_compute_scaler.add_argument('--workspace', type=str, required=True)
    parser_compute_scaler.add_argument('--data_type', type=str, required=True)
    #parser_compute_scaler.add_argument('--snr', type=float, required=True)
    #parser_compute_scaler.add_argument('--TF', type=str, required=True)
    
    args = parser.parse_args()
    if args.mode == 'calculate_features':
        calculate_features(args)
    elif args.mode == 'pack_features':
        pack_features(args)       
    elif args.mode == 'compute_scaler':
        compute_scaler(args)
    else:
        raise Exception("Error!")
