from __future__ import print_function
import tensorflow as tf
import numpy as np
from collections import namedtuple, OrderedDict
from subprocess import call
import scipy.io.wavfile as wavfile
import argparse
import codecs
import timeit
import struct
import toml
import re
import sys
import os
import scipy
from scipy import signal
from numpy.lib import stride_tricks
import librosa

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def slice_signal(signal, window_size, stride=0.5):
    """ Return windows of the given signal by sweeping in stride fractions
        of window
    """
    assert signal.ndim == 1, signal.ndim
    n_samples = signal.shape[0]
    offset = int(window_size * stride)
    slices = []
    for beg_i, end_i in zip(range(0, n_samples, offset),
                            range(window_size, n_samples + offset,
                                  offset)):
        if end_i - beg_i < window_size:
            break
        slice_ = signal[beg_i:end_i]
        if slice_.shape[0] == window_size:
            slices.append(slice_)
    return np.array(slices, dtype=np.int32)

def read_and_slice(filename, wav_canvas_size, stride=0.5):
    fm, wav_data = wavfile.read(filename)
    if fm != 16000:
        raise ValueError('Sampling rate is expected to be 16kHz!')
    signals = slice_signal(wav_data, wav_canvas_size, stride)
    return signals


def frame_func(sig, frameSize, overLap, window=scipy.signal.hamming):
    """ short time fourier transform of audio signal """
    win = window(frameSize)
    hopSize = frameSize - overLap
    # samples = np.array(sig, dtype='float64')
    # cols = int(np.ceil((len(sig) - frameSize) / float(hopSize)))
    cols = int(np.floor((len(sig) - frameSize) / float(hopSize))) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(sig, np.zeros(frameSize))
    frames = stride_tricks.as_strided(
        samples,
        shape=(cols, frameSize),
        strides=(samples.strides[0] * int(hopSize), samples.strides[0])).copy()
    frames *= win

    return frames

def make_spectrum(filename, preemph, use_normalize):

    def pre_emph(x, coeff=0.95):
        x0 = np.reshape(x[0], [1, ])
        diff = x[1:] - coeff * x[:-1]
        # concat = tf.concat(0, [x0, diff])
        concat = np.concatenate((x0, diff))
        return concat
    '''
    sr, y = wavfile.read(filename)
    if sr != 16000:
        raise ValueError('Sampling rate is expected to be 16kHz!')

    if y.dtype!='float32':
        y = np.float32(y/32767.)

    if preemph > 0:
        print('pre_emphase the wave signal with {}'.format(preemph))
        y = pre_emph(y, preemph)
    #D=librosa.stft(y,n_fft=512,hop_length=256,win_length=512,window=scipy.signal.hamming)
    y_frames = frame_func(y, 320, 160)
    Y_frames = np.fft.fft(y_frames)
    Y_frames = Y_frames.T
    #Y_temp = Y_frames[:, 0:161].T
    Y_angles = np.angle(Y_frames)

    Sxx = 20 * np.log10(abs(Y_frames) + 1e-12)
    #Sxx=np.log10(abs(D)**2)
    if use_normalize:
        mean = np.mean(Sxx, axis=1).reshape((320,1))
        std = np.std(Sxx, axis=1).reshape((320,1))+1e-12
        Sxx = (Sxx-mean)/std

    
    slices = []
    FRAMELENGTH = 10
    OVERLAP = 10//2
    for i in range(0, Sxx.shape[1]-FRAMELENGTH, OVERLAP):
        slices.append(Sxx[:,i:i+FRAMELENGTH])
    
    Sxx = np.concatenate((Sxx, Y_angles), axis=0)
    n_samples = Sxx.shape[1]
    window_size = 10
    stride = 0.5
    offset = int(window_size * stride)
    slices = []
    for beg_i, end_i in zip(range(0, n_samples, offset),
                            range(window_size, n_samples + offset,
                                  offset)):
        if end_i - beg_i < window_size:
            break
        slice_ = Sxx[:, beg_i:end_i]
        #print('slice_ shape is {}'.format(slice_.shape))
        if slice_.shape[1] == window_size:
            slices.append(slice_)
    return np.array(slices, dtype=np.float32)
    '''

    sr, y = wavfile.read(filename)
    if sr != 16000:
        raise ValueError('Sampling rate is expected to be 16kHz!')

    if y.dtype!='float32':
        y = np.float32(y/32767.)

    if preemph > 0:
        print('pre_emphase the wave signal with {}'.format(preemph))
        y = pre_emph(y, preemph)
    #D=librosa.stft(y,n_fft=512,hop_length=256,win_length=512,window=scipy.signal.hamming)
    y_frames = frame_func(y, 320, 0)
    Y_frames = np.fft.fft(y_frames)
    #Y_frames = librosa.core.stft(y, n_fft=320, hop_length=160, win_length=320, window='hamming', center=False)
    Y_frames = Y_frames.T
    #Y_temp = Y_frames[:, 0:161].T
    Y_angles = np.angle(Y_frames)

    #Sxx = 20 * np.log10(abs(Y_frames) + 1e-12)
    Sxx = abs(Y_frames)
    #Sxx = np.log10(abs(Y_frames) ** 2)
    #Sxx=np.log10(abs(D)**2)
    if use_normalize:
        mean = np.mean(Sxx, axis=1).reshape((320,1))
        std = np.std(Sxx, axis=1).reshape((320,1))+1e-12
        Sxx = (Sxx-mean)/std

    '''
    slices = []
    FRAMELENGTH = 10
    OVERLAP = 10//2
    for i in range(0, Sxx.shape[1]-FRAMELENGTH, OVERLAP):
        slices.append(Sxx[:,i:i+FRAMELENGTH])
    '''
    #Sxx = np.concatenate((Sxx, Y_angles), axis=0)
    n_samples = Sxx.shape[1]
    window_size = 10
    stride = 0.5
    offset = int(window_size * stride)
    slices = []
    for beg_i, end_i in zip(range(0, n_samples, offset),
                            range(window_size, n_samples + offset,
                                  offset)):
        if end_i - beg_i < window_size:
            break
        slice_ = Sxx[:, beg_i:end_i]
        phase_ = Y_angles[:, beg_i:end_i]
        #print('slice_ shape is {}'.format(slice_.shape))
        assert slice_.shape == phase_.shape
        if slice_.shape[1] == window_size:
            slice_flat = (slice_.T).flatten()
            phase_flat = (phase_.T).flatten()
            #print('slice_flat shape is {}'.format(slice_flat.shape))
            slice_phase = np.concatenate((slice_flat, phase_flat))
            #print('slice_phase shape is {}'.format(slice_phase.shape))
            slices.append(slice_phase)
    return np.array(slices, dtype=np.float32)




def encoder_proc(wav_filename, noisy_path, out_file, wav_canvas_size, feature_type):
    """ Read and slice the wav and noisy files and write to TFRecords.
        out_file: TFRecordWriter.
    """
    ppath, wav_fullname = os.path.split(wav_filename)
    noisy_filename = os.path.join(noisy_path, wav_fullname)
    #feature_type = 'logspec'

    if feature_type == 'wavform':
        wav_signals = read_and_slice(wav_filename, wav_canvas_size)
        noisy_signals = read_and_slice(noisy_filename, wav_canvas_size)
        assert wav_signals.shape == noisy_signals.shape

        for (wav, noisy) in zip(wav_signals, noisy_signals):
            wav_raw = wav.tostring()
            noisy_raw = noisy.tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'wav_raw': _bytes_feature(wav_raw),
                'noisy_raw': _bytes_feature(noisy_raw)}))
            out_file.write(example.SerializeToString())
    elif feature_type == 'logspec':
        preemph = 0.95
        wav_signals = make_spectrum(wav_filename, preemph, False)
        noisy_signals = make_spectrum(noisy_filename, preemph, False)
        print('wav_signals shape is {}'.format(wav_signals.shape))
        assert wav_signals.shape == noisy_signals.shape

        for (wav, noisy) in zip(wav_signals, noisy_signals):
            wav_raw = wav.tostring()
            noisy_raw = noisy.tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'wav_raw': _bytes_feature(wav_raw),
                'noisy_raw': _bytes_feature(noisy_raw)}))
            out_file.write(example.SerializeToString())




def main(opts):
    if not os.path.exists(opts.save_path):
        # make save path if it does not exist
        os.makedirs(opts.save_path)
    # set up the output filepath
    out_filepath = os.path.join(opts.save_path, opts.out_file)
    if os.path.splitext(out_filepath)[1] != '.tfrecords':
        # if wrong extension or no extension appended, put .tfrecords
        out_filepath += '.tfrecords'
    else:
        out_filename, ext = os.path.splitext(out_filepath)
        out_filepath = out_filename + ext
    # check if out_file exists and if force flag is set
    if os.path.exists(out_filepath) and not opts.force_gen:
        raise ValueError('ERROR: {} already exists. Set force flag (--force-gen) to '
                         'overwrite. Skipping this speaker.'.format(out_filepath))
    elif os.path.exists(out_filepath) and opts.force_gen:
        print('Will overwrite previously existing tfrecords')
        os.unlink(out_filepath)
    with open(opts.cfg) as cfh:
        # read the configuration description
        cfg_desc = toml.loads(cfh.read())
        beg_enc_t = timeit.default_timer()
        out_file = tf.python_io.TFRecordWriter(out_filepath)
        # process the acoustic and textual data now
        for dset_i, (dset, dset_desc) in enumerate(cfg_desc.iteritems()):
            print('-' * 50)
            wav_dir = dset_desc['clean']
            wav_files = [os.path.join(wav_dir, wav) for wav in
                           os.listdir(wav_dir) if wav.endswith('.wav')]
            noisy_dir = dset_desc['noisy']
            nfiles = len(wav_files)
            for m, wav_file in enumerate(wav_files):
                print('Processing wav file {}/{} {}-->{}{}'.format(m + 1,
                                                              nfiles,
                                                            noisy_dir,
                                                              wav_file,
                                                              ' ' * 10),
                      end='\r')
                sys.stdout.flush()
                #if opts.feature_type == 'wavform':
                encoder_proc(wav_file, noisy_dir, out_file, 320*10, opts.feature_type)
        out_file.close()
        end_enc_t = timeit.default_timer() - beg_enc_t
        print('')
        print('*' * 50)
        print('Total processing and writing time: {} s'.format(end_enc_t))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert the set of txt and '
                                                 'wavs to TFRecords')
    parser.add_argument('--cfg', type=str, default='cfg/e2e_maker.cfg',
                        help='File containing the description of datasets '
                             'to extract the info to make the TFRecords.')
    parser.add_argument('--save_path', type=str, default='packetloss_data/',
                        help='Path to save the dataset')
    parser.add_argument('--out_file', type=str, default='bl56spk_wavform_200ms.tfrecords',
                        help='Output filename')
    parser.add_argument('--feature_type', type=str, default='wavform', #wavform, logspec
                        help='feature type to extract')
    parser.add_argument('--force-gen', dest='force_gen', action='store_true',
                        help='Flag to force overwriting existing dataset.')
    parser.set_defaults(force_gen=False)
    opts = parser.parse_args()
    main(opts)
