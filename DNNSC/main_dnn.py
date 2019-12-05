#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Summary:  Train, inference and evaluate speech enhancement. 
Author:   Yupeng Shi
Created:  07/20/2018
Modified: 07/27/2018
"""
import numpy as np
import os
import pickle
import cPickle
import argparse
import time
#import glob
import matplotlib.pyplot as plt

import prepare_data as pp_data
import config as cfg
# from data_generator import DataGenerator
#from spectrogram_to_wave import recover_wav, time_recover_wav, spectra_to_wav
from frames_to_wav import time_recover_wav, spectra_to_wav

#from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.models import load_model
from keras.layers.normalization import BatchNormalization
from keras import regularizers

from keras.callbacks import Callback, ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
from keras import backend as K
from keras.utils.vis_utils import plot_model

from file_logger import FileLogger
import warnings
import librosa
import scipy

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import h5py
    
from models import *   

from spectrum import lpc
from scipy import signal
import multiprocessing 
import fnmatch   

def eval(model, gen, x, y):
    """Validation function. 
    
    Args:
      model: keras model. 
      gen: object, data generator. 
      x: 3darray, input, (n_segs, n_concat, n_freq)
      y: 2darray, target, (n_segs, n_freq)
    """
    pred_all, y_all = [], []
    
    # Inference in mini batch. 
    for (batch_x, batch_y) in gen.generate(xs=[x], ys=[y]):
        pred = model.predict(batch_x)
        pred_all.append(pred)
        y_all.append(batch_y)
        
    # Concatenate mini batch prediction. 
    pred_all = np.concatenate(pred_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    
    # Compute loss. 
    loss = pp_data.np_mean_absolute_error(y_all, pred_all)
    return loss

def func(n_frames, n_freqs, data_frames):
    #win=np.hanning(frame_length)
    print("computing the weight factor...")
    w_matrix = np.ones_like(data_frames)
    for i in range(n_frames):
            
        ak, _ = lpc(data_frames[i, :], 10)
        a_numerator = np.insert(ak, 0, 1)
        a_denominator = np.ones_like(a_numerator)
        for ii in range(1, len(a_numerator)):
            a_denominator[ii] = (0.8 ** ii) * a_numerator[ii]

        _, H = signal.freqz(a_numerator, a_denominator, worN=n_freqs, whole=True)
        w_matrix[i, :] = H.copy()
     
    return 20*np.log10(abs(w_matrix[:, 0:n_freqs]))
            ##h = np.fft.irfft(H, len(H))
            # hhh = np.fft.fft(hh)

def find_models(directory, pattern=['*.h5']):
    '''find files in the directory'''
    models = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern[0]):
            models.append(os.path.join(root, filename))
            
    return models


def weight_mean_absolute_error(y_true, y_pred):
    #y_sig = K.get_value(y_true)
    tf_session = K.get_session()
    y_sig = y_true.eval(session=tf_session)
    #y_sig_pred = K.get_value(y_pred)
    y_sig_pred = y_pred.eval(session=tf_session)
    (n_frame, n_freq) = y_sig.shape
    wf = func(n_frame, n_freq, y_sig)
    YK_true = np.fft.fft(y_sig)
    YK_pred = np.fft.fft(y_sig_pred)
    
    
    val = np.abs((YK_true - YK_pred))*np.abs(wf)
    y_error = K.variable(val.mean(), dtype='float32')

    #yy = K.placeholder(ndim=2)
    
    return y_error

def train(args):
    """Train the neural network. Write out model every several iterations. 
    
    Args:
      workspace: str, path of workspace. 
      tr_snr: float, training SNR. 
      te_snr: float, testing SNR. 
      lr: float, learning rate. 
    """
    class MetricsHistory(Callback):
        def on_epoch_end(self, epoch, logs={}):
            file_logger.write([str(epoch),
                           str(logs['loss']),
                           str(logs['val_loss'])
                           ])
    
    
    
    print(args)
    workspace = args.workspace

    #tr_snr = args.tr_snr
    #te_snr = args.te_snr
    lr = args.lr
    #TF = args.TF
    model_name = args.model_name
    #model_save_dir = os.path.join(args.workspace, 'saved_models')
    
    # Load data
    t1 = time.time()
    print("Loading the train and vallidation dataset")
    tr_hdf5_path = os.path.join(workspace, "packed_features", "train", "mag.h5")
    te_hdf5_path = os.path.join(workspace, "packed_features", "val", "mag.h5")
    (tr_x, tr_y) = pp_data.load_hdf5(tr_hdf5_path)
    (te_x, te_y) = pp_data.load_hdf5(te_hdf5_path)
    
    print('train_x shape:')
    print(tr_x.shape, tr_y.shape)
    print('test_x shape:')
    print(te_x.shape, te_y.shape)
    print("Load data time: %f s" % (time.time() - t1))
    print('\n')
    
    # Scale data
    if True:
        print("Scaling train and test dataset. This will take some time, please wait patiently...")
        t1 = time.time()
        scaler_path = os.path.join(workspace, "packed_features", "train", "mag_scaler.p")
        scaler = pickle.load(open(scaler_path, 'rb'))
        tr_x = pp_data.scale_on_3d(tr_x, scaler)
        tr_y = pp_data.scale_on_2d(tr_y, scaler)
        te_x = pp_data.scale_on_3d(te_x, scaler)
        te_y = pp_data.scale_on_2d(te_y, scaler)
        print("Scale data time: %f s" % (time.time() - t1))
        
    # Debug plot. 
    if False:
        plt.matshow(tr_x[0 : 1000, 0, :].T, origin='lower', aspect='auto', cmap='jet')
        plt.show()
        #time.sleep(secs)
        os.system("pause")
        
    # Build model
    batch_size = 150
    epoch = 100
    print("The neural networks you have chosed is %s" % model_name)
    print("The training batch is set to %d and the %s will be training for at most %d epoches" % (batch_size, model_name.upper(), epoch))
    print("======iteration of one epoch======" )
    iter_each_epoch = int(tr_x.shape[0] / batch_size)
    #val_each_epoch = int(te_x.shape[0] / batch_size)
    #print("There are %d iterations / epoch" % int(tr_x.shape[0] / batch_size))
    print("There are %d iterations / epoch" % iter_each_epoch)
    
    log_save_dir = os.path.join(workspace, 'log')
    if not os.path.isdir(log_save_dir):
        os.makedirs(log_save_dir)
    log_path = os.path.join(log_save_dir, 'out_{}.csv'.format(model_name))
    #log_path = os.path.join(log_save_dir, 'out_%ddb_%s.csv' %(int(snr[0]), model_name))
    file_logger = FileLogger(log_path, ['epoch', 'train_loss', 'val_loss'])
    
    (_, n_concat, n_freq) = tr_x.shape
    #temp_tr_x = tr_x[:, 3, :][:, np.newaxis, :]
    #print(temp_tr_x.shape)
    #np.axis
    n_hid = 2048
    
    #data_gen = DataGenerator(batch_size=batch_size, type='train')
    #tr_gen = data_gen.generate(xs=[tr_x], ys=[tr_y])
    #te_gen = data_gen.generate(xs=[te_x], ys=[te_y])
    #temp_tr_x = tr_gen[:, 3, :][:, np.newaxis, :]
    
    
    '''
    model = Sequential()
    model.add(Flatten(input_shape=(n_concat, n_freq)))
    model.add(BatchNormalization())
    model.add(Dense(n_hid, activation='relu', kernel_regularizer=regularizers.l2(l=0.0001)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(n_hid, activation='relu', kernel_regularizer=regularizers.l2(l=0.0001)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(n_hid, activation='relu', kernel_regularizer=regularizers.l2(l=0.0001)))
    model.add(Dropout(0.2))
    model.add(Dense(n_freq, activation='linear'))
    #model.summary()
    '''
    
    
    print('Model selected:', model_name.lower())
    if model_name == 'dnn':
        model = dnn(n_hid, n_concat, n_freq)
    
    elif model_name == 'sdnn1':
        model = sdnn1(n_hid, n_concat, n_freq)
        
    
    elif model_name == 'sdnn2':
        model = sdnn2(n_hid, n_concat, n_freq)
    
    elif model_name == 'sdnn3':
        model = sdnn3(n_hid, n_concat, n_freq)
    
    elif model_name == 'fcn':
        model = fcn(n_concat, n_freq)
        
    elif model_name == 'fcn1':
        model = fcn1(n_concat, n_freq)
        
    elif model_name == 'fcn1':
        model = fcn1_re(n_concat, n_freq)
    
    elif model_name == 'fcn2':
        model = fcn2(n_concat, n_freq)
        
    elif model_name == 'fcn3':
        model = fcn3(n_concat, n_freq)
        
    elif model_name == 'fcn4':
        model = fcn4(n_concat, n_freq)
        
    elif model_name == 'm_vgg':
        model = m_vgg(n_concat, n_freq)
        
    elif model_name == 'm_vgg1':
        model = m_vgg1(n_concat, n_freq)
        
    elif model_name == 'm_vgg2':
        model = m_vgg2(n_concat, n_freq)
        
    elif model_name == 'm_vgg3':
        model = m_vgg3(n_concat, n_freq)
        
    elif model_name == 'm_vgg4':
        model = m_vgg3(n_concat, n_freq)
        
    elif model_name == 'CapsNet':
        model = CapsNet(n_concat, n_freq, 3)
        
    elif model_name == 'brnn' :
        recur_layers = 7
        unit = 256
        output_dim = n_freq
        model = brnn(n_concat, n_freq, unit, recur_layers, output_dim)
        
    elif model_name == 'rnn' :
        output_dim = n_freq
        model = rnn(n_concat, n_freq, output_dim)
        
    elif model_name == 'tcn' :
        input_dim = n_freq
        model = tcn(n_concat, input_dim)
        
    if model is None:
        exit('Please choose a valid model: [dnn, sdnn, sdnn1, cnn, scnn1]')
        
   
    #mean_squared_error
    model.compile(loss = 'mean_squared_error',
                  optimizer=Adam(lr=lr))
    
    print(model.summary())
    #plot model
    #plot_model(model, to_file=args.save_dir+'/model.png', show_shapes=True)
    #plot_model(model, to_file='%s/%s_model.png' % (log_save_dir, model_name), show_shapes=True)
    # Save model and weights
    model_save_dir = os.path.join(workspace, 'saved_models', "%s" % model_name)
    model_save_name = "weights-checkpoint-{epoch:02d}-{val_loss:.2f}.h5"
    if not os.path.isdir(model_save_dir):
        os.makedirs(model_save_dir)
    model_path = os.path.join(model_save_dir, model_save_name)
    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    print('Saved trained model at %s' % model_save_dir)
    
    
    #reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=0.00001, verbose=1)
    lr_decay = LearningRateScheduler(schedule=lambda epoch: lr * (0.9 ** epoch))
    metrics_history = MetricsHistory()
    
    hist = model.fit(x=tr_x,
                     y=tr_y,
                     batch_size=batch_size,
                     epochs=epoch,
                     verbose=1,
                     shuffle=True,
                     validation_data=(te_x, te_y),
                     #validation_split=0.1,
                     callbacks=[metrics_history, checkpoint, lr_decay])
    '''
    hist = model.fit_generator(tr_gen, 
                               steps_per_epoch=iter_each_epoch, 
                               epochs=epoch, 
                               verbose=1, 
                               validation_data=te_gen, 
                               validation_steps=val_each_epoch, 
                               callbacks=[metrics_history, checkpoint, reduce_lr])

    '''
    
    print(hist.history.keys())
    
    # list all data in history
    #print(hist.history.keys())
    '''
    # summarize history for accuracy
    plt.plot(hist.history['acc'])
    plt.plot(hist.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    '''
    # summarize history for loss
    model_png = "train_test_loss"
    loss_fig_dir = os.path.join(log_save_dir, '%s_%s.png' % (model_name, model_png))
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.savefig(loss_fig_dir)
    #plt.show()
    
    
    
    '''
    fig = plt.gcf()
    plt.show()
    fig.savefig('tessstttyyy.png', dpi=100)
    '''
    
    file_logger.close()
    
    
    
    '''
    # Data generator. 
    tr_gen = DataGenerator(batch_size=batch_size, type='train')
    eval_te_gen = DataGenerator(batch_size=batch_size, type='test', te_max_iter=100)
    eval_tr_gen = DataGenerator(batch_size=batch_size, type='test', te_max_iter=100)
    
    # Directories for saving models and training stats
    model_dir = os.path.join(workspace, "models", "%ddb" % int(tr_snr))
    pp_data.create_folder(model_dir)
    
    stats_dir = os.path.join(workspace, "training_stats", "%ddb" % int(tr_snr))
    pp_data.create_folder(stats_dir)
    
    # Print loss before training. 
    iter = 0
    tr_loss = eval(model, eval_tr_gen, tr_x, tr_y)
    te_loss = eval(model, eval_te_gen, te_x, te_y)
    print("Iteration: %d, tr_loss: %f, te_loss: %f" % (iter, tr_loss, te_loss))
    
    # Save out training stats. 
    stat_dict = {'iter': iter, 
                    'tr_loss': tr_loss, 
                    'te_loss': te_loss, }
    stat_path = os.path.join(stats_dir, "%diters.p" % iter)
    cPickle.dump(stat_dict, open(stat_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)
    
    # Train. 
    t1 = time.time()
    for (batch_x, batch_y) in tr_gen.generate(xs=[tr_x], ys=[tr_y]):
        #loss = model.train_on_batch(batch_x, batch_y)
 	if iter % 2000 == 0:
            lr *= 0.1
        model.train_on_batch(batch_x, batch_y)
        iter += 1
        
        
        # Validate and save training stats. 
        if iter % 1000 == 0:
            tr_loss = eval(model, eval_tr_gen, tr_x, tr_y)
            te_loss = eval(model, eval_te_gen, te_x, te_y)
            print("Iteration: %d, tr_loss: %f, te_loss: %f" % (iter, tr_loss, te_loss))
            
            # Save out training stats. 
            stat_dict = {'iter': iter, 
                         'tr_loss': tr_loss, 
                         'te_loss': te_loss, }
            stat_path = os.path.join(stats_dir, "%diters.p" % iter)
            cPickle.dump(stat_dict, open(stat_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)
            
        # Save model. 
        if iter % 5000 == 0:
            model_path = os.path.join(model_dir, "md_%diters.h5" % iter)
            model.save(model_path)
            print("Saved model to %s" % model_path)
        
        if iter == 10001:
            break
     '''     
    print("Training time: %s s" % (time.time() - t1,))

def inference1111(args):
    """Inference all test data, write out recovered wavs to disk. 
    
    Args:
      workspace: str, path of workspace. 
      tr_snr: float, training SNR. 
      te_snr: float, testing SNR. 
      n_concat: int, number of frames to concatenta, should equal to n_concat 
          in the training stage. 
      iter: int, iteration of model to load. 
      visualize: bool, plot enhanced spectrogram for debug. 
    """
    print(args)
    workspace = args.workspace
    #tr_snr = args.tr_snr
    #te_snr = args.te_snr
    n_concat = args.n_concat
    #iter = args.iteration
    TF = args.TF
    model_name = args.model_name
    
    n_window = cfg.n_window
    n_overlap = cfg.n_overlap
    #snr = cfg.SNR
    n_hop = int(n_window-n_overlap)
    fs = cfg.sample_rate
    scale = True
    
    # Load model
    t1 = time.time()
    #model_path = os.path.join(workspace, "saved_models", "%s" % model_name, "weights-checkpoint-25-0.41.h5")
    model_root = os.path.join(workspace, "saved_models", "%s" % model_name )
    #model_root = '/home/szuer/CI_DNN/workspace_16kHz/cis_strategy/noise10/mixture/saved_models/0/sdnn1'
    model_files = find_models(model_root)
    epoch_num = []
    for i in range(len(model_files)):
        epoch_num.append(int(model_files[i].split("/")[-1].split('-')[2]))
    model_index = epoch_num.index(max(epoch_num))
    model_path = model_files[model_index]
    print("The selected model path is %s :" % model_path)
    
    model = load_model(model_path)
    
    # Load scaler
    scaler_path = os.path.join(workspace, "packed_features", "train", "scaler.p")
    scaler = pickle.load(open(scaler_path, 'rb'))
    
    # Load test data. 
    feat_dir = os.path.join(workspace, "features", "test")
    names = os.listdir(feat_dir)

    for (cnt, na) in enumerate(names):
        # Load feature. 
        feat_path = os.path.join(feat_dir, na)
        data = cPickle.load(open(feat_path, 'rb'))
        [mixed_cmplx_x, speech_x, na] = data
        n_pad = (n_concat - 1) / 2
        
        if TF == "spectrogram":
            mixed_x = np.abs(mixed_cmplx_x)
        
            # Process data. 
            #n_pad = (n_concat - 1) / 2
            mixed_x = pp_data.pad_with_border(mixed_x, n_pad)
            mixed_x = pp_data.log_sp(mixed_x)
            speech_x = pp_data.log_sp(speech_x)
            
        elif TF == "timedomain":
            #n_pad = (n_concat - 1) / 2
            mixed_x = pp_data.pad_with_border(mixed_cmplx_x, n_pad)
            
        elif TF == "fftmagnitude":
            #n_pad = (n_concat - 1) / 2
            mixed_x = np.abs(mixed_cmplx_x)
            mixed_x = pp_data.pad_with_border(mixed_x, n_pad)
            
        else:
            raise Exception("TF must be spectrogram, timedomain or fftmagnitude!")
            
        # Scale data. 
        if scale:
            mixed_x = pp_data.scale_on_2d(mixed_x, scaler)
            speech_x = pp_data.scale_on_2d(speech_x, scaler)
        
        # Cut input spectrogram to 3D segments with n_concat. 
        #mixed_x_3d = pp_data.mat_2d_to_3d(mixed_x, agg_num=n_concat, hop=1)
        mixed_x_3d = pp_data.mat_2d_to_3d(mixed_x, agg_num=n_concat, hop=1)
        #print("loading data time: %s s" % (time.time() - t1,))
        '''
        layer_1 = K.function([model.layers[0].input], [model.layers[2].output])#第一个 model.layers[0],不修改,表示输入数据；第二个model.layers[you wanted],修改为你需要输出的层数的编号
        f1 = layer_1([mixed_x_3d])[0]#只修改inpu_image
        #第一层卷积后的特征图展示，输出是（1,149,149,32），（样本个数，特征图尺寸长，特征图尺寸宽，特征图个数）
        for _ in range(12):
            show_img = f1[1, :, :, _]
            show_img.shape = [1, 257]
            plt.subplot(3, 4, _ + 1)
            plt.imshow(show_img.T, cmap='gray')
            plt.axis('off')
        plt.show()
        '''
        # Predict. 
        t2 = time.time()
        pred = model.predict(mixed_x_3d)
        print("model predicts %d utterance : %s successfully" % (cnt, na))
        #print(pred)
        
        # Inverse scale. 
        if scale:
            mixed_x = pp_data.inverse_scale_on_2d(mixed_x, scaler)
            speech_x = pp_data.inverse_scale_on_2d(speech_x, scaler)
            pred = pp_data.inverse_scale_on_2d(pred, scaler)
        
        #(frames, frame_length) = pred.shape
        #print("pred domensions %d and %d : " % (frames, frame_length))
        # Debug plot. 
        if args.visualize:
            if TF == "spectrogram":
                fig, axs = plt.subplots(3,1, sharex=False)
                axs[0].matshow(mixed_x.T, origin='lower', aspect='auto', cmap='jet')
                axs[1].matshow(speech_x.T, origin='lower', aspect='auto', cmap='jet')
                axs[2].matshow(pred.T, origin='lower', aspect='auto', cmap='jet')
                axs[0].set_title("%ddb mixture log spectrogram" % int(te_snr))
                axs[1].set_title("Clean speech log spectrogram")
                axs[2].set_title("Enhanced speech log spectrogram")
                for j1 in xrange(3):
                    axs[j1].xaxis.tick_bottom()
                    plt.tight_layout()
                    plt.savefig('debug_model_spectra.png')
                    plt.show()
            elif TF == "timedomain":
                fig, axs = plt.subplots(3,1, sharex=False)
                axs[0].matshow(mixed_x.T, origin='lower', aspect='auto', cmap='jet')
                axs[1].matshow(speech_x.T, origin='lower', aspect='auto', cmap='jet')
                axs[2].matshow(pred.T, origin='lower', aspect='auto', cmap='jet')
                axs[0].set_title("%ddb mixture time domain" % int(te_snr))
                axs[1].set_title("Clean speech time domain")
                axs[2].set_title("Enhanced speech time domain")
                for j1 in xrange(3):
                    axs[j1].xaxis.tick_bottom()
                    plt.tight_layout()
                    plt.savefig('debug model_time.png')
                    plt.show()
            else:
                raise Exception("TF must be spectrogram or timedomain!")
                    

        # Recover enhanced wav. 
        #pred_sp = np.exp(pred)
        if TF == "spectrogram":
            pred_sp = (10**(pred/20))-1e-10
            #s = recover_wav(pred_sp, mixed_cmplx_x, n_overlap, np.hamming)
            #s *= np.sqrt((np.hamming(n_window)**2).sum())   # Scaler for compensate the amplitude 
            s = spectra_to_wav(pred_sp, mixed_cmplx_x, n_window, n_hop, 'hamming')
                                                        # change after spectrogram and IFFT. 
        elif TF == "timedomain":
            s = time_recover_wav(pred, n_window, n_hop, 'hamming')
            #s *= np.sqrt((np.hamming(n_window)**2).sum())
            
        elif TF == "fftmagnitude":
            #n_pad = (n_concat - 1) / 2
            s = spectra_to_wav(pred, mixed_cmplx_x, n_window, n_hop, 'hamming')
            
        else:
            raise Exception("TF must be spectrogram timedomain or fftmagnitude!")
            
        # Write out enhanced wav. 
        out_path = os.path.join(workspace, "enh_wavs", "test", "%s" % model_name, "%s.wav" % na)
        pp_data.create_folder(os.path.dirname(out_path))
        pp_data.write_audio(out_path, s, fs)
        print("predict an utterance time: %s s" % (time.time() - t2,))
        
    print("total test time: %s s" % (time.time() - t1,))
    
    
def inference(args):
    """Inference all test data, write out recovered wavs to disk. 
    
    Args:
      workspace: str, path of workspace. 
      tr_snr: float, training SNR. 
      te_snr: float, testing SNR. 
      n_concat: int, number of frames to concatenta, should equal to n_concat 
          in the training stage. 
      iter: int, iteration of model to load. 
      visualize: bool, plot enhanced spectrogram for debug. 
    """
    print(args)
    workspace = args.workspace
    #tr_snr = args.tr_snr
    #te_snr = args.te_snr
    n_concat = args.n_concat
    #iter = args.iteration
    TF = args.TF
    model_name = args.model_name
    
    n_window = cfg.n_window
    n_overlap = cfg.n_overlap
    #snr = cfg.SNR
    n_hop = int(n_window-n_overlap)
    fs = cfg.sample_rate
    scale = True
    
    # Load model
    t1 = time.time()
    #model_path = os.path.join(workspace, "saved_models", "%s" % model_name, "weights-checkpoint-25-0.41.h5")
    mag_model_root = os.path.join(workspace, "saved_models", "%s" % model_name )
    #model_root = '/home/szuer/CI_DNN/workspace_16kHz/cis_strategy/noise10/mixture/saved_models/0/sdnn1'
    mag_model_files = find_models(mag_model_root)
    epoch_num = []
    for i in range(len(mag_model_files)):
        epoch_num.append(int(mag_model_files[i].split("/")[-1].split('-')[2]))
    mag_model_index = epoch_num.index(max(epoch_num))
    mag_model_path = mag_model_files[mag_model_index]
    print("The selected model path is %s :" % mag_model_path)
    
    mag_model = load_model(mag_model_path)
    
    '''
    # loading phase model
    phase_model_root = os.path.join(workspace, "phase_saved_models", "%s" % model_name )
    #model_root = '/home/szuer/CI_DNN/workspace_16kHz/cis_strategy/noise10/mixture/saved_models/0/sdnn1'
    phase_model_files = find_models(phase_model_root)
    epoch_num1 = []
    for i in range(len(phase_model_files)):
        epoch_num1.append(int(phase_model_files[i].split("/")[-1].split('-')[2]))
    phase_model_index = epoch_num1.index(max(epoch_num1))
    phase_model_path = phase_model_files[phase_model_index]
    print("The selected model path is %s :" % phase_model_path)
    
    phase_model = load_model(phase_model_path)
    '''
    # Load scaler
    mag_scaler_path = os.path.join(workspace, "packed_features", "train", "mag_scaler.p")
    mag_scaler = pickle.load(open(mag_scaler_path, 'rb'))
    
    #phase_scaler_path = os.path.join(workspace, "packed_features", "train", "phase_scaler.p")
    #phase_scaler = pickle.load(open(phase_scaler_path, 'rb'))
    
    # Load test data. 
    feat_dir = os.path.join(workspace, "features", "test")
    names = os.listdir(feat_dir)

    for (cnt, na) in enumerate(names):
        # Load feature. 
        feat_path = os.path.join(feat_dir, na)
        data = cPickle.load(open(feat_path, 'rb'))
        [mixed_cmplx_x, speech_cmplx_x] = data
        n_pad = (n_concat - 1) / 2
        
        if TF == "spectrogram":
            mixed_x = np.abs(mixed_cmplx_x)
            # mixed_phase = np.angle(mixed_cmplx_x)
            # Process data. 
            #n_pad = (n_concat - 1) / 2
            mixed_x = pp_data.pad_with_border(mixed_x, n_pad)
            mixed_x = pp_data.log_sp(mixed_x)
            # mixed_phase = pp_data.pad_with_border(mixed_phase, n_pad)
            
            # speech_x = pp_data.log_sp(np.abs(speech_cmplx_x))
            #speech_phase = np.angle(speech_cmplx_x)

            
        else:
            raise Exception("TF must be spectrogram, timedomain or fftmagnitude!")
            
        # Scale data. 
        if scale:
            mixed_x = pp_data.scale_on_2d(mixed_x, mag_scaler)
            # speech_x = pp_data.scale_on_2d(speech_x, mag_scaler)
            #mixed_phase = pp_data.scale_on_2d(mixed_phase, phase_scaler)
            #speech_phase = pp_data.scale_on_2d(speech_phase, phase_scaler)
        
        # Cut input spectrogram to 3D segments with n_concat. 
        #mixed_x_3d = pp_data.mat_2d_to_3d(mixed_x, agg_num=n_concat, hop=1)
        mixed_x_3d = pp_data.mat_2d_to_3d(mixed_x, agg_num=n_concat, hop=1)
        #mixed_phase_3d = pp_data.mat_2d_to_3d(mixed_phase, agg_num=n_concat, hop=1)
        #print("loading data time: %s s" % (time.time() - t1,))
        '''
        layer_1 = K.function([model.layers[0].input], [model.layers[2].output])#第一个 model.layers[0],不修改,表示输入数据；第二个model.layers[you wanted],修改为你需要输出的层数的编号
        f1 = layer_1([mixed_x_3d])[0]#只修改inpu_image
        #第一层卷积后的特征图展示，输出是（1,149,149,32），（样本个数，特征图尺寸长，特征图尺寸宽，特征图个数）
        for _ in range(12):
            show_img = f1[1, :, :, _]
            show_img.shape = [1, 257]
            plt.subplot(3, 4, _ + 1)
            plt.imshow(show_img.T, cmap='gray')
            plt.axis('off')
        plt.show()
        '''
        # Predict. 
        t2 = time.time()
        mag_pred = mag_model.predict(mixed_x_3d)
        #phase_pred = phase_model.predict(mixed_phase_3d)
        print("model predicts %d utterance : %s successfully" % (cnt, na))
        #print(pred)
        
        # Inverse scale. 
        if scale:
            # mixed_x = pp_data.inverse_scale_on_2d(mixed_x, mag_scaler)
            # speech_x = pp_data.inverse_scale_on_2d(speech_x, mag_scaler)
            mag_pred = pp_data.inverse_scale_on_2d(mag_pred, mag_scaler)
            
            #mixed_phase = pp_data.inverse_scale_on_2d(mixed_phase, phase_scaler)
            #speech_phase = pp_data.inverse_scale_on_2d(speech_phase, phase_scaler)
            #phase_pred = pp_data.inverse_scale_on_2d(phase_pred, phase_scaler)
        
       
                    

        # Recover enhanced wav. 
        #pred_sp = np.exp(pred)
        if TF == "spectrogram":
            pred_sp = (10**(mag_pred/10))-1e-10
            #pred_ph = np.exp(1j * phase_pred)
            '''
            R = np.multiply(pred_sp, pred_ph)
            result = librosa.istft(R.T,
                                   hop_length=n_hop,
                                   win_length=cfg.n_window,
                                   window=scipy.signal.hamming, center=False)
            result /= abs(result).max()
            y_out = result*0.8'''
            #s = recover_wav(pred_sp, mixed_cmplx_x, n_overlap, np.hamming)
            #s *= np.sqrt((np.hamming(n_window)**2).sum())   # Scaler for compensate the amplitude 
            s = spectra_to_wav(pred_sp, mixed_cmplx_x, n_window, n_hop, 'hamming')
            
        # Write out enhanced wav. 
        out_path = os.path.join(workspace, "enh_flipphase", "test", "%s" % model_name, "{}_fft_dnn_map.wav".format(na.split('.')[0]))
        pp_data.create_folder(os.path.dirname(out_path))
        pp_data.write_audio(out_path, s, fs)
        print("predict an utterance time: %s s" % (time.time() - t2,))
        
    print("total test time: %s s" % (time.time() - t1,))    

'''
def get_layers(args):
    print(args)
    workspace = args.workspace
    #tr_snr = args.tr_snr
    #te_snr = args.te_snr
    #n_concat = args.n_concat
    #iter = args.iteration
    #TF = args.TF
    model_name = args.model_name
    
    model_path = os.path.join(workspace, "saved_models", "%s" % model_name, "weights-checkpoint-24-0.34.h5")
    model = load_model(model_path) #replaced by your model name
    layer_1 = K.function([model.layers[0].input], [model.layers[1].output])#第一个 model.layers[0],不修改,表示输入数据；第二个model.layers[you wanted],修改为你需要输出的层数的编号
    f1 = layer_1([input_image])[0]#只修改inpu_image
    #第一层卷积后的特征图展示，输出是（1,149,149,32），（样本个数，特征图尺寸长，特征图尺寸宽，特征图个数）
    for _ in range(12):
        show_img = f1[1, :, :, _]
        show_img.shape = [1, 257]
        plt.subplot(4, 3, _ + 1)
        plt.imshow(show_img, cmap='gray')
        plt.axis('off')
    plt.show()
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--workspace', type=str, required=True)
    #parser_train.add_argument('--tr_snr', type=float, required=True)
    #parser_train.add_argument('--te_snr', type=float, required=True)
    parser_train.add_argument('--lr', type=float, required=True)
    #parser_train.add_argument('--TF', type=str, required=True)
    parser_train.add_argument('--model_name', type=str, required=True)
    
    parser_inference = subparsers.add_parser('inference')
    parser_inference.add_argument('--workspace', type=str, required=True)
    #parser_inference.add_argument('--tr_snr', type=float, required=True)
    #parser_inference.add_argument('--te_snr', type=float, required=True)
    parser_inference.add_argument('--n_concat', type=int, required=True)
    #parser_inference.add_argument('--iteration', type=int, required=True)
    parser_inference.add_argument('--visualize', action='store_true', default=False)
    parser_inference.add_argument('--TF', type=str, required=True)
    parser_inference.add_argument('--model_name', type=str, required=True)
    
    #parser_inference = subparsers.add_parser('inference')
    #parser_inference.add_argument('--workspace', type=str, required=True)
    #parser_inference.add_argument('--model_name', type=str, required=True)
    '''
    parser_calculate_pesq = subparsers.add_parser('calculate_pesq')
    parser_calculate_pesq.add_argument('--workspace', type=str, required=True)
    parser_calculate_pesq.add_argument('--speech_dir', type=str, required=True)
    parser_calculate_pesq.add_argument('--te_snr', type=float, required=True)
    '''
    args = parser.parse_args()
    
    if args.mode == 'train':
        train(args)
    elif args.mode == 'inference':
        inference(args)
        
    #elif args.mode == 'calculate_pesq':
    #    calculate_pesq(args)
    else:
        raise Exception("Error!")
