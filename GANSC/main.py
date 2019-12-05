from __future__ import print_function
import tensorflow as tf
import numpy as np

import os
from tensorflow.python.client import device_lib
from scipy.io import wavfile
from data_loader import pre_emph
import time


devices = device_lib.list_local_devices()

flags = tf.app.flags
flags.DEFINE_integer("seed",111, "Random seed (Def: 111).")
flags.DEFINE_integer("epoch", 150, "Epochs to train (Def: 150).")
flags.DEFINE_integer("batch_size", 150, "Batch size (Def: 150).")
flags.DEFINE_integer("save_freq", 50, "Batch save freq (Def: 50).")
flags.DEFINE_integer("canvas_size", 3200, "Canvas size (Def: 2^14).")
flags.DEFINE_integer("denoise_epoch", 5, "Epoch where noise in disc is "
                                          "removed (Def: 5).")
flags.DEFINE_integer("l1_remove_epoch", 150, "Epoch where L1 in G is "
                                           "removed (Def: 150).")
flags.DEFINE_boolean("bias_deconv", False,
                     "Flag to specify if we bias deconvs (Def: False)")
flags.DEFINE_boolean("bias_downconv", False,
                     "flag to specify if we bias downconvs (def: false)")
flags.DEFINE_boolean("bias_D_conv", False,
                     "flag to specify if we bias D_convs (def: false)")
# TODO: noise decay is under check
flags.DEFINE_float("denoise_lbound", 0.01, "Min noise std to be still alive (Def: 0.001)")
flags.DEFINE_float("noise_decay", 0.7, "Decay rate of noise std (Def: 0.7)")
flags.DEFINE_float("d_label_smooth", 0.25, "Smooth factor in D (Def: 0.25)")
flags.DEFINE_float("init_noise_std", 0.5, "Init noise std (Def: 0.5)")
flags.DEFINE_float("init_l1_weight", 100., "Init L1 lambda (Def: 100)")
flags.DEFINE_integer("z_dim", 256, "Dimension of input noise to G (Def: 256).")
flags.DEFINE_integer("z_depth", 256, "Depth of input noise to G (Def: 256).")
flags.DEFINE_string("save_path", "segan_results", "Path to save out model "
                                                   "files. (Def: dwavegan_model"
                                                   ").")
flags.DEFINE_string("g_nl", "leaky", "Type of nonlinearity in G: leaky or prelu. (Def: leaky).")
flags.DEFINE_string("model", "gan", "Type of model to train: gan or ae. (Def: gan).")
flags.DEFINE_string("loss_type", "l1_adv_loss", "Type of loss for D to train: wasserstein or l1_adv_loss. (Def: wasserstein).")
flags.DEFINE_string("deconv_type", "deconv", "Type of deconv method: deconv or "
                                             "nn_deconv (Def: deconv).")
flags.DEFINE_string("g_type", "ae", "Type of G to use: ae or dwave. (Def: ae).")
flags.DEFINE_float("g_learning_rate", 0.0002, "G learning_rate (Def: 0.0002)")
flags.DEFINE_float("d_learning_rate", 0.0002, "D learning_rate (Def: 0.0002)")
flags.DEFINE_float("beta_1", 0.5, "Adam beta 1 (Def: 0.5)")
flags.DEFINE_float("preemph", 0.95, "Pre-emph factor (Def: 0.95)")
flags.DEFINE_string("synthesis_path", "dwavegan_samples", "Path to save output"
                                                          " generated samples."
                                                          " (Def: dwavegan_sam"
                                                          "ples).")
flags.DEFINE_string("e2e_dataset", "/home/szuer/PLGAN2019/bwe_tfrecords/bl56spk_wavform_200ms.tfrecords", "TFRecords"
                                                          " (Def: reverberation_data/"
                                                          "pl56spk_wavformv1_200ms.tfrecords.")
flags.DEFINE_string("save_clean_path", "test_clean_results", "Path to save clean utts")
flags.DEFINE_string("test_wav", None, "name of test wav (it won't train)")
flags.DEFINE_string("weights", None, "Weights file")
flags.DEFINE_string("feature_type", "wavform", "Feature type of GAN to use: wavform or logspec. (Def: wavform).")
FLAGS = flags.FLAGS

def pre_emph_test(coeff, canvas_size):
    x_ = tf.placeholder(tf.float32, shape=[canvas_size,])
    x_preemph = pre_emph(x_, coeff)
    return x_, x_preemph

def main(_):
    if FLAGS.feature_type == 'wavform':
        from model import SEGAN, SEAE
    elif FLAGS.feature_type == 'logspec':
        from spec_model import SEGAN, SEAE

    print('Parsed arguments: ', FLAGS.__flags)

    # make save path if it is required
    if not os.path.exists(FLAGS.save_path):
        os.makedirs(FLAGS.save_path)
    if not os.path.exists(FLAGS.synthesis_path):
        os.makedirs(FLAGS.synthesis_path)
    np.random.seed(FLAGS.seed)
    #gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement=True
    config.gpu_options.allocator_type = 'BFC'
    udevices = []
    for device in devices:
        print("Device lists:{}".format(devices))
        if len(devices) > 1 and 'cpu' in device.name:
            # Use cpu only when we dont have gpus
            continue
        print('Using device: ', device.name)
        udevices.append(device.name)
    print("device:{}".format(udevices))
    # execute the session
    with tf.Session(config=config) as sess:
        if FLAGS.model == 'gan':
            print('Creating GAN model')
            se_model = SEGAN(sess, FLAGS, udevices)
        elif FLAGS.model == 'ae':
            print('Creating AE model')
            se_model = SEAE(sess, FLAGS, udevices)
        else:
            raise ValueError('{} model type not understood!'.format(FLAGS.model))

        if FLAGS.test_wav is None:
            mode = 'stage2'
            se_model.train(FLAGS, udevices, mode)

        else:
            if FLAGS.weights is None:
                raise ValueError('weights must be specified!')
            print('Loading model weights...')
            se_model.load(FLAGS.save_path, FLAGS.weights)

            noisy_test_filelist = []
            for (dirpath, dirnames, filenames) in os.walk(FLAGS.test_wav):
                # print('dirpath = ' + dirpath)
                for filename in filenames:
                    file_path = os.path.join(dirpath, filename)
                    noisy_test_filelist.append(file_path)
            nlist = noisy_test_filelist

            for name in nlist:
                t1 = time.time()
                fm, wav_data = wavfile.read(name)

                wavname = name.split('/')[-1]
                if fm != 16000:
                    raise ValueError('16kHz required! Test file is different')
                    #import librosa
                    #print('16kHz is required: test file is {}kHz, have to resample to the required samplerate')
                    #wav_data = librosa.resample(wav_data, fm, 16000)
                if FLAGS.feature_type == 'wavform':
                    wave = (2./65535.) * (wav_data.astype(np.float32) - 32767) + 1.
                    if FLAGS.preemph  > 0:
                        print('preemph test wave with {}'.format(FLAGS.preemph))
                        x_pholder, preemph_op = pre_emph_test(FLAGS.preemph, wave.shape[0])
                        wave = sess.run(preemph_op, feed_dict={x_pholder:wave})
                    print('test wave shape: ', wave.shape)
                    print('test wave min:{}  max:{}'.format(np.min(wave), np.max(wave)))
                    c_wave = se_model.clean(wave)
                    print('c wave min:{}  max:{}'.format(np.min(c_wave), np.max(c_wave)))
                    wavfile.write(os.path.join(FLAGS.save_clean_path, wavname), 16e3, np.int16(c_wave*32767)) #(0.9*c_wave/max(abs(c_wave)))
                    t2 = time.time() #np.int16((1.0*c_wave/max(abs(c_wave)))*32767)
                    print('Done cleaning {}/{}s and saved '
                            'to {}'.format(name, t2-t1,
                                        os.path.join(FLAGS.save_clean_path, wavname)))

                if FLAGS.feature_type == 'logspec':
                    if wav_data.dtype != 'float32':
                        wave = np.float32(wav_data / 32767.)
                    #wave = (2./65535.) * (wav_data.astype(np.float32) - 32767) + 1.
                    if FLAGS.preemph  > 0:
                        print('preemph test wave with {}'.format(FLAGS.preemph))
                        x_pholder, preemph_op = pre_emph_test(FLAGS.preemph, wave.shape[0])
                        wave = sess.run(preemph_op, feed_dict={x_pholder:wave})
                    print('test wave shape: ', wave.shape)
                    print('test wave min:{}  max:{}'.format(np.min(wave), np.max(wave)))
                    c_wave = se_model.clean(wave)
                    print('c wave min:{}  max:{}'.format(np.min(c_wave), np.max(c_wave)))
                    wavfile.write(os.path.join(FLAGS.save_clean_path, wavname), 16e3, np.int16(c_wave * 32767))  #(0.9*c_wave/max(abs(c_wave)))
                    t2 = time.time()
                    print('Done cleaning {}/{}s and saved '
                            'to {}'.format(name, t2-t1,
                                        os.path.join(FLAGS.save_clean_path, wavname)))
    '''
    if FLAGS.test_wav is not None:
        with tf.device("/cpu:0"):
            if FLAGS.weights is None:
                raise ValueError('weights must be specified!')
            print('Loading model weights...')
            # load_checkpoint = os.path.join(FLAGS.save_path, 'checkpoint')
            se_model.load(FLAGS.save_path, FLAGS.weights)

            noisy_test_filelist = []
            for (dirpath, dirnames, filenames) in os.walk(FLAGS.test_wav):
                # print('dirpath = ' + dirpath)
                for filename in filenames:
                    file_path = os.path.join(dirpath, filename)
                    noisy_test_filelist.append(file_path)
            nlist = noisy_test_filelist

            for name in nlist:
                fm, wav_data = wavfile.read(name)

                wavname = name.split('/')[-1]
                if fm != 16000:
                    raise ValueError('16kHz required! Test file is different')
                    # import librosa
                    # print('16kHz is required: test file is {}kHz, have to resample to the required samplerate')
                    # wav_data = librosa.resample(wav_data, fm, 16000)
                wave = (2. / 65535.) * (wav_data.astype(np.float32) - 32767) + 1.
                if FLAGS.preemph > 0:
                    print('preemph test wave with {}'.format(FLAGS.preemph))
                    x_pholder, preemph_op = pre_emph_test(FLAGS.preemph, wave.shape[0])
                    wave = sess.run(preemph_op, feed_dict={x_pholder: wave})
                print('test wave shape: ', wave.shape)
                print('test wave min:{}  max:{}'.format(np.min(wave), np.max(wave)))
                c_wave = se_model.clean(wave)
                print('c wave min:{}  max:{}'.format(np.min(c_wave), np.max(c_wave)))
                wavfile.write(os.path.join(FLAGS.save_clean_path, wavname), 16e3, c_wave)
                print('Done cleaning {} and saved '
                      'to {}'.format(name,
                                     os.path.join(FLAGS.save_clean_path, wavname)))
    '''




if __name__ == '__main__':
    tf.app.run()
