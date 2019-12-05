from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib.layers import batch_norm, fully_connected, flatten
from tensorflow.contrib.layers import xavier_initializer
from ops import *
import numpy as np
import ops_dropout as d_ops


class Generator(object):

    def __init__(self, segan):
        self.segan = segan

    def __call__(self, noisy_w, is_ref, spk=None):
        """ Build the graph propagating (noisy_w) --> x
        On first pass will make variables.
        """
        segan = self.segan

        def make_z(shape, mean=0., std=1., name='z'):
            if is_ref:
                with tf.variable_scope(name) as scope:
                    z_init = tf.random_normal_initializer(mean=mean, stddev=std)
                    z = tf.get_variable("z", shape,
                                        initializer=z_init,
                                        trainable=False
                                        )
                    if z.device != "/device:GPU:0":
                        # this has to be created into gpu0
                        print('z.device is {}'.format(z.device))
                        assert False
            else:
                z = tf.random_normal(shape, mean=mean, stddev=std,
                                     name=name, dtype=tf.float32)
            return z

        if hasattr(segan, 'generator_built'):
            tf.get_variable_scope().reuse_variables()
            make_vars = False
        else:
            make_vars = True

        print('*** Building Generator ***')
        in_dims = noisy_w.get_shape().as_list()
        h_i = noisy_w
        if len(in_dims) == 2:
            h_i = tf.expand_dims(noisy_w, -1)
        elif len(in_dims) < 2 or len(in_dims) > 3:
            raise ValueError('Generator input must be 2-D or 3-D')
        kwidth = 11
        z = make_z([segan.batch_size, h_i.get_shape().as_list()[1],
                    segan.g_enc_depths[-1]])
        h_i = tf.concat(2, [h_i, z])
        skip_out = True
        skips = []
        for block_idx, dilation in enumerate(segan.g_dilated_blocks):
                name = 'g_residual_block_{}'.format(block_idx)
                if block_idx >= len(segan.g_dilated_blocks) - 1:
                    skip_out = False
                if skip_out:
                    res_i, skip_i = residual_block(h_i,
                                                   dilation, kwidth, num_kernels=32,
                                                   bias_init=None, stddev=0.02,
                                                   do_skip = True,
                                                   name=name)
                else:
                    res_i = residual_block(h_i,
                                           dilation, kwidth, num_kernels=32,
                                           bias_init=None, stddev=0.02,
                                           do_skip = False,
                                           name=name)
                # feed the residual output to the next block
                h_i = res_i
                if segan.keep_prob < 1:
                    print('Adding dropout w/ keep prob {} '
                          'to G'.format(segan.keep_prob))
                    h_i = tf.nn.dropout(h_i, segan.keep_prob_var)
                if skip_out:
                    # accumulate the skip connections
                    skips.append(skip_i)
                else:
                    # for last block, the residual output is appended
                    skips.append(res_i)
        print('Amount of skip connections: ', len(skips))
        # TODO: last pooling for actual wave
        with tf.variable_scope('g_wave_pooling'):
            skip_T = tf.stack(skips, axis=0)
            skips_sum = tf.reduce_sum(skip_T, axis=0)
            skips_sum = leakyrelu(skips_sum)
            wave_a = conv1d(skips_sum, kwidth=1, num_kernels=1,
                            init=tf.truncated_normal_initializer(stddev=0.02))
            wave = tf.tanh(wave_a)
            segan.gen_wave_summ = histogram_summary('gen_wave', wave)
        print('Last residual wave shape: ', res_i.get_shape())
        print('*************************')
        segan.generator_built = True
        return wave, z

class AEGenerator(object):

    def __init__(self, segan):
        self.segan = segan

    def __call__(self, noisy_w, wav_w, is_ref, spk=None, z_on=True, do_prelu=False):
        # TODO: remove c_vec
        """ Build the graph propagating (noisy_w) --> x
        On first pass will make variables.
        """
        segan = self.segan
        #fft_error = tf.Variable(0., trainable=True, name='error')
        def make_z(shape, mean=0., std=1., name='z'):
            if is_ref:
                with tf.variable_scope(name) as scope:
                    z_init = tf.random_normal_initializer(mean=mean, stddev=std)
                    z = tf.get_variable("z", shape,
                                        initializer=z_init,
                                        trainable=False
                                        )
                    if z.device != "/device:GPU:0":
                        # this has to be created into gpu0
                        print('z.device is {}'.format(z.device))
                        assert False
            else:
                z = tf.random_normal(shape, mean=mean, stddev=std,
                                     name=name, dtype=tf.float32)
            return z

        if hasattr(segan, 'generator_built'):
            tf.get_variable_scope().reuse_variables()
            make_vars = False
        else:
            make_vars = True
        if is_ref:
            print('*** Building Generator ***')
        in_dims = noisy_w.get_shape().as_list()
        h_i = noisy_w
        if len(in_dims) == 2:
            h_i = tf.expand_dims(noisy_w, -1)
        elif len(in_dims) < 2 or len(in_dims) > 3:
            raise ValueError('Generator input must be 2-D or 3-D')
        if segan.feature_type == 'wavform':
            kwidth = 31
        elif segan.feature_type == 'logspec':
            kwidth = 31

        enc_layers = 7
        skips = []
        if is_ref and do_prelu:
            #keep track of prelu activations
            alphas = []
        with tf.variable_scope('g_ae'):
            #AE to be built is shaped:
            # enc ~ [16384x1, 8192x16, 4096x32, 2048x32, 1024x64, 512x64, 256x128, 128x128, 64x256, 32x256, 16x512, 8x1024]
            # dec ~ [8x2048, 16x1024, 32x512, 64x512, 8x256, 256x256, 512x128, 1024x128, 2048x64, 4096x64, 8192x32, 16384x1]
            #FIRST ENCODER
            print('g_enc_depths: ', segan.g_enc_depths)
            for layer_idx, layer_depth in enumerate(segan.g_enc_depths):
                bias_init = None
                if segan.bias_downconv:
                    if is_ref:
                        print('Biasing downconv in G')
                    bias_init = tf.constant_initializer(0.)
                if h_i.get_shape().as_list()[2] == layer_depth:
                    pool = 1
                else:
                    pool = 2
                h_i_dwn = downconv(h_i, layer_depth, kwidth=kwidth, pool=pool,
                                   init=tf.truncated_normal_initializer(stddev=0.02),
                                   bias_init=bias_init,
                                   name='enc_{}'.format(layer_idx), lnorm=False)
                if is_ref:
                    print('Downconv {} -> {}'.format(h_i.get_shape(),
                                                     h_i_dwn.get_shape()))
                h_i = h_i_dwn
                if layer_idx < len(segan.g_enc_depths) - 1:
                    if is_ref:
                        print('Adding skip connection downconv '
                              '{}'.format(layer_idx))
                    # store skip connection
                    # last one is not stored cause it's the code
                    skips.append(h_i)
                if do_prelu:
                    if is_ref:
                        print('-- Enc: prelu activation --')
                    h_i = prelu(h_i, ref=is_ref, name='enc_prelu_{}'.format(layer_idx))
                    if is_ref:
                        # split h_i into its components
                        alpha_i = h_i[1]
                        h_i = h_i[0]
                        alphas.append(alpha_i)
                else:
                    if is_ref:
                        print('-- Enc: leakyrelu activation --')
                    h_i = leakyrelu(h_i)

            if z_on:
                # random code is fused with intermediate representation
                z = make_z([segan.batch_size, h_i.get_shape().as_list()[1],
                            segan.g_enc_depths[-1]])
                h_i = tf.concat(2, [z, h_i])

            #SECOND DECODER (reverse order)
            #g_dec_depths=[512, 256, 256, 128, 128, 64, 64, 32, 32, 16, 1]
            if segan.feature_type == 'wavform':
                g_dec_depths = segan.g_enc_depths[:-1][::-1] + [1]
            elif segan.feature_type == 'logspec':
                g_dec_depths = segan.g_enc_depths[:-1][::-1] + [in_dims[2]]
            if is_ref:
                print('g_dec_depths: ', g_dec_depths)
            for layer_idx, layer_depth in enumerate(g_dec_depths):
                h_i_dim = h_i.get_shape().as_list()
                if h_i_dim[2]/2 == layer_depth:
                    out_shape = [h_i_dim[0], h_i_dim[1], layer_depth]
                else:
                    out_shape = [h_i_dim[0], h_i_dim[1] * 2, layer_depth]
                bias_init = None
                # deconv
                if segan.deconv_type == 'deconv':
                    if is_ref:
                        print('-- Transposed deconvolution type --')
                        if segan.bias_deconv:
                            print('Biasing deconv in G')
                    if segan.bias_deconv:
                        bias_init = tf.constant_initializer(0.)

                    if h_i_dim[2]/2 == layer_depth:
                        dilation = 1
                    else:
                        dilation = 2
                    h_i_dcv = deconv(h_i, out_shape, kwidth=kwidth, dilation=dilation,
                                     init=tf.truncated_normal_initializer(stddev=0.02),
                                     bias_init=bias_init,
                                     name='dec_{}'.format(layer_idx))
                elif segan.deconv_type == 'nn_deconv':
                    if is_ref:
                        print('-- NN interpolated deconvolution type --')
                        if segan.bias_deconv:
                            print('Biasing deconv in G')
                    if segan.bias_deconv:
                        bias_init = 0.
                    if h_i_dim[2]/2 == layer_depth:
                        dilation = 1
                    else:
                        dilation = 2
                    h_i_dcv = nn_deconv(h_i, kwidth=kwidth, dilation=2,
                                        init=tf.truncated_normal_initializer(stddev=0.02),
                                        bias_init=bias_init,
                                        name='dec_{}'.format(layer_idx))
                else:
                    raise ValueError('Unknown deconv type {}'.format(segan.deconv_type))
                if is_ref:
                    print('Deconv {} -> {}'.format(h_i.get_shape(),
                                                   h_i_dcv.get_shape()))
                h_i = h_i_dcv
                if layer_idx < len(g_dec_depths) - 1:
                    if do_prelu:
                        if is_ref:
                            print('-- Dec: prelu activation --')
                        h_i = prelu(h_i, ref=is_ref,
                                    name='dec_prelu_{}'.format(layer_idx))
                        if is_ref:
                            # split h_i into its components
                            alpha_i = h_i[1]
                            h_i = h_i[0]
                            alphas.append(alpha_i)
                    else:
                        if is_ref:
                            print('-- Dec: leakyrelu activation --')
                        h_i = leakyrelu(h_i)
                    # fuse skip connection
                    skip_ = skips[-(layer_idx + 1)]
                    if is_ref:
                        print('Fusing skip connection of '
                              'shape {}'.format(skip_.get_shape()))
                    h_i = tf.concat(2, [h_i, skip_])

                else:
                    if is_ref:
                        print('-- Dec: tanh activation --')
                    h_i = tf.tanh(h_i)
                    #h_i = prelu(h_i)

            wave = h_i
            if is_ref and do_prelu:
                print('Amount of alpha vectors: ', len(alphas))
            segan.gen_wave_summ = histogram_summary('gen_wave', wave)
            if is_ref:
                print('Amount of skip connections: ', len(skips))
                print('Last wave shape: ', wave.get_shape())
                print('*************************')
            segan.generator_built = True
            # ret feats contains the features refs to be returned
            ret_feats = [wave]
            G_output_nidm = wav_w.get_shape().as_list()
            G = wave
            if len(G_output_nidm) == 2:
                wavbatch = tf.expand_dims(wav_w, -1)
            elif len(G_output_nidm) == 3:
                wavbatch = wav_w
            else:
                raise ValueError('Clean wav input must be 2-D or 3-D')
            fft_error=[]
            #GG=G.eval(session=segan.sess)
            #print(GG.shape)
            for i in range(0, G_output_nidm[0]):
                G_target_frame = tf.reshape(G[i,:,:],[10,320])
                wav_target_frame = tf.reshape(wavbatch[i,:,:],[10,320])
                assert len(G_target_frame.get_shape().as_list()) == len(wav_target_frame.get_shape().as_list())
                G_fft = tf.fft(tf.cast(G_target_frame, tf.complex64))
                #print("G_fft shape is {}".format(G_fft.get_shape().as_list()))
                wav_fft = tf.fft(tf.cast(wav_target_frame, tf.complex64))
                G_fft_abs = tf.log(tf.maximum(tf.abs(G_fft)**2, 1e-12))
                wav_fft_abs = tf.log(tf.maximum(tf.abs(wav_fft)**2, 1e-12))
                fft_error.append(tf.sqrt(tf.reduce_mean(tf.squared_difference(G_fft_abs, wav_fft_abs))))
                #self.lsd_error.append(tf.sqrt(tf.reduce_mean(tf.squared_difference(G_fft_abs, wav_fft_abs))))
            print('the LSD error shape is {}'.format(len(fft_error)))
            lsd_error = tf.reduce_mean(fft_error)
            #print('lsd_error is {}'.format(lsd_error.get_shape().as_list()))

            if z_on:
                ret_feats.append(z)
            ret_feats.append(lsd_error)
            if is_ref and do_prelu:
                ret_feats += alphas
            return ret_feats






class AEGenerator_dropout(object):
    def __init__(self, segan):
        self.segan = segan

    def __call__(self, x, is_ref= True, spk=None, z_on=True, do_prelu=False):
        # TODO: remove c_vec
        """ Build the graph propagating (noisy_w) --> x
        On first pass will make variables.
        """
        segan = self.segan


        def make_z(shape, mean=0., std=1., name='z'):
            if is_ref:
                with tf.variable_scope(name) as scope:
                    z_init = tf.random_normal_initializer(mean=mean, stddev=std)
                    z = tf.get_variable("z", shape,
                                        initializer=z_init,
                                        trainable=False
                                        )
                    if z.device != "/device:GPU:0":
                        # this has to be created into gpu0
                        print('z.device is {}'.format(z.device))
                        assert False
            else:
                z = tf.random_normal(shape, mean=mean, stddev=std,
                                     name=name, dtype=tf.float32)
            return z

        if hasattr(segan, 'generator_built'):
            tf.get_variable_scope().reuse_variables()
            make_vars = False
        else:
            make_vars = True
        if is_ref:
            print('*** Building Generator ***')
        in_dims = x.get_shape().as_list()
        h_i = x
        if len(in_dims) == 2:
            h_i = tf.expand_dims(x, -1)
            h_i = tf.expand_dims(h_i, 1)
        if len(in_dims) == 3:
            h_i = tf.expand_dims(h_i, 1)
        elif len(in_dims) < 2 or len(in_dims) > 4:
            raise ValueError('Generator input must be 2-D or 3-D')
        kwidth = 31
        enc_layers = 7
        skips = []

        if is_ref and do_prelu:
            #keep track of prelu activations
            alphas = []
        with tf.variable_scope('g_ae'):

            ''' Can be conditioned on `y` or not '''
            ngf = 16
            # nc, nh, nw = self.n_shape
            # cc, ch, cw = self.c_shape
            layers = []
            #AE to be built is shaped:
            # enc ~ [16384x1, 8192x16, 4096x32, 2048x32, 1024x64, 512x64, 256x128, 128x128, 64x256, 32x256, 16x512, 8x1024]
            # dec ~ [8x2048, 16x1024, 32x512, 64x512, 8x256, 256x256, 512x128, 1024x128, 2048x64, 4096x64, 8192x32, 16384x1]
            #FIRST ENCODER
            print('g_enc_depths: ', segan.g_enc_depths)
            output = d_ops.conv2d(h_i, ngf, [11, 1], [1, 1, 1, 1], name="encoder_1")
            layers.append(output)

            layer_coder = [
                ngf * 1,  # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
                ngf * 2,  # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
                ngf * 2,  # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
                ngf * 4,  # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
                ngf * 4,  # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
                ngf * 8,  # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
                ngf * 8,  # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
                ngf * 16,  # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
                ngf * 16,  # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
                ngf * 32,  # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
                # ngf * 32,
                ngf * 64
            ]

            for out_channels in layer_coder:
                name = "encoder_%d" % (len(layers) + 1)
                rectified = d_ops.activation(layers[-1], 'prelu', name + '_activation')
                # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
                output = d_ops.conv2d(rectified, out_channels, [11, 1], [1, 1, 2, 1], name=name)
                output = tf.layers.batch_normalization(output, axis=1, name=name + '_bn')
                layers.append(output)


            #SECOND DECODER (reverse order)
            #g_dec_depths=[512, 256, 256, 128, 128, 64, 64, 32, 32, 16, 1]
            layer_decoder = [
                (ngf * 64, 0.5),
                # (ngf * 32, 0.5),
                (ngf * 32, 0.5),  # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
                (ngf * 16, 0.5),  # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
                (ngf * 16, 0.5),  # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
                (ngf * 8, 0.5),  # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
                (ngf * 8, 0.5),  # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
                (ngf * 4, 0.5),  # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
                (ngf * 4, 0.5),  # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
                (ngf * 2, 0.5),  # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
                (ngf * 2, 0.0),  # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
                (ngf * 1, 0.0),  # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
            ]

            num_encoder_layers = len(layers)
            for decoder_layer, (out_channels, dropout) in enumerate(layer_decoder):
                skip_layer = num_encoder_layers - decoder_layer - 1
                name = "decoder_%d" % (skip_layer + 1)
                if decoder_layer == 0:
                    # first decoder layer doesn't have skip connections
                    # since it is directly connected to the skip_layer
                    input = layers[-1]
                else:
                    input = tf.concat([layers[-1], layers[skip_layer]], axis=1)

                rectified = d_ops.activation(input, 'prelu', name + '_activation')
                # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
                output = d_ops.deconv_up(rectified, out_channels, [11, 1], [1, 1, 1, 1], name=name)
                output = tf.layers.batch_normalization(output, axis=1, name=name + '_bn')

                # if dropout > 0.0:
                #     output = tf.nn.dropout(output, keep_prob=1 - dropout)

                layers.append(output)

            # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
            input = tf.concat([layers[-1], layers[0]], axis=1)
            name = 'decoder_1'
            rectified = d_ops.activation(input, 'prelu', name + '_activation')
            output = d_ops.conv2d(rectified, ngf, [11, 1], [1, 1, 1, 1], name=name)
            layers.append(output)

            name = 'output_layers'
            rectified = d_ops.activation(layers[-1], 'prelu', name + '_activation')
            output = d_ops.conv2d(rectified, 1, [3, 1], [1, 1, 1, 1], name=name)
            layers.append(output)

            output_dims = layers[-1].get_shape().as_list()
            ret_feats = tf.reshape(layers[-1], [output_dims[0], output_dims[2], output_dims[3]])

            return ret_feats


