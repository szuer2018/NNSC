# NNSC
speech compensation (BWE &amp; PLC) based on deep neural networks
/**********************************************************************************************************************************/

This document includes brief instructions on how to build an NN-based processing chain for speech compensation.
- Author:  Yupeng Shi, Nengheng Zheng, Yuyong Kang, and Weicong Rong
- e-mail:  szucieer@gmail.com
- Date: 05/12/2019

/*********************************************************************************************************************************/

# Speech compensation based deep neural networks (DNN & GANs)
This project is a Python implementation of Speech Compensation (SC) based on deep neural networks plus extra pre- and post-processing. 
***

## Dependence
            
                tensorflow_gpu          0.12.1
                librosa                 0.6.3  or later
                Soundfile               0.10.2 or later
                numpy                  1.12.1
                scipy                   0.18.1  or later
                tensorboardX            1.2    or later
                python                  2.7    

 We suggest to install anaconda3 in Linux, and you can install those dependences by conda or pip.
***
## Systematic description

#### 1. Data preparation
###### 1.1 Generating stimulated wav files
  a) In this project, you can train the models with the impaired sampels simulated by **Low-pass filters** and **OPUS codec** . The sample rate of the NN-based **SC** system is **16kHz**.
            
  b) Some python scripts in are designed for processing the raw data. For generating narrowband signal, Please refer to **boneloss_lowpass.py** in **./GANSC**. As for obtaining pack loss simulation, Opus codec codes can be download from GitHub (https://github.com/xiph/opus). Besides, the ITU-T Software Tool Library (G.191) ((https://github.com/openitu/STL)) can also be implemented to generate telephone transmitted narrowband speech.
###### 1.2 preprocessing the wav files and storing the features in **.tfrecords** files for GANs while **.h5** for DNN
  To accelerate the whole NN training, parallel computing have been adopted in the data preprocessing. **Short Time Fourier Transform (STFT)** or **waveform chunks** are extracted for NN input. More details can be referred to **./GANSC/make_tfrecords.py** and **./DNNSC/prepare_data.py**.
#### 2. NN training
1. Put the **.tfrecords** or **.h5** files to the specific path where ./GANSC/data_loader.py or ./DNNSC/main_dnn.py can load the training or validating data for training and validating the NN-based SC model;

2. training a NN-based SC model with specific hyperparameter:
                
		    quick start: $ ./GANSC/train_gan.sh or ./DNNSC/runme.sh

           If you want to modify the hyperparameters, see the help information by:
                
		    $ python train.py --help
		    

               
        
#### 3. NN testing
1. Put the testing wav files in the specific path;
2. Set the required parameters for testing.
e.g.,
                
        $ bash ./GANSC/clean_wav.sh or ./DNNSC/runme.sh
        
#### 4. More details
You can visualize the train process using **tensorboard**:
                
        $ cd projcet_path
        $ tensorboard --logdir=$log_path
                
	    and then ,open the browser and enter: IP:6006
	    An example for remote server, if you are training NN in local PC, the IP can be localhost:
        e.g.,
		10.10.88.47:6006
		localhost:6006                
***

#### 5. Some references
In this project, the GANs struction is modified from the proposed model by Santiago et al..
If you find it useful to your research, pleade cite the following papers:

[1] Y. Xu, JunDu, L. R. Dai, and C. H. Lee, "An Experimental Study on Speech Enhancement Based on Deep Neural Networks," IEEE signal processing letters, pp. 65-68,vol.21,no. 1, JaN. 2014.

[2] S. Pascual, A. Bonafonte, and J. Serrà, “SEGAN: Speech enhancement generative adversarial network,” In INTERSPEECH, 2017.

[3] Y. P. Shi, N. H. Zheng, Y. Y. Kang, and W. C. Rong, "Speech Loss Compensation by Generative Adversarial Networks," In APSIPA, 2019.

