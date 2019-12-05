"""
Summary:  Calculate PESQ and overal stats of enhanced speech. 
Author:   Qiuqiang Kong
Created:  2017.12.22
Modified: Yupeng Shi
"""
import argparse
import os
import csv
import numpy as np
import cPickle
import matplotlib.pyplot as plt
import subprocess as sub
import config as cfg

def plot_training_stat(args):
    """Plot training and testing loss. 
    
    Args: 
      workspace: str, path of workspace. 
      tr_snr: float, training SNR. 
      bgn_iter: int, plot from bgn_iter
      fin_iter: int, plot finish at fin_iter
      interval_iter: int, interval of files. 
    """
    workspace = args.workspace
    tr_snr = args.tr_snr
    bgn_iter = args.bgn_iter
    fin_iter = args.fin_iter
    interval_iter = args.interval_iter

    tr_losses, te_losses, iters = [], [], []
    
    # Load stats. 
    stats_dir = os.path.join(workspace, "training_stats", "%ddb" % int(tr_snr))
    for iter in xrange(bgn_iter, fin_iter, interval_iter):
        stats_path = os.path.join(stats_dir, "%diters.p" % iter)
        dict = cPickle.load(open(stats_path, 'rb'))
        tr_losses.append(dict['tr_loss'])
        te_losses.append(dict['te_loss'])
        iters.append(dict['iter'])
        
    # Plot
    line_tr, = plt.plot(tr_losses, c='b', label="Train")
    line_te, = plt.plot(te_losses, c='r', label="Test")
    plt.axis([0, len(iters), 0, max(tr_losses)])
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend(handles=[line_tr, line_te])
    plt.xticks(np.arange(len(iters)), iters)
    plt.show()


def calculate_pesq(args):
    """Calculate PESQ of all enhaced speech. 
    
    Args:
      workspace: str, path of workspace. 
      speech_dir: str, path of clean speech. 
      te_snr: float, testing SNR. 
    """
    workspace = args.workspace
    speech_dir = args.speech_dir
    te_enh = args.te_enh
    #te_snr = args.te_snr
    #snr = cfg.SNR
    model_name = args.model_name
    
    # Remove already existed file. 
    os.system('rm _pesq_itu_results.txt')
    os.system('rm _pesq_results.txt')
    
    # Calculate PESQ of all enhaced speech. 
    if te_enh == "enhance" :
        enh_speech_dir = os.path.join(workspace, "enh_wavs", "test", "%s" % model_name)
    elif te_enh == "test" :
        enh_speech_dir = os.path.join(workspace, "mixed_audios", "test")
        
    names = os.listdir(enh_speech_dir)
    for (cnt, na) in enumerate(names):
        print(cnt, na)
        
        enh_path = os.path.join(enh_speech_dir, na)
        
        #speech_na = na.split('.')[0]
        enh_na = na.split('_')
        speech_path = os.path.join(speech_dir, "%s_%s.wav" % (enh_na[0], enh_na[1]))
        
        # Call executable PESQ tool. =
        cmd = ' '.join(["./pesq", speech_path, enh_path, "+16000"])
        os.system(cmd)
	#subout=sub.Popen('./pesq',stdout=sub.PIPE,shell=True,cwd='/home/szuer/SE_CNN_DNN')  
    
        
        
def get_stats(args):
    """Calculate stats of PESQ. 
    """
    #compute_mode = args.compute_mode
    #pesq_path = "_pesq_results.txt"
    mode = 'fcn1'
    pesq_path = "tablescore.txt"
    with open(pesq_path, 'rb') as f:
        reader = csv.reader(f, delimiter='\t')
        lis = list(reader)
        
    pesq_dict = {}
    lsd_dict = {}
    stoi_dict = {}
    segsnr_dict = {}
    for i1 in xrange(1, len(lis) - 1):
        li = lis[i1]
        na = li[0]
        pesq = float(li[1])
        lsd = float(li[2])
        stoi = float(li[3])
        segsnr = float(li[4])
        noise_type = na.split('_')[2]
        if noise_type not in pesq_dict.keys():
            pesq_dict[noise_type] = [pesq]
            lsd_dict[noise_type] = [lsd]
            stoi_dict[noise_type] = [stoi]
            segsnr_dict[noise_type] = [segsnr]
        else:
            pesq_dict[noise_type].append(pesq)
            lsd_dict[noise_type].append(lsd)
            stoi_dict[noise_type].append(stoi)
            segsnr_dict[noise_type].append(segsnr)
        
    pesq_avg_list, pesq_std_list = [], []
    lsd_avg_list, lsd_std_list = [], []
    stoi_avg_list, stoi_std_list = [], []
    segsnr_avg_list, segsnr_std_list = [], []
    #http://www.runoob.com/python/att-string-format.html
    #print('-'*100)
    t = "{:^80}"
    print(t.format("%s" %mode))
    print('-'*100)
    f = "{0:^16} {1:^16} {2:^16} {3:^16} {4:^16}"
    print(f.format("Noise types", "PESQ", "LSD", "STOI", "segSNR"))
    print("-"*100)
    for noise_type in pesq_dict.keys():
        pesqs = pesq_dict[noise_type]
        avg_pesq = np.mean(pesqs)
        std_pesq = np.std(pesqs)
        pesq_avg_list.append(avg_pesq)
        pesq_std_list.append(std_pesq)
        
        lsds = lsd_dict[noise_type]
        avg_lsd = np.mean(lsds)
        std_lsd = np.std(lsds)
        lsd_avg_list.append(avg_lsd)
        lsd_std_list.append(std_lsd)
        
        stois = stoi_dict[noise_type]
        avg_stoi = np.mean(stois)
        std_stoi = np.std(stois)
        stoi_avg_list.append(avg_stoi)
        stoi_std_list.append(std_stoi)
        
        segsnrs = segsnr_dict[noise_type]
        avg_segsnr = np.mean(segsnrs)
        std_segsnr = np.std(segsnrs)
        segsnr_avg_list.append(avg_segsnr)
        segsnr_std_list.append(std_segsnr)
        
        print(f.format(noise_type, "%.3f +- %.3f" % (avg_pesq, std_pesq), "%.3f +- %.3f" % (avg_lsd, std_lsd),
                       "%.3f +- %.3f" % (avg_stoi, std_stoi), "%.3f +- %.3f" % (avg_segsnr, std_segsnr)))
    print("-"*100)
    print(f.format("Avg.", "%.3f +- %.3f" % (np.mean(pesq_avg_list), np.mean(pesq_std_list)), 
                   "%.3f +- %.3f" % (np.mean(lsd_avg_list), np.mean(lsd_std_list)), 
                   "%.3f +- %.3f" % (np.mean(stoi_avg_list), np.mean(stoi_std_list)), 
                   "%.3f +- %.3f" % (np.mean(segsnr_avg_list), np.mean(segsnr_std_list))))
    print('\n')

def get_standars(args):
    """Calculate stats of PESQ. 
    """
    #compute_mode = args.compute_mode
    #pesq_path = "_pesq_results.txt"
    mode = 'm_vgg1'
    pesq_path = "st_tablescore.txt"
    with open(pesq_path, 'rb') as f:
        reader = csv.reader(f, delimiter='\t')
        lis = list(reader)
        
    pesq_dict = {}
    lsd_dict = {}
    stoi_dict = {}
    segsnr_dict = {}
    for i1 in xrange(1, len(lis) - 1):
        li = lis[i1]
        na = li[0]
        pesq = float(li[1])
        lsd = float(li[2])
        stoi = float(li[3])
        segsnr = float(li[4])
        noise_type = na.split('_')[2]
        if noise_type not in pesq_dict.keys():
            pesq_dict[noise_type] = [pesq]
            lsd_dict[noise_type] = [lsd]
            stoi_dict[noise_type] = [stoi]
            segsnr_dict[noise_type] = [segsnr]
        else:
            pesq_dict[noise_type].append(pesq)
            lsd_dict[noise_type].append(lsd)
            stoi_dict[noise_type].append(stoi)
            segsnr_dict[noise_type].append(segsnr)
        
    pesq_avg_list, pesq_std_list = [], []
    lsd_avg_list, lsd_std_list = [], []
    stoi_avg_list, stoi_std_list = [], []
    segsnr_avg_list, segsnr_std_list = [], []
    #http://www.runoob.com/python/att-string-format.html
    #print('-'*100)
    t = "{:^80}"
    print(t.format("%s" %mode))
    print('-'*100)
    f = "{0:^16} {1:^16} {2:^16} {3:^16} {4:^16}"
    print(f.format("Noise types", "PESQ", "LSD", "STOI", "segSNR"))
    print("-"*100)
    for noise_type in pesq_dict.keys():
        pesqs = pesq_dict[noise_type]
        avg_pesq = np.mean(pesqs)
        std_pesq = np.std(pesqs)
        pesq_avg_list.append(avg_pesq)
        pesq_std_list.append(std_pesq)
        
        lsds = lsd_dict[noise_type]
        avg_lsd = np.mean(lsds)
        std_lsd = np.std(lsds)
        lsd_avg_list.append(avg_lsd)
        lsd_std_list.append(std_lsd)
        
        stois = stoi_dict[noise_type]
        avg_stoi = np.mean(stois)
        std_stoi = np.std(stois)
        stoi_avg_list.append(avg_stoi)
        stoi_std_list.append(std_stoi)
        
        segsnrs = segsnr_dict[noise_type]
        avg_segsnr = np.mean(segsnrs)
        std_segsnr = np.std(segsnrs)
        segsnr_avg_list.append(avg_segsnr)
        segsnr_std_list.append(std_segsnr)
        
        
    #print("-"*100)
    
    st_pesq_list = []
    st_lsd_list = []
    st_stoi_list = []
    st_segsnr_list = []
    noise_ty = pesq_dict.keys()
    for i in range(len(pesq_avg_list)):
        
        st_pesq = abs((pesq_avg_list[i]-np.mean(pesq_avg_list))/np.mean(pesq_std_list))
        st_pesq_list.append(st_pesq)
        
        st_lsd = abs((lsd_avg_list[i]-np.mean(lsd_avg_list))/np.mean(lsd_std_list))
        st_lsd_list.append(st_lsd)
        
        st_stoi = abs((stoi_avg_list[i]-np.mean(stoi_avg_list))/np.mean(stoi_std_list))
        st_stoi_list.append(st_stoi)
        
        st_segsnr = abs((segsnr_avg_list[i]-np.mean(segsnr_avg_list))/np.mean(segsnr_std_list))
        st_segsnr_list.append(st_segsnr)
        
        print(f.format(noise_ty[i], "%.3f" % st_pesq, "%.3f" % st_lsd,
                       "%.3f" % st_stoi, "%.3f" % st_segsnr))
    print("-"*100)   
    print(f.format("Avg.", "%.3f" % np.mean(st_pesq_list), 
                   "%.3f" % np.mean(st_lsd_list), 
                   "%.3f" % np.mean(st_stoi_list), 
                   "%.3f" % np.mean(st_segsnr_list)))
    print('\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_plot_training_stat = subparsers.add_parser('plot_training_stat')
    parser_plot_training_stat.add_argument('--workspace', type=str, required=True)
    parser_plot_training_stat.add_argument('--tr_snr', type=float, required=True)
    parser_plot_training_stat.add_argument('--bgn_iter', type=int, required=True)
    parser_plot_training_stat.add_argument('--fin_iter', type=int, required=True)
    parser_plot_training_stat.add_argument('--interval_iter', type=int, required=True)

    parser_calculate_pesq = subparsers.add_parser('calculate_pesq')
    parser_calculate_pesq.add_argument('--workspace', type=str, required=True)
    parser_calculate_pesq.add_argument('--speech_dir', type=str, required=True)
    parser_calculate_pesq.add_argument('--te_enh', type=str, required=True)
    parser_calculate_pesq.add_argument('--model_name', type=str, required=True)
    #parser_calculate_pesq.add_argument('--te_enh', type=str, required=True)
    
    parser_get_stats = subparsers.add_parser('get_stats')
    parser_get_standars = subparsers.add_parser('get_standars')
    
    #parser_get_stats.add_argument('--te_enh', type=str, required=True)
    
    args = parser.parse_args()
    
    if args.mode == 'plot_training_stat':
        plot_training_stat(args)
    elif args.mode == 'calculate_pesq':
        calculate_pesq(args)
    elif args.mode == 'get_stats':
        get_stats(args)
    elif args.mode == 'get_standars':
        get_standars(args)
    else:
        raise Exception("Error!")
