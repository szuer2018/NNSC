import re
import os
enh_path = '/media/szuer/8e1f3765-286d-4d57-a4d7-d64aa2a176e0/home/amax/boneloss_rawdata/noisy_trainset_56spk_wav_16k/p234_001.wav'
speech_path = '/media/szuer/8e1f3765-286d-4d57-a4d7-d64aa2a176e0/home/amax/boneloss_rawdata/clean_trainset_56spk_wav_16k/p234_001.wav'

# Call executable PESQ tool. =
cmd = ' '.join(["/home/szuer/pesq", speech_path, enh_path, "+16000"])
# strout = os.system('./pesq.txt')
# os.system(cmd)
r = os.popen(cmd)
info = r.readlines()
print info[-1]


print float(re.findall(r'\d+\.?\d*', info[-1])[0])