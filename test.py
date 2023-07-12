# import soundfile as sf

# # this will convert the flac file into numpy array
# data, samplerate = sf.read(r'LibriSpeech\dev-clean\422\122949\422-122949-0000.flac')

# print(data.shape)

# # len of audio  = data.shape[0] / samplerate
# # converting flac file to wav file 
# sf.write('writing_file_output.wav', data, samplerate)


import numpy as np
import soundfile as sf
import scipy.signal as signal
import matplotlib.pyplot as plt


import sys
import glob
import os
import random
import json

path = 'LibriSpeech\\dev-clean'
folders = os.listdir(path)

maxi = float(0)
json_data = []

for folder in folders:
    print(folder)
    subfolders = os.listdir(path + '\\' +folder)
    print(subfolders)
    for subfolder in subfolders:
        print(subfolders)
        flacFiles = glob.glob(path + '\\' + folder + '\\'+ subfolder + "\\*.flac")
        transctipt = glob.glob(path + '\\' + folder + '\\'+ subfolder + "\\*.txt")
        with open(transctipt[0] , 'r') as script:
            content = script.read().lower()
        
        # print(content)
        for i,flac in enumerate(flacFiles):
            file_name = flac.split('\\')[4].split('.')[0]
            # print(file_name)
            data, samplerate = sf.read(flac)
            maxi = max(maxi,data.shape[0]/samplerate)
            text = content.split('\n')[i].split(file_name)[1]
            text = text.strip()
            json_data.append(
                { "key" : "dest_wav_file" + '\\' + file_name + ".wav",
                    "text" : text
                }
            )

# print(maxi)

random.shuffle(data) 
print("creating Json file !!!")

save_path = 