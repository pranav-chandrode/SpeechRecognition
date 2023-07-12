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

folders = os.listdir('LibriSpeech\\dev-clean')

maxi = float(0)

for folder in folders:
    print(folder)
    files = os.listdir('LibriSpeech\dev-clean\\' +folder)
    print(files)
    for audioFile in files:
        print(audioFile)
        flacFiles = glob.glob('LibriSpeech\dev-clean\\' + folder + '\\'+ audioFile + "\\*.flac")
        transctipt = glob.glob('LibriSpeech\dev-clean\\' + folder + '\\'+ audioFile + "\\*.txt")
        with open(transctipt[0] , 'r') as script:
            content = script.read().lower()
        
        # print(content)
        for i,flac in enumerate(flacFiles):
            file_name = flac.split('\\')[4].split('.')[0]
            # print(file_name)
            data, samplerate = sf.read(flac)
            maxi = max(maxi,data.shape[0]/samplerate)
            # freq, time, Sxx = signal.spectrogram(data, samplerate, scaling='spectrum')
            # plt.pcolormesh(time, freq, Sxx)
            # Pxx, freqs, bins, im = plt.specgram(data, Fs=samplerate)
            
            text = content.split('\n')[i].split(file_name)[1]
            

print(maxi)
# with open('8842-302196.trans.txt') as f:
#     content = f.read()

# print(content.split('8842-302196-0000')[1])