import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset
from utils import TextProcess
import pandas as pd
import numpy as np
from AudioTransform import AudioUtil

sample_rate = 16000
freq_mask = 15
time_mask = 80

class SpecAugment(nn.Module):
    def __init__(self,time_mask,freq_mask):
        super(SpecAugment,self).__init__()
        self.transform  = nn.Sequential(torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask),
                                        torchaudio.transforms.TimeMasking(time_mask_param=time_mask))
        
    def forward(self,x):
        return self.transform(x)
    
class LogMelSpec(nn.Module):
    def __init__(self,sample_rate):
        super(LogMelSpec,self).__init__()
        self.transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,win_length=160,hop_length=80,n_mels = 64)

    def forward(self,x):
        x = self.transform(x)
        x = np.log(x + 1e-14)
        return x

def createMelSpec(sample_rate):
    return LogMelSpec(sample_rate=sample_rate)


class Data(Dataset):
    data_hparams = {
        'sample_rate' : sample_rate,
        'time_mask' : time_mask,
        'freq_mask' : freq_mask
    }
    def __init__(self,json_file,sample_rate,time_mask,freq_mask):
        self.data = pd.read_json(json_file)
        self.audioTransform = nn.Sequential(SpecAugment(time_mask= time_mask,freq_mask=freq_mask),
                                            LogMelSpec(sample_rate=sample_rate))
        self.AudioPreProcessor = AudioUtil()
        self.textProcessor = TextProcess()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.item()
        
        audioFilePath = self.data.key.iloc[index]
        label = self.textProcessor.text_to_int(self.data.text.iloc[index])

        # waveform, _ = torchaudio.load(audioFilePath)
        waveform, _ = self.AudioPreProcessor.openAudioFile(audioFilePath)
        
        waveform = self.AudioPreProcessor.trunc(wave= waveform)
        waveform = self.AudioPreProcessor.padder(wave= waveform)

        spectrogram = self.audioTransform(waveform)
        # spectrogram_len = spectrogram.shape[-1]//2
        # spectrogram_len = (spectrogram.shape[0],)
        # spectrogram_len = (64,)
        spectrogram_len = (1,)
        label_len = len(label)

        return spectrogram,label,spectrogram_len,label_len


def Padding(data):
    "This function we will pad the data to make same batch size"

    spectrograms  = []
    labels = []
    spec_lengths = []
    label_lengths = []


    """
    we have shape of spectrograms = [1,64,x]
    now this x can vary in corresponding to the length of audio input, so in order to have batches of same size we will use rnn.pad_sequence

    in order to use pad_sequence the last dimension of all the elements should be same.

    so, we will take 64 to last dim by firstly squeezing and then transposing the sequence.
    so shape of all the spectrograms will becom [x,64]
    now we will pad these spectrograms and keep batchfirst= True
    we will also unsqueeze the spectrograms after the padding and to take back the spectrograms to there original shape we will again transpose them
    shape after paddding(without unsqueeze and transpose) -> [batch,y,64]  y -> max of all the x's (decided by the rnn.pad_sequence funciton)
    shape after paddding(with unsqueeze and transpose) -> [batch,1,64,y], which is the original shape of spectrogram

    """
    for (spectrogram,label, spec_length,label_length) in data:
        spectrograms.append(spectrogram.squeeze(0).transpose(0,1))
        labels.append(torch.Tensor(label))
        spec_lengths.append(spec_length)
        label_lengths.append(label_length)

    spectrograms = nn.utils.rnn.pad_sequence(spectrograms,batch_first=True).transpose(1,2).unsqueeze(1)
    labels = nn.utils.rnn.pad_sequence(labels,batch_first= True)

    return spectrograms, labels,spec_lengths,label_lengths

