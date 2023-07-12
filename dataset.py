import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset
from utils import TextProcess
import pandas as pd


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
        self.transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate)

    def forward(self,x):
        x = self.transform(x)
        return x

class Data(Dataset):
    def __init__(self,json_file,sample_rate,time_mask,freq_mask):
        self.data = pd.read_json(json_file)
        self.audioTransform = nn.Sequential(SpecAugment(time_mask= time_mask,freq_mask=freq_mask),
                                            LogMelSpec(sample_rate=sample_rate))
        
        self.textProcessor = TextProcess()

    def __len__(data):
        return len(data)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.item()
        
        audioFilePath = self.data.key.iloc[index]
        label = self.textProcessor(self.data.text.iloc[index])

        waveform, _ = torchaudio.load(audioFilePath)
        spectrogram = self.audioTransform(waveform)
        spectrogram_len = spectrogram.shape[-1]//2
        label_len = len(label)

        return spectrogram,label,spectrogram_len,label_len

    
        