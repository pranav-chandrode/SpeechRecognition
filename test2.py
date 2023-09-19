# # import torch
# # import torch.nn as nn

# # m = nn.Conv1d(128, 128, 2, stride=2)
# # input = torch.randn(2, 1, 128, 879)
# # input = input.squeeze()

# # output = m(input)
# # print(output.shape)


# import pandas as pd
# from utils import TextProcess
# # data = pd.read_json('save_json/test.json')

# # idx = 1
# # text_processor = TextProcess()

# # print(text_processor.text_to_int(data.text.iloc[idx]))


# from dataset import Data, Padding
# import torch
# from torch.utils.data import DataLoader
# import torch.nn as nn

# ok = Data('save_json/test.json',sample_rate=16000,time_mask=80,freq_mask=15)
# # print(ok.__getitem__(1))
# batch = ok.__getitem__(3)
# # spectrograms,label ,spec_len ,label_len = batch
# # print(spectrograms.shape)
# # print(label)
# # print(torch.Tensor(label))
# # print(type(label[0]))
# spectrograms,label ,spec_len ,label_len = Padding([batch])

# print(spectrograms.shape)

# # loss = criterion()
# # import sys
# # sys.setrecursionlimit(10000)
# # d_params = Data.data_hparams
# # val_dataset = Data('save_json/test.json', **d_params)


# # data = DataLoader(dataset= val_dataset,batch_size= 32,
# #                           shuffle=False,collate_fn=Padding)

# # print(data)

# # from model import SpeechModel

# # criterion = nn.CTCLoss()
# # h_params = SpeechModel.hyper_parameters
# # model = SpeechModel(**h_params)
# # # print("Printing model")
# # # print(model)

# # h0, c0 = model.hidden_initialize(64)
# # out = model(spectrograms,(h0,c0))
# # print(out)


# import torch.nn.functional as F
# import torch
# import numpy as np

# a = np.arange(1,11).reshape(2,5)
# # a.reshape((10,50))
# print(a)
# a = torch.Tensor(a)
# a = F.softmax(a,dim=1)
# print(a)

import torchaudio

transform  = torchaudio.transforms.TimeMasking(time_mask_param=28)
transform2 = torchaudio.transforms.FrequencyMasking(freq_mask_param= 20)

wave, _ = torchaudio.load("dest_wav_file/84-121123-0000.wav")

print(wave.shape)
wave = transform(wave)
wave = transform2(wave)
print(wave.shape)