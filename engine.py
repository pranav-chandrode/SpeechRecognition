import torch
from model import SpeechModel
from dataset import Data, Padding
from torch.utils.data import DataLoader
from decoder import CTCBeamDecoder
import pyaudio    # conda install -c anaconda pyaudio -> used this to install pyaudion
import time
import threading

h_param = SpeechModel.hyper_parameters

Testmodel = SpeechModel(**h_param)
checkpoint_path = r"speech_logger\\Speech_loggs\\version_8\\checkpoints\\epoch=1-step=75.ckpt"

model_state_dict = torch.load(checkpoint_path)['state_dict']
Testmodel.load_state_dict(model_state_dict,strict=False)  
# donot use strict= False this may lead tp wrong predicitons
# solutions :- https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/13 
Testmodel.eval()

class Listen():
    def __init__(self,sample_rate = 16000,record_time = 2):
        self.sample_rate = sample_rate
        self.record_time = record_time
        self.reciever = pyaudio.PyAudio()
        self.CHUNK = 1024
        self.channels = 1
        self.stream = self.reciever.open(format= pyaudio.paInt16, rate= self.sample_rate,channels=self.channels,
                                         input=True,output= True, frames_per_buffer=self.CHUNK)

    
    def reader(self,frames):
        while True:
            data = self.stream.read(self.CHUNK,exception_on_overflow=False)
            frames.append(data)
            time.sleep(0.01)
        
    def run(self,frame):
        thread = threading.Thread(target=self.reader, args=(frame,),daemon=True)
        thread.start()
        print("Listnign Speech!!!\n")


































# d_params = Data.data_hparams
# train_data = Data('save_json/train.json',**d_params)
# val_data = Data('save_json/test.json',**d_params)


# train_loader = DataLoader(dataset= train_data,batch_size= 64,shuffle=True,collate_fn=Padding)
# val_loader = DataLoader(dataset=val_data,batch_size= 64,shuffle= False,collate_fn=Padding)

# for batch in train_loader:
#     spectrograms , lables, spec_len , label_len = batch
#     h0,c0 = Testmodel.hidden_initialize(64)
#     out, _ = Testmodel(spectrograms,(h0,c0))
#     print(out.shape)

    


