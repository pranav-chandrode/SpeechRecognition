import torch
import torchaudio
from model import SpeechModel
from dataset import Data, Padding ,createMelSpec
from torch.utils.data import DataLoader
from AudioTransform import AudioUtil
from decoder import CTCBeamDecoder
import pyaudio    # conda install -c anaconda pyaudio -> used this to install pyaudion
import wave
import time
import threading


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
        print("Listning Speech!!!\n")


class SpeechListner():
    def __init__(self,checkpoint_path,kenlm_file,context_length = 10):

        h_param = SpeechModel.hyper_parameters
        self.Testmodel = SpeechModel(**h_param)
        # checkpoint_path = r"speech_logger\\Speech_loggs\\version_8\\checkpoints\\epoch=1-step=75.ckpt"

        model_state_dict = torch.load(checkpoint_path)['state_dict']
        self.Testmodel.load_state_dict(model_state_dict,strict=False)  
        # donot use strict= False this may lead tp wrong predicitons
        # solutions :- https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/13 
        self.Testmodel.eval().to('cpu')
        self.listner = Listen()
        self.audio_list = list()
        self.AudioTrucPad = AudioUtil()
        self.LogMelCreater = createMelSpec(16000)
        self.hidden = self.Testmodel.hidden_initialize(1)
        self.beam = ""
        self.out_arg = None
        self.beam_result = CTCBeamDecoder(beam_size=100, kenlm_path=kenlm_file)

    def save(self,waveforms,file_name):
        wf = wave.open(file_name,'wb')
        wf.setnchannels(1)
        wf.setsampwidth(self.listner.reciever.get_sample_size(format= pyaudio.paInt16))
        wf.setframerate(16000)
        wf.writeframes(b"".join(waveforms))
        wf.close()
        return file_name
    
    def predict(self, audio):
        with torch.no_grad():
            fname = self.save(audio)
            waveform , _ = torchaudio.load(fname)
            waveform = self.AudioTrucPad.trunc(wave=waveform)
            waveform = self.AudioTrucPad.padder(wave=waveform)

            logMel = self.LogMelCreater(waveform)
            out,_ = self.Testmodel(logMel,self.hidden)
            out = torch.argmax(out, dim = 2)
            self.out_arg = out  if self.out_arg is None else torch.cat((self.out_arg,out),dim=1)

            self.out_arg.squeeze(0)
            self.out_arg = self.out_arg.transpose(0,1)

            




























# ok = Data('save_json/test.json',sample_rate=16000,time_mask=80,freq_mask=15)
# # print(ok.__getitem__(1))
# batch = ok.__getitem__(3)
# spectrograms,label ,spec_len ,label_len = batch
# h0,c0 = Testmodel.hidden_initialize(1)
# out,_ = Testmodel(spectrograms,(h0,c0))
# Decoder_obj = CTCBeamDecoder()
# print(out.shape)
# out = out.squeeze(0)
# print(out.shape)
# text = Decoder_obj.deconding(out)
# print(text)
