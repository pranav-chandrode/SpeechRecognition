from typing import Any
import torch
import torchaudio
from model import SpeechModel
from dataset import Data, Padding ,createMelSpec
from torch.utils.data import DataLoader
from AudioTransform import AudioUtil
from decoder import CTCBeamDecoder
import pyaudio    # conda install -c anaconda pyaudio -> used this to install pyaudio
import wave
import time
import threading
import argparse



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
        self.context_length = context_length * 50

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
            fname = self.save(audio,"Audio.wav")
            waveform , _ = torchaudio.load(fname)
            print(f"waveform  shape = {waveform.shape}")
            waveform = self.AudioTrucPad.trunc(wave=waveform)
            waveform = self.AudioTrucPad.padder(wave=waveform)

            logMel = self.LogMelCreater(waveform)
            out,_ = self.Testmodel(logMel,self.hidden)
            print(f"out shape = {out.shape}")
            # out = torch.argmax(out, dim = 2)
            # print(f"out shape = {out.shape}")
            self.out_arg = out  if self.out_arg is None else torch.cat((self.out_arg,out),dim=1)
            print(self.out_arg.shape)
            # self.out_arg.squeeze(0)
            print(self.out_arg.shape)
            # self.out_arg = self.out_arg.transpose(0,1)
            results = self.beam_result(self.out_arg)
            current_context_length = self.out_arg.shape[1] / 50 
            if self.out_arg.shape[1] > self.context_length:
                self.out_arg = None
            return results, current_context_length

    def infernce(self, action):
        while True:
            if len(self.audio_list) < 5:
                continue
            else:
                pred_list = self.audio_list.copy()
                self.audio_list.clear()
                action(self.predict(pred_list))
            time.sleep(0.05)

    def run(self,action):
        self.listner.run(self.audio_list)
        thread = threading.Thread(target=self.infernce,args=(action,),daemon=True)
        thread.start()

    
class DemoAction():
    def __init__(self):
        self.asr_result = ""
        self.current_beam = ""

    def __call__(self,x):
        results, current_context_len = x
        self.current_beam = results
        # print(f"result type = {type(results)}")
        # print(f"asr_result type = {type(self.asr_result)}")
        transcript = "".join(str(self.asr_result) + str(results.split()))
        print("printing transcript !!!")
        print(transcript)
        if current_context_len > 10:
            self.asr_result = transcript

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description= "Testing Speech Recognition System")
    parser.add_argument("--model_path_ckpt",type=str,default=None, required=True, 
                        help="Enter the checkpoint path for the model")
    parser.add_argument("--kenlm_path",type= str, default=None, required=False,
                         help= "Path to your language model" )
    
    args = parser.parse_args()

    asr_engine = SpeechListner(args.model_path_ckpt, args.kenlm_path)
    action = DemoAction()

    asr_engine.run(action = action)
    obj = threading.Event()
    obj.wait()