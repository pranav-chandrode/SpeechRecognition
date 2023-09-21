from typing import Any
import torch
from utils import TextProcess
# import ctcdecode
from pyctcdecode import build_ctcdecoder
import numpy


textprocess = TextProcess()

labels = [
    "'",  
    " ",  
    "a",  
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",  # 28
    "_",  # 29, blank
]

# class CTCBeamDecoder:

#     def __init__(self, beam_size=100, blank_id=labels.index('_'), kenlm_path=None):
#         print("loading beam search with Language model...")
#         self.decoder = ctcdecode.CTCBeamDecoder(labels=labels,model_path=kenlm_path,
#                                                 alpha=0.522729216841, beta=0.96506699808,
#                                                 beam_width=beam_size, blank_id=labels.index('_'))
     
#         print("finished loading beam search")

#     def __call__(self, output):
#         beam_results, beam_scores, timesteps, out_seq_len = self.decoder.decode(output)
#         print("decoding!!!")
#         return self.convert_to_string(beam_results[0][0],labels, out_seq_len[0][0])
#         # # output.squeeze(0)
#         # # output = output.transpose(0,1)
#         # text = self.decoder.decode(output)
#         # return text

#     def convert_to_string(self,tokens, vocab, seq_len):
#         return "".join([vocab[x] for  x in tokens[0:seq_len]])

class Pyctcdecoder():
    def __init__(self,labels = labels, kenlm_model_path = None):
        self.decoder = build_ctcdecoder(labels=labels, 
                                        kenlm_model_path=kenlm_model_path,
                                        alpha=0.5,  # tuned on a val set
                                        beta=1.0)
        
    def __call__(self,output):
        print("Decoding !!!")
        # output = output.squeeze()
        output = torch.argmax(input=output,dim=0)
        output = output.detach().numpy()
        # print(f"output shape in decoder after squeeze = {output.shape}")

        text = self.decoder.decode(output)
        return text
        

