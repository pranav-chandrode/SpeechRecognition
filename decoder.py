from utils import TextProcess
from pyctcdecode  import build_ctcdecoder

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
    "z",  # 27
    "_",  # 28, blank
]

class CTCBeamDecoder:

    def __init__(self, beam_size=100, blank_id=labels.index('_'), kenlm_path=None):
        print("loading beam search with Language model...")
        self.decoder = build_ctcdecoder(labels=labels,kenlm_model_path=kenlm_path)
     
        print("finished loading beam search")

    # def __call__(self, output):
    def deconding(self,output):
        # output.squeeze(0)
        # output = output.transpose(0,1)
        text = self.decoder.decode(output)
        return text