""" Converts the trained model in from checkpoint to traced model which will be used for testing"""

import torch
from model import SpeechModel
from collections import OrderedDict

def trace(model):
    model.eval()
    x = torch.rand(1,64,6001)
    hidden = model.hidden_initialize(1)     # 1 -> batch size when used during inference
    traced = torch.jit.trace(model,(x,hidden))
    return traced

def main():    
    h_param = SpeechModel.hyper_parameters
    Testmodel = SpeechModel(**h_param)

    checkpoint_path = r"speech_logger\\Speech_loggs\\version_8\\checkpoints\\epoch=1-step=75.ckpt"

    model_state_dict = torch.load(checkpoint_path)['state_dict']
    new_state_dict = OrderedDict()

    # removing the "model." from all the layers 
    for u,v in model_state_dict.items():
        name = u.replace("model.", "")
        new_state_dict[name] = v


    Testmodel.load_state_dict(new_state_dict)
    traced_model = trace(Testmodel)
    traced_model.save("saved_model\\traced_model.pt")
    print("Traced Model has been successfully saved!!!")


if __name__ == "__main__":
    main()