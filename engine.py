import torch
from model import SpeechModel
from dataset import Data, Padding
from torch.utils.data import DataLoader

h_param = SpeechModel.hyper_parameters

Testmodel = SpeechModel(**h_param)
checkpoint_path = r"speech_logger\\Speech_loggs\\version_8\\checkpoints\\epoch=1-step=75.ckpt"

model_state_dict = torch.load(checkpoint_path)['state_dict']
Testmodel.load_state_dict(model_state_dict,strict=False)  
# donot use strict= False this may lead tp wrong predicitons
# solutions :- https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/13 
Testmodel.eval()




































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

    


