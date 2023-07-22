import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F

class SpeechModel(nn.Module):
    hyper_parameters = {
        'num_classes' : 2,
        'hidden_size' : 1024,
        'num_layers' : 2,
        'input_channel' : 128,
        'dropout' : 0.1,
        'batch_size' : 32
    }
    def __init__(self,num_classes,hidden_size,num_layers,input_channel,dropout,batch_size): 
        super(SpeechModel,self).__init__()
        """the input size of MelSpectrogram is [B,128,y]
           therefore, input channnel  = 128 and we will be keeping output channle 
           also 128 """
        self.cnn = nn.Conv1d(input_channel,input_channel,kernel_size=2)
        self.dense = nn.Sequential(
                                    nn.Linear(128,128),
                                    nn.LayerNorm(128),
                                    nn.GELU(),
                                    nn.Dropout(dropout),
                                    nn.Linear(128,128),
                                    nn.LayerNorm(128),
                                    nn.GELU(),
                                    nn.Dropout(),
                                    nn.Linear(128,128),
                                    nn.LayerNorm(128),
                                    nn.GELU(),
                                    nn.Dropout()
                                   )

        self.lstm = nn.LSTM(input_size = 128, hidden_size = hidden_size,num_layers = num_layers,batch_first = True, bidirectional = True)
        self.layerNorm = nn.LayerNorm(hidden_size)
        self.dense2 = nn.Sequential(
                                    nn.Linear(hidden_size,64),
                                    nn.LayerNorm(64),
                                    nn.GELU(),
                                    nn.Dropout(dropout),
                                    nn.Linear(64,num_classes)
                                    )
        
    def hidden_initialize(self,batch_size):
        n ,hs = self.num_classes, self.hidden_size
        return (torch.zeros(n*2,batch_size,hs),
                torch.zeros(n*2,batch_size,hs))
    
    def forward(self,x,hidden):
        """Input size was [batch,1,128,some_no] so we have squeeze this"""
        x = x.squeeze()
        x = F.gelu(self.cnn(x))
        x = F.gelu(self.dense(x))    # batch, time, feature 
        x = x.transpose(0,1) # time, batch, feature
        out, hn , cn = self.lstm(x,hidden)
        x = F.gelu(self.layerNorm(x))
        x = F.gelu(self.dense2(x))

        return x , (hn,cn)