import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F

class SpeechModel(nn.Module):
    hyper_parameters = {
        'num_classes' : 29,
        'hidden_size' : 1024,
        'num_layers' : 2,
        'input_channel' : 64,
        'dropout' : 0.1,
        'batch_size' : 64
    }
    def __init__(self,num_classes,hidden_size,num_layers,input_channel,dropout,batch_size): 
        super(SpeechModel,self).__init__()
        """the input size of MelSpectrogram is [B,64,y]
           therefore, input channnel  = 64 and we will be keeping output channle 
           also 64 """

        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cnn = nn.Conv1d(input_channel,input_channel,kernel_size=5,stride= 5, padding= 10//2)
        # self.cnn = nn.Conv1d(input_channel,input_channel,10,2, padding= 10//2)
        
        
        # self.dense = nn.Sequential(
        #                             nn.Linear(1202,256),  # 1202 is cacluated value after the CNN layer
        #                             nn.LayerNorm(256),
        #                             nn.GELU(),
        #                             nn.Dropout(dropout),
        #                             nn.Linear(256,128),
        #                             nn.LayerNorm(128),
        #                             nn.GELU(),
        #                             nn.Dropout(dropout),
        #                             nn.Linear(128,128),
        #                             nn.LayerNorm(128),
        #                             nn.GELU(),
        #                             nn.Dropout(dropout),
        #                             nn.Linear(64,64),
        #                             nn.LayerNorm(64),
        #                             nn.GELU(),
        #                             nn.Dropout(dropout)
        #                            )

        self.dense = nn.Sequential(
                                    nn.Linear(1202,256),
                                    nn.LayerNorm(256),
                                    nn.GELU(),
                                    nn.Dropout(dropout),
                                    nn.Linear(256,64),
                                    nn.LayerNorm(64),
                                    nn.GELU(),
                                    nn.Dropout(),
                                    nn.Linear(64,64),
                                    nn.LayerNorm(64),
                                    nn.GELU(),
                                    nn.Dropout()
                                   )

        self.lstm = nn.LSTM(input_size = 64, hidden_size = hidden_size,num_layers = num_layers,batch_first = True, bidirectional = True)
        self.layerNorm = nn.LayerNorm(2*hidden_size)
        self.dense2 = nn.Sequential(
                                    nn.Linear(2*hidden_size,64),
                                    nn.LayerNorm(64),
                                    nn.GELU(),
                                    nn.Dropout(dropout),
                                    nn.Linear(64,num_classes)
                                    )
        
    def hidden_initialize(self,batch_size):
        n ,hs = self.num_layers, self.hidden_size
        return (torch.zeros(n*2,batch_size,hs),
                torch.zeros(n*2,batch_size,hs))
    
    # def calculate_input_size(self,out,batch_size):
    #     out = out.view(batch_size,-1)
    #     in_fc = out.shape[1]
    #     return in_fc

    def forward(self,x,hidden):
        """Input size was [batch,1,64,6001] so we have squeeze this"""
        x = x.squeeze(1)              # [batch,64,6001]
        print(f"Initial shape = {x.shape}")      # [batch,64,1202]
        x = F.gelu(self.cnn(x))
        # print(f"shape after cnn = {x.shape}")      # [batch,64,1202]
        # x = x.view(x.size(0),-1)
        # x = torch.max(x, dim=2)[0]
        x = F.gelu(self.dense(x))    # [batch, 64,64] 
        # x = x.transpose(0,1) # time, batch, feature
        # x = x.unsqueeze(1)
        x, (hn , cn) = self.lstm(x,hidden)   # [batch,64,2048]
        # x = x.squeeze(1)
        x = F.gelu(self.layerNorm(x)) 
        x = F.gelu(self.dense2(x))          # [batch, 64,29]

        return x , (hn,cn)