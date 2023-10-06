import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
import torch.nn.functional as F
# from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.core.module import LightningModule
from model import SpeechModel
from dataset import Data, Padding
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from argparse import ArgumentParser
import sys
sys.setrecursionlimit(10000)

hyper_parameters = {
    'input_channel' : 128,
    'num_classes' : 2,
    'num_layers' : 2,
    'hidden_size' : 1024,
    'batch_size' : 64,
    'epochs' : 2,
    'learning_rate' : 0.0001
}

class Speech(LightningModule):
    def __init__(self,model,args):
        super(Speech,self).__init__()
        self.model = model
        self.criterion = nn.CTCLoss(blank= 28,zero_infinity= True)
        self.args = args

    def forward(self,x,hidden):
        return self.model(x,hidden)
    
    def configure_optimizers(self):
        self.optimizer = Adam(self.model.parameters() ,lr= hyper_parameters['learning_rate'])
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer,mode = 'min',patience=6,factor=0.5)

        # return [self.optimizer], [self.scheduler]
        return {'optimizer' : self.optimizer, 
                'scheduler' : self.scheduler,
                'monitor': 'val_loss'
                }
    
    def train_dataloader(self):
        d_params = Data.data_hparams

        train_dataset = Data(self.args.train_file, **d_params)
        return DataLoader(dataset= train_dataset,batch_size= hyper_parameters['batch_size'],
                          shuffle=True,collate_fn=Padding,num_workers=4)
    
    def step1(self,batch):
        spectrograms , lables, spec_len , label_len = batch
        batch_size = spectrograms.shape[0]
        spec_len = torch.tensor(spec_len,dtype=torch.long)
        label_len = torch.tensor(label_len,dtype=torch.long)
        h0, c0 = self.model.hidden_initialize(batch_size)
        output, _ = self(spectrograms,(h0,c0))
        # output = output.unsqueeze(1)
        output = F.log_softmax(output,dim= 2)
        # print(f"spec_len = {spec_len}")
        # print(f"spec_len shape = {spec_len.shape}")
        # print(f"spec len = {spec_len}")
        loss = self.criterion(output,lables,spec_len,label_len)
        
        return loss
    
    def training_step(self,batch,batch_idx):
        loss = self.step1(batch)
        logs = {'loss': loss , 'lr' : self.optimizer.param_groups[0]['lr']}
        return {'loss': loss, 'logs' : logs}
        # return {'loss' :loss }

    def val_dataloader(self):
        d_params = Data.data_hparams

        val_dataset = Data(self.args.val_file, **d_params)
        return DataLoader(dataset= val_dataset,batch_size= hyper_parameters['batch_size'],
                          shuffle=False,collate_fn=Padding,num_workers=4)

    def validation_step(self,batch,batch_idx):
        loss = self.step1(batch)
        return {"val_loss" : loss}
        
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        # self.scheduler.step(avg_loss)
        # self.scheduler.step(metrics=avg_loss,epoch=1)
        tensorboard_logs = {'val_loss' : avg_loss}
        return {'val_loss' : avg_loss, 'logs' : tensorboard_logs}
    


def checkpoint_callback(args):
    return ModelCheckpoint(
        # dirpath=args.save_model_path,
        dirpath=r'speech_logger\\Speech_loggs\\version_8\\checkpoints',
        save_top_k=True,
        verbose=True,
        monitor='val_loss',
        mode='min',
    )

def main(args):
    h_params = SpeechModel.hyper_parameters
    model = SpeechModel(**h_params)
    
    logger = TensorBoardLogger(save_dir=args.logger_dir, name= "Speech_loggs")

    speech_module = Speech(model,args)
    trainer = Trainer(max_epochs = 2,log_every_n_steps=1,gpus = 0,logger = logger,checkpoint_callback = checkpoint_callback(args),fast_dev_run = False)
    
    
    # d_params = Data.data_hparams
    # val_dataset = Data('save_json/test.json',**d_params)
    # val_dataloader = DataLoader(dataset= val_dataset,batch_size= hyper_parameters['batch_size'],
    #                       shuffle=False,collate_fn=Padding,num_workers=4)
    # trainer = Trainer()
    # trainer.validate(speech_module,dataloaders=val_dataloader)
    
    # trainer = Trainer(max_epochs = hyper_parameters['epochs'],gpus = 0,logger = logger,checkpoint_callback = checkpoint_callback(args),fast_dev_run = False)
    
    
    trainer.fit(speech_module)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--train_file', default= None, required= True,type= str, help="Enter the training json file path")
    parser.add_argument('--val_file', default= None, required= True,type= str, help="Enter the validation json file path")
    parser.add_argument('--logger_dir',default= None, required= True,type= str,help="Logging directory for tensorboard logger")
    parser.add_argument('--save_model_path',default= None, required= True,type= str,help="Save directory for model")
    args = parser.parse_args()

    main(args)

# python trainer.py --train_file save_jsos/train.py --val_file save_json/test.py --logger_dir speech_logger --save_model_path saved_model