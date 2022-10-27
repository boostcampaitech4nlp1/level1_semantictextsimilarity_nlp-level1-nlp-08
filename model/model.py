import transformers
import torch
import torchmetrics
import pytorch_lightning as pl

from . import loss as loss_module


class Model(pl.LightningModule):
    def __init__(self, model_name, lr, bce):
        super().__init__()
        
        self.model_name = model_name
        self.lr = lr
        self.bce = bce
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name, num_labels=1)
        if self.bce:
            self.loss_func = loss_module.BCEWithLogitsLoss()
        else:
            self.loss_func = loss_module.L1_loss()
            
    def forward(self, x):
        x = self.plm(x)['logits']

        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log('train_loss', loss)
        if self.bce:
            self.log('train_f1', torchmetrics.functional.f1_score(logits, y))
        else:
            self.log('train_pearson', torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))
            # self.log('train_log_pearson', np.log(torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze())).cpu().numpy())
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log('val_loss', loss)
        if self.bce:
            self.log('val_f1', torchmetrics.functional.f1_score(logits, y))
        else:
            self.log('val_pearson', torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))
            # self.log('val_log_pearson', np.log(torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze())).cpu().numpy())
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        if self.bce:
            self.log('test_f1', torchmetrics.functional.f1_score(logits, y))
        else:
            self.log('test_pearson', torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))
            # self.log('test_log_pearson', np.log(torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze())).cpu().numpy())
            
    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(x)
        
        return logits.squeeze()
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        
        return optimizer