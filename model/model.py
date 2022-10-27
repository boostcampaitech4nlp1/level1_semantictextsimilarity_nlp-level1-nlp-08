import transformers
import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl

from . import loss as loss_module


class Model(pl.LightningModule):
    def __init__(self, model_name, lr):
        super().__init__()
        self.save_hyperparameters()
        
        self.model_name = model_name
        self.lr = lr
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name, num_labels=1)
        self.loss_func = loss_module.L1_loss
            
    def forward(self, x):
        x = self.plm(x)['logits']

        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log('train_loss', loss)
        self.log('train_pearson', torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log("val_loss", loss)
        self.log("val_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        self.log("test_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(x)
        
        return logits.squeeze()
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        
        return optimizer


class HeadClassifier(nn.Module):
    def __init__(self,input_dim, hidden_dim, drop_out_rate=0.2):
        super().__init__()
        self.dense = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(drop_out_rate)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, features):
        x = features[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.output(x)
        
        return x

class BaseModel(pl.LightningModule): # Base 모델로 이름 변경

    def __init__(self, model_name, lr):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.lr = lr
        
        self.plm = transformers.AutoModel.from_pretrained(pretrained_model_name_or_path=model_name)
        input_dim = transformers.AutoConfig.from_pretrained(pretrained_model_name_or_path=model_name).hidden_size # 히든벡터의 차원을 input으로 사용
        self.classifier = HeadClassifier(input_dim, 1024, 0.2)
        self.loss_func = loss_module.L1_loss

    def forward(self, x):
        x = self.plm(x)[0]
        x = self.classifier(x)
        
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log("val_loss", loss)
        self.log("val_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        self.log("test_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(x)

        return logits.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        
        return optimizer