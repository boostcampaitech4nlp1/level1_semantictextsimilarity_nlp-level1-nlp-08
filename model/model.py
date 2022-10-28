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
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name, num_labels=1
        )
        self.plm.resize_token_embeddings(32002)
        self.loss_func = loss_module.L1_loss

    def forward(self, x):
        x = self.plm(x)["logits"]

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log("train_loss", loss)
        self.log(
            "train_pearson",
            torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()),
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log("val_loss", loss)
        self.log(
            "val_pearson",
            torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()),
        )

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        self.log(
            "test_pearson",
            torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()),
        )

    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(x)

        return logits.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        return optimizer


class HeadClassifier(nn.Module):
    def __init__(self, hidden_dim=1024, drop_out_rate=0.2):
        super().__init__()
        self.dense = nn.Linear(768, hidden_dim)
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


class RoBERTa_Base_Model(pl.LightningModule):
    """
    model_name: 'klue/roberta-base'
    """

    def __init__(self, model_name, lr):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.lr = lr

        self.plm = transformers.AutoModel.from_pretrained(
            pretrained_model_name_or_path=model_name
        )
        self.classifier = HeadClassifier(1024, 0.2)
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
        self.log(
            "val_pearson",
            torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()),
        )

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        self.log(
            "test_pearson",
            torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()),
        )

    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(x)

        return logits.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        return optimizer
