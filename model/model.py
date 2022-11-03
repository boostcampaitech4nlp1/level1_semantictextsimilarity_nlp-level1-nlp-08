import transformers
import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl

from . import loss as loss_module
from torch.optim.lr_scheduler import LambdaLR


class Model(pl.LightningModule):
    def __init__(
        self, model_name, lr, loss, new_vocab_size, frozen
    ):  # 새로운 vocab 사이즈 설정
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.lr = lr

        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name, num_labels=1
        )

        if frozen == "True":
            self.frozen()
        self.plm.resize_token_embeddings(new_vocab_size)  # 임베딩 차원 재조정
        self.loss_func = loss_module.loss_config[loss]

    def frozen(self):  # 추후 레이어를 반복하면서 얼리고 풀고 할 수 있게 훈련
        for name, param in self.plm.named_parameters():
            param.requires_grad = False
            if name in [
                "classifier.dense.weight",
                "classifier.dense.bias",
                "classifier.out_proj.weight",
                "classifier.out_proj.bias",
            ]:
                param.requires_grad = True

    def forward(self, x):
        x = self.plm(x)["logits"]

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
        scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lrsc_func)
        return [optimizer], [scheduler]


def lrsc_func(epoch):
    if epoch < 30:
        return 1
    elif epoch < 32:
        return 0.5
    elif epoch < 42:
        return 0.5 * 0.70 ** epoch(32 - epoch)
    else:
        return 0.02


class ExampleCustomModel(pl.LightningModule):
    def __init__(self, model_name, lr, loss, new_vocab_size, frozen):
        super().__init__()
        self.save_hyperparameters()
        self.model_name = model_name
        self.lr = lr
        self.classifier_input = 1024
        self.plm = (
            transformers.AutoModelForSequenceClassification.from_pretrained(  # 기존 모델
                pretrained_model_name_or_path=model_name,
                num_labels=self.classifier_input,
            )
        )
        self.plm.resize_token_embeddings(new_vocab_size)  # 임베딩 차원 재조정

        self.loss_func = loss_module.loss_config[loss]

        self.MLP_HEAD = nn.Sequential(
            nn.Dropout(0.2), nn.Tanh(), nn.Linear(self.classifier_input, 1)
        )

        if frozen == "True":
            self.frozen()
        self.plm.resize_token_embeddings(new_vocab_size)  # 임베딩 차원 재조정
        self.loss_func = loss_module.loss_config[loss]

    def frozen(self):  # 추후 레이어를 반복하면서 얼리고 풀고 할 수 있게 훈련
        for name, param in self.plm.named_parameters():
            param.requires_grad = False
            if name in [
                "classifier.dense.weight",
                "classifier.dense.bias",
                "classifier.out_proj.weight",
                "classifier.out_proj.bias",
            ]:
                param.requires_grad = True

    def forward(self, x):
        x = self.plm(x)["logits"]
        x = self.MLP_HEAD(x)
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
