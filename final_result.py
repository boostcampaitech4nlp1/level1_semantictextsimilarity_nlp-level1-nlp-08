import random

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import model.model as module_arch
import wandb
from pytorch_lightning.loggers import WandbLogger
import utils.utils as utils

from data_loader.data_loaders import Dataloader, KfoldDataloader
from omegaconf import OmegaConf
import re
import create_instance


# fix random seeds for reproducibility


def new_instance_KLUE(conf):  # sweep 부분 때문에 두번째 인자 추가
    dataloader = Dataloader(
        conf.model.model_name,
        conf.train.batch_size,
        conf.data.train_ratio,
        conf.data.shuffle,
        conf.path.train_path,
        conf.path.test_path,
        conf.path.predict_path,
        conf.data.swap,
    )
    model = module_arch.Klue_CustomModel(
        conf.model.model_name,
        conf.train.learning_rate,
        conf.train.loss,
        dataloader.new_vocab_size(),
        conf.train.use_frozen,
    )  # 새롭게 추가한 토큰 사이즈 반영

    return dataloader, model


def new_instance_XLM(conf):  # sweep 부분 때문에 두번째 인자 추가
    dataloader = Dataloader(
        conf.model.model_name,
        conf.train.batch_size,
        conf.data.train_ratio,
        conf.data.shuffle,
        conf.path.train_path,
        conf.path.test_path,
        conf.path.predict_path,
        conf.data.swap,
    )
    model = module_arch.Xlm_CustomModel(
        conf.model.model_name,
        conf.train.learning_rate,
        conf.train.loss,
        dataloader.new_vocab_size(),
        conf.train.use_frozen,
    )  # 새롭게 추가한 토큰 사이즈 반영

    return dataloader, model


def new_instance_FUNNEL(conf):  # sweep 부분 때문에 두번째 인자 추가
    dataloader = Dataloader(
        conf.model.model_name,
        conf.train.batch_size,
        conf.data.train_ratio,
        conf.data.shuffle,
        conf.path.train_path,
        conf.path.test_path,
        conf.path.predict_path,
        conf.data.swap,
        True,
    )
    model = module_arch.Funnel_CustomModel(
        conf.model.model_name,
        conf.train.learning_rate,
        conf.train.loss,
        dataloader.new_vocab_size(),
        conf.train.use_frozen,
    )  # 새롭게 추가한 토큰 사이즈 반영

    return dataloader, model


def full_model_step(conf, model_name, idx):
    SEED = conf.utils.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    if idx == 0:
        dataloader, model = new_instance_FUNNEL(conf)
    elif idx == 1:
        dataloader, model = new_instance_KLUE(conf)
    elif idx == 2:
        dataloader, model = new_instance_XLM(conf)
    else:
        dataloader, model = new_instance_XLM(conf)
    project_name = conf.wandb.project

    wandb_logger = WandbLogger(project=project_name)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=conf.train.max_epoch,
        log_every_n_steps=1,
        logger=wandb_logger,
    )
    trainer.fit(model=model, datamodule=dataloader)
    trainer.test(model=model, datamodule=dataloader)

    filename = re.sub("/", "_", conf.model.model_name)
    trainer.save_checkpoint("./result/" + f"{model_name}.ckpt")
    predictions = trainer.predict(
        model=model,
        datamodule=dataloader,
    )
    predictions = list(float(i) for i in torch.cat(predictions))  # 리스트화

    output = pd.read_csv("../data/sample_submission.csv")
    output["target"] = predictions
    output.to_csv(f"./result/output_{model_name}.csv", index=False)
    wandb.finish()


if __name__ == "__main__":

    config_name_list = ["funnel", "klue", "xlm"]
    for idx, model_name in enumerate(config_name_list):
        conf = OmegaConf.load(f"./config/{model_name}_ensemble.yaml")
        full_model_step(conf, model_name, idx)

    # for i in config_name_list:
