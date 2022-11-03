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


def full_model_step(conf, model_name):
    SEED = conf.utils.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    if conf.model.model_name == "klue/roberta-large":
        dataloader, model = new_instance_KLUE(conf)
    elif conf.model.model_name == "xlm-roberta-large":
        dataloader, model = new_instance_XLM(conf)
    elif conf.model.model_name == "kykim/funnel-kor-base":
        dataloader, model = new_instance_FUNNEL(conf)
    else:
        dataloader, model = new_instance_XLM(conf)
    project_name = conf.wandb.project
    dataloader, model = create_instance.new_instance(conf)  # 함수화로 변경
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
    trainer.save_checkpoint("./result/" + "{}.ckpt")
    predictions = trainer.predict(
        model=model,
        datamodule=dataloader,
    )
    predictions = list(float(i) for i in torch.cat(predictions))  # 리스트화

    output = pd.read_csv("../data/sample_submission.csv")
    output["target"] = predictions
    output.to_csv(f"output_{model_name}.csv", index=False)
    wandb.finish()


if __name__ == "__main__":
    # 하이퍼 파라미터 등 각종 설정값을 입력받습니다
    # 터미널 실행 예시 : python3 run.py --batch_size=64 ...
    # 실행 시 '--batch_size=64' 같은 인자를 입력하지 않으면 default 값이 기본으로 실행됩니다
    config_name_list = ["funnel", "klue", "xlm"]
    for model_name in config_name_list:
        conf = OmegaConf.load(f"./config/{model_name}_ensemble.yaml")
        full_model_step(conf, model_name)

    # for i in config_name_list:
