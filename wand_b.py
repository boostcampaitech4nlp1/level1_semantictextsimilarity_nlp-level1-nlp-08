# import

import os

import wandb
import argparse

import random
import numpy as np
import pandas as pd
import json

import requests

from tqdm.auto import tqdm

import transformers
import torch
import torchmetrics
import pytorch_lightning as pl

from pytorch_lightning.loggers import WandbLogger

from data_loader.data_loaders import Dataloader
import model.model as module_arch


# fix random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

if __name__ == '__main__':
    # 하이퍼 파라미터 등 각종 설정값을 입력받습니다
    # 터미널 실행 예시 : python3 run.py --batch_size=64 ...
    # 실행 시 '--batch_size=64' 같은 인자를 입력하지 않으면 default 값이 기본으로 실행됩니다
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='klue/roberta-small', type=str)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--max_epoch', default=3, type=int)
    parser.add_argument('--shuffle', default=True)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--train_path', default='../data/train.csv')
    parser.add_argument('--dev_path', default='../data/dev.csv')
    parser.add_argument('--test_path', default='../data/dev.csv')
    parser.add_argument('--predict_path', default='../data/test.csv')
    parser.add_argument('--wandb_name', default='nlp-08-sts')

    args = parser.parse_args(args=[])

# https://docs.wandb.ai/v/ko/sweeps/configuration
# Sweep을 통해 최적화된 hyperparameter를 찾을 수 있습니다.
# 찾기를 원하는 hyperparameter를 다음과 같이 sweep_config에 추가합니다.
# 본 미션에서는 learning rate를 searching하는 예시를 보입니다.

sweep_config = {
    'method': 'random',  # random: 임의의 값의 parameter 세트를 선택
    'parameters': {
        'lr': {
            # parameter를 설정하는 기준을 선택합니다. uniform은 연속적으로 균등한 값들을 선택합니다.
            'distribution': 'uniform',
            'min': 1e-5,                 # 최소값을 설정합니다.
            'max': 1e-4                  # 최대값을 설정합니다.
        }
    }
}

# pearson 점수가 최대화가 되는 방향으로 학습을 진행합니다.
sweep_config['metric'] = {'name': 'val_pearson', 'goal': 'maximize'}


def sweep_train(config=None):
    wandb.init(config=config)
    config = wandb.config

    dataloader = Dataloader(args.model_name, args.batch_size, args.shuffle,
                            args.train_path, args.dev_path, args.test_path, args.predict_path)
    model = module_arch.Model(args.model_name, config.lr)
    wandb_logger = WandbLogger(project=args.wandb_name)  # 로컬 쪽에서 로깅되는 이름

    trainer = pl.Trainer(gpus=1, max_epochs=args.max_epoch,
                         logger=wandb_logger, log_every_n_steps=1)
    trainer.fit(model=model, datamodule=dataloader)
    trainer.test(model=model, datamodule=dataloader)


sweep_id = wandb.sweep(
    sweep=sweep_config,     # config 딕셔너리를 추가합니다.
    project='test1'         # project의 이름을 추가합니다.
)
wandb.agent(
    sweep_id=sweep_id,      # sweep의 정보를 입력하고
    function=sweep_train,   # train이라는 모델을 학습하는 코드를
    count=5                 # 총 5회 실행해봅니다.
)
