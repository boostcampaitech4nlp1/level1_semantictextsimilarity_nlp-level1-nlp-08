import argparse
import random
import numpy as np
import pandas as pd

import torch
import pytorch_lightning as pl
from data_loader.data_loaders import Dataloader
import model.model as module_arch
from utils import utils


# fix random seeds for reproducibility
# SEED = 42
# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# torch.cuda.manual_seed_all(SEED)
# torch.backends.cudnn.benchmark = False
# torch.use_deterministic_algorithms(True)

if __name__ == '__main__':
    # 하이퍼 파라미터 등 각종 설정값을 입력받습니다
    # 터미널 실행 예시 : python3 run.py --batch_size=64 ...
    # 실행 시 '--batch_size=64' 같은 인자를 입력하지 않으면 default 값이 기본으로 실행됩니다
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='monologg/koelectra-base-finetuned-sentiment', type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--max_epoch', default=40, type=int)
    parser.add_argument('--shuffle', default=True)
    parser.add_argument('--bce', default=False)
    parser.add_argument('--train_ratio', default=0.8)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--train_path', default='../data/train.csv')
    parser.add_argument('--test_path', default='../data/dev.csv')
    parser.add_argument('--predict_path', default='../data/test.csv')
    parser.add_argument('--saved_model', default='model.pt')
    args = parser.parse_args(args=[])

    dataloader = Dataloader(args.model_name, args.batch_size, args.train_ratio, args.shuffle, args.bce,
                            args.train_path, args.test_path, args.predict_path)

    trainer = pl.Trainer(gpus=1, max_epochs=args.max_epoch, log_every_n_steps=1, 
                         accelerator='gpu', devices=1)

    
    # Inference part
    # 저장된 모델로 예측을 진행합니다.
    model = module_arch.Model(args.model_name, args.learning_rate, args.bce)
    model.load_from_checkpoint(args.saved_model)
    print("###### 모델 불러오기 완료 ######")
    predictions = trainer.predict(model=model, datamodule=dataloader)

    predictions = list(round(float(i), 1) for i in torch.cat(predictions))

    # output 형식을 불러와서 예측된 결과로 바꿔주고, output.csv로 출력합니다.
    output = pd.read_csv('../data/test.csv')
    output['target'] = predictions
    output.to_csv('output_{args.model_name}.csv', index=False)
