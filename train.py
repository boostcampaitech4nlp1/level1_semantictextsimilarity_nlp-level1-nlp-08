import argparse
import random
import numpy as np

import torch
import pytorch_lightning as pl

from data_loader.data_loaders import Dataloader
import model.model as module_arch
import utils.utils

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
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--max_epoch', default=1, type=int)
    parser.add_argument('--shuffle', default=True)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--train_path', default='../data/train.csv')
    parser.add_argument('--dev_path', default='../data/dev.csv')
    parser.add_argument('--test_path', default='../data/dev.csv')
    parser.add_argument('--predict_path', default='../data/test.csv')
    parser.add_argument('--save_model', default='model/model.pt')
    parser.add_argument('--save_path', default='model/save_models/')
    args = parser.parse_args()

    # dataloader와 model을 생성합니다.
    dataloader = Dataloader(args.model_name, args.batch_size, args.shuffle, args.train_path, args.dev_path,
                            args.test_path, args.predict_path)
    model = module_arch.Model(args.model_name, args.learning_rate)

    # gpu가 없으면 'gpus=0'을, gpu가 여러개면 'gpus=4'처럼 사용하실 gpu의 개수를 입력해주세요
    trainer = pl.Trainer(gpus=1, max_epochs=args.max_epoch, log_every_n_steps=1, 
                        callbacks=[utils.early_stop(monitor='val_loss', patience=5, mode='min'), utils.best_save(args.save_path)])  # save_path에 모델명을 포함해주면 불러올 때 모델명을 split을 하여 같은 모델 구조 만들어주기 편할듯

    # Train part
    trainer.fit(model=model, datamodule=dataloader)
    trainer.test(model=model, datamodule=dataloader)

    # # 학습이 완료된 모델을 저장합니다.
    # torch.save(model, args.save_model)