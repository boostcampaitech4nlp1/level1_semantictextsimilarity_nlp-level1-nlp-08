import argparse
import random
import numpy as np
import pandas as pd

import torch
import pytorch_lightning as pl

from data_loader.data_loaders import Dataloader

import inference
import train

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
    parser.add_argument('-b','--batch_size', default=8, type=int)
    parser.add_argument('-e','--max_epoch', default=5, type=int)
    parser.add_argument('--shuffle', default=True)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--train_path', default='../data/train.csv')
    parser.add_argument('--dev_path', default='../data/dev.csv')
    parser.add_argument('--test_path', default='../data/dev.csv')
    parser.add_argument('--predict_path', default='../data/test.csv')
    parser.add_argument('--saved_model', default='/model/model.pt')
    parser.add_argument('-n','--project_name', default='nlp-08-sts')
    parser.add_argument('-m', '--mode', required=True)
    args = parser.parse_args()

    assert (args.mode != 'train' or args.mode != 'inference' or args.mode !=
            'exp' or args.mode != 'full'), "모드를 다시 설정해주세요"

    if args.mode == 'train' or args.mode == 't':
        train.train(args)
    elif args.mode == 'exp' or args.mode == 'e':
        exp_count = int(input("실험할 횟수를 입력해주세요"))
        train.sweep(args,exp_count)
        
    elif args.mode == 'inference' or args.mode == 'i':
        inference.inference(args)
    elif args.mode == 'full' or args.mode == 'f':
        train.train(args)
        inference.inference(args)
    else:
        print("모드를 다시 설정해주세요")
        print("train     : t,\ttrain")
        print("exp       : e,\texp")
        print("inference : i,\tinference")
        print("full      : f,\tfull")
        
        