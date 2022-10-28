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


if __name__ == "__main__":
    # 하이퍼 파라미터 등 각종 설정값을 입력받습니다
    # 터미널 실행 예시 : python3 run.py --batch_size=64 ...
    # 실행 시 '--batch_size=64' 같은 인자를 입력하지 않으면 default 값이 기본으로 실행됩니다
    parser = argparse.ArgumentParser()

    parser.add_argument("--shuffle", default=True)
    parser.add_argument("--train_path", default="../data/train.csv")
    parser.add_argument("--test_path", default="../data/dev.csv")
    parser.add_argument("--predict_path", default="../data/test.csv")

    parser.add_argument("--monitor", default="val_loss", type=str)
    parser.add_argument("--patience", default=25, type=int)

    parser.add_argument("--top_k", default=3, type=int)
    parser.add_argument("--train_ratio", default=0.8, type=float)
    parser.add_argument("--loss", default="l1", type=str)
    parser.add_argument("--project_name", default="nlp-08-sts")
    parser.add_argument("--save_path", default="save_models/", type=str)
    parser.add_argument("--frozen", "-f", default=False)
    parser.add_argument("--nums_folds", default=1, type=int)
    # 주로 만지게 될 것 같은 인자들
    # 프로젝트 name은 모델명과 batch_size, epoch로 잘 지정하면 될 듯
    parser.add_argument("--learning_rate", "-l", default=1e-5, type=float)
    parser.add_argument("--batch_size", "-b", default=32, type=int)
    parser.add_argument("--max_epoch", "-e", default=100, type=int)
    parser.add_argument("--model_name", "-n", default="klue/roberta-small")
    parser.add_argument("--mode", "-m", required=True)
    parser.add_argument(
        "--saved_model",
        "-s",
        default="save_models/model.pt",
        help="저장된 모델의 파일 경로를 입력해주세요. 예시: save_models/klue/roberta-small/epoch=?-step=?.ckpt 또는 save_models/model.pt",
    )
    args = parser.parse_args()

    if args.mode == "train" or args.mode == "t":
        # num_folds 변수 확인
        if args.nums_folds > 1:
            train.k_train(args)
        else:
            train.train(args)

    elif args.mode == "exp" or args.mode == "e":
        exp_count = int(input("실험할 횟수를 입력해주세요 "))
        train.sweep(args, exp_count)

    elif args.mode == "inference" or args.mode == "i":
        inference.inference(args)
    else:
        print("모드를 다시 설정해주세요 ")
        print("train     : t,\ttrain")
        print("exp       : e,\texp")
        print("inference : i,\tinference")
