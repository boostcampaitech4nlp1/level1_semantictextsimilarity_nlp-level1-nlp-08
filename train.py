import argparse
import random
import numpy as np

import torch
import pytorch_lightning as pl

from data_loader.data_loaders import Dataloader
from pytorch_lightning.loggers import WandbLogger
import model.model as module_arch
import utils.utils as utils

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
    parser.add_argument('--max_epoch', default=100, type=int)
    parser.add_argument('--shuffle', default=True)
    parser.add_argument('--bce', default=False)
    parser.add_argument('--train_ratio', default=0.8)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--train_path', default='../data/train.csv')
    parser.add_argument('--test_path', default='../data/dev.csv')
    parser.add_argument('--predict_path', default='../data/test.csv')
    parser.add_argument('--saved_model', default='model.pt')
    parser.add_argument('--save_path', default='model/save_models/')
    args = parser.parse_args()

    # dataloader와 model을 생성합니다.
    wandb_logger = WandbLogger(project='koelectra-sts')
    dataloader = Dataloader(args.model_name, args.batch_size, args.train_ratio, args.shuffle, args.bce,
                            args.train_path, args.test_path, args.predict_path)
    model = module_arch.Model(args.model_name, args.learning_rate, args.bce)

    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=args.max_epoch, log_every_n_steps=1, 
                        callbacks=[utils.early_stop(monitor='val_loss', patience=10, mode='min'), utils.best_save(save_path=args.save_path, top_k=5, monitor='val_loss')],
                        logger=wandb_logger)

    # Train part
    trainer.fit(model=model, datamodule=dataloader)
    trainer.test(model=model, datamodule=dataloader)

    # # 학습이 완료된 모델을 저장합니다.
    torch.save(model, args.saved_model)
    
    
    
# K-Fold

# Kmodel = Model(args.model_name, args.learning_rate, args.bce)

# results = []
# nums_folds = 3

# for k in range(nums_folds):
#     datamodule = KfoldDataloader(args.model_name, args.batch_size, args.shuffle, args.bce, k=k, num_splits=nums_folds)
#     datamodule.prepare_data()
#     datamodule.setup()
    
#     checkpoint_callback = ModelCheckpoint(dirpath='model_save/', filename='{k}_fold_{epoch:02d}',
#                                           save_top_k=3, save_last=False, mode='max',
#                                           monitor='val_pearson')
#     earlystopping = EarlyStopping(monitor="val_pearson", patience=10, verbose=False)
    
#     trainer = pl.Trainer(gpus=1, max_epochs=args.max_epoch, logger=wandb_logger, 
#                          callbacks=[checkpoint_callback, earlystopping], accelerator='gpu', devices=1)
#     trainer.fit(model=Kmodel, datamodule=datamodule)
#     score = trainer.test(model=Kmodel, datamodule=datamodule)
    
#     results.extend(score)

# if args.bce:
#     result = [x['test_f1'] for x in results]
#     score = sum(result) / nums_folds
#     print('K-fold Test f1 score: ', score)
# else:
#     result = [x['test_pearson'] for x in results]
#     score = sum(result) / nums_folds
#     print('K-fold Test pearson score: ', score)