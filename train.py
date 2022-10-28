import re

import wandb
import torch
import random
import pytorch_lightning as pl


from data_loader.data_loaders import Dataloader, KfoldDataloader
from pytorch_lightning.loggers import WandbLogger
import model.model as module_arch
import utils.utils as utils

from pytorch_lightning.callbacks import ModelCheckpoint


def train(args):
    project_name = re.sub(
        "/",
        "_",
        f"{args.model_name}_epoch_{args.max_epoch}_batchsize_{args.batch_size}",
    )
    project_name = args.project_name + project_name
    dataloader = Dataloader(
        args.model_name,
        args.batch_size,
        args.train_ratio,
        args.shuffle,
        args.train_path,
        args.test_path,
        args.predict_path,
    )
    model = module_arch.Model(args.model_name, args.learning_rate, args.loss)
    wandb_logger = WandbLogger(project=project_name)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=args.max_epoch,
        log_every_n_steps=1,
        logger=wandb_logger,
        callbacks=[
            utils.early_stop(
                monitor=utils.monitor_config[args.monitor]["monitor"],
                patience=args.patience,
                mode=utils.monitor_config[args.monitor]["mode"],
            ),
            utils.best_save(
                save_path=args.save_path + f"{args.model_name}/",
                top_k=args.top_k,
                monitor=utils.monitor_config[args.monitor]["monitor"],
                mode=utils.monitor_config[args.monitor]["mode"],
            ),
        ],
    )

    trainer.fit(model=model, datamodule=dataloader)
    trainer.test(model=model, datamodule=dataloader)

    ## Todo. ckpt형식으로 저장할 수 있게끔 개선
    save_model = f"{args.save_path}{args.model_name}_epoch_{args.max_epoch}_batchsize_{args.batch_size}.pt"
    torch.save(model, save_model)


def k_train(args):
    random_count = random.randrange(1, 1000)
    random_count = str(random_count).zfill(4)

    project_name = re.sub(
        "/",
        "_",
        f"{args.model_name}_epoch_{args.max_epoch}_batchsize_{args.batch_size}",
    )
    project_name = args.project_name + "_" + project_name + "_" + random_count

    Kmodel = module_arch.Model(args.model_name, args.learning_rate, args.loss)

    results = []
    nums_folds = args.nums_folds
    random_count = random.randrange(1, 1000)
    random_count = str(random_count).zfill(4)

    for k in range(nums_folds):
        k_datamodule = KfoldDataloader(
            args.model_name, args.batch_size, args.shuffle, k=k, num_splits=nums_folds
        )
        k_datamodule.prepare_data()
        k_datamodule.setup()

        wandb_logger = WandbLogger(project=project_name)
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            max_epochs=args.max_epoch,
            log_every_n_steps=1,
            logger=wandb_logger,
            callbacks=[
                utils.early_stop(
                    monitor=utils.monitor_config[args.monitor]["monitor"],
                    patience=args.patience,
                    mode=utils.monitor_config[args.monitor]["mode"],
                ),
                utils.best_save(
                    save_path=args.save_path + f"{args.model_name}/fold{k}",
                    top_k=args.top_k,
                    monitor=utils.monitor_config[args.monitor]["monitor"],
                    mode=utils.monitor_config[args.monitor]["mode"],
                ),
            ],
        )

        trainer.fit(model=Kmodel, datamodule=k_datamodule)
        score = trainer.test(model=Kmodel, datamodule=k_datamodule)
        wandb.finish()

        results.extend(score)
        save_model = f"{args.save_path}{args.model_name}_fold_{k}_epoch_{args.max_epoch}_batchsize_{args.batch_size}.pt"
        torch.save(Kmodel, save_model)

    result = [x["test_pearson"] for x in results]
    score = sum(result) / nums_folds
    print(score)


def sweep(args, exp_count):  # 메인에서 받아온 args와 실험을 반복할 횟수를 받아옵니다
    project_name = re.sub(
        "/",
        "_",
        f"{args.model_name}_epoch_{args.max_epoch}_batchsize_{args.batch_size}",
    )
    project_name = args.project_name + project_name

    sweep_config = {
        "method": "bayes",  # random: 임의의 값의 parameter 세트를 선택, #bayes : 베이지안 최적화
        "parameters": {
            "lr": {
                # parameter를 설정하는 기준을 선택합니다. uniform은 연속적으로 균등한 값들을 선택합니다.
                "distribution": "uniform",
                "min": 1e-5,  # 최소값을 설정합니다.
                "max": 1e-4,  # 최대값을 설정합니다.
            },
            "batch_size": {
                "values": [
                    16,
                    32,
                    64,
                ]  # 배치 사이즈 조절, OOM 안나는 선에서 할 수 있도록 실험할 때 미리 세팅해주어야 함
            },
            "loss": {
                "values": [
                    "nll",
                    "l1",
                    "mse",
                ]  # loss 사용할 것들 지정, bce는 이진 분류가 아니기 때문에 일단 제외
            },
        },
        # 위의 링크에 있던 예시
        "early_terminate": {
            "type": "hyperband",
            "max_iter": 30,  # 프로그램에 대해 최대 반복 횟수 지정, min과 max는 같이 사용 불가능한듯
            "s": 2,
        },
    }

    # pearson 점수가 최대화가 되는 방향으로 학습을 진행합니다.
    sweep_config["metric"] = {"name": "val_pearson", "goal": "maximize"}

    def sweep_train(config=None):
        wandb.init(config=config)
        config = wandb.config

        dataloader = Dataloader(
            args.model_name,
            config.batch_size,
            args.train_ratio,
            args.shuffle,
            args.train_path,
            args.test_path,
            args.predict_path,
        )

        model = module_arch.Model(args.model_name, config.lr, args.loss)
        wandb_logger = WandbLogger(project=project_name)

        trainer = pl.Trainer(
            gpus=1, max_epochs=args.max_epoch, logger=wandb_logger, log_every_n_steps=1
        )
        trainer.fit(model=model, datamodule=dataloader)
        trainer.test(model=model, datamodule=dataloader)

    sweep_id = wandb.sweep(
        sweep=sweep_config,  # config 딕셔너리를 추가합니다.
        project=project_name,  # project의 이름을 추가합니다.
    )

    wandb.agent(sweep_id=sweep_id, function=sweep_train, count=exp_count)  # 실험할 횟수 지정
