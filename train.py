import re

import wandb
import torch
import pytorch_lightning as pl

from data_loader.data_loaders import Dataloader, KfoldDataloader
from pytorch_lightning.loggers import WandbLogger
import model.model as module_arch
import utils.utils as utils

from pytorch_lightning.callbacks import ModelCheckpoint

# train.train(cfg)
def train(cfg):
    project_name = re.sub(
        "/",
        "_",
        f"{cfg.model.model_name}_epoch_{cfg.train.max_epoch}_batchsize_{cfg.train.batch_size}",
    )
    project_name = cfg.wandb.project + project_name

    dataloader = Dataloader(
        cfg.model.model_name,
        cfg.train.batch_size,
        cfg.data.train_ratio,
        cfg.data.shuffle,
        cfg.path.train_path,
        cfg.path.test_path,
        cfg.path.predict_path,
        cfg.data.swap,
    )
    model = module_arch.Model(
        cfg.model.model_name,
        cfg.train.learning_rate,
        cfg.train.loss,
        dataloader.new_vocab_size(),
        cfg.train.use_frozen,
    )  # 새롭게 추가한 토큰 사이즈 반영

    wandb_logger = WandbLogger(name=cfg.wandb.name, project=project_name)
    save_path = f"{cfg.path.save_path}{cfg.model.model_name}_maxEpoch{cfg.train.max_epoch}_batchSize{cfg.train.batch_size}/"
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=cfg.train.max_epoch,
        log_every_n_steps=1,
        logger=wandb_logger,
        callbacks=[
            utils.early_stop(
                monitor=utils.monitor_config[cfg.utils.monitor]["monitor"],
                patience=cfg.utils.patience,
                mode=utils.monitor_config[cfg.utils.monitor]["mode"],
            ),
            utils.best_save(
                save_path=save_path,
                top_k=cfg.utils.top_k,
                monitor=utils.monitor_config[cfg.utils.monitor]["monitor"],
                mode=utils.monitor_config[cfg.utils.monitor]["mode"],
                filename="{epoch}-{step}-{val_pearson}",  # best 모델 저장시에 filename 설정
            ),
        ],
    )

    trainer.fit(model=model, datamodule=dataloader)
    trainer.test(model=model, datamodule=dataloader)

    trainer.save_checkpoint(save_path + "model.ckpt")
    torch.save(model, save_path + "model.pt")


def k_train(cfg):
    project_name = re.sub(
        "/",
        "_",
        f"{cfg.model.model_name}_epoch_{cfg.train.max_epoch}_batchsize_{cfg.train.batch_size}",
    )
    project_name = cfg.wandb.project + project_name

    k_datamodule = KfoldDataloader(
        cfg.model.model_name,
        cfg.train.batch_size,
        cfg.data.shuffle,
        cfg.k_fold.num_folds,
        cfg.k_fold.k,
        cfg.path.train_path,
        cfg.path.test_path,
        cfg.path.predict_path,
        cfg.data.swap,
    )

    Kmodel = module_arch.Model(
        cfg.model.model_name,
        cfg.train.learning_rate,
        cfg.train.loss,
        k_datamodule.new_vocab_size(),
        cfg.train.use_frozen,
    )

    results = []
    num_folds = cfg.k_fold.num_folds

    for k in range(num_folds):
        k_datamodule.prepare_data()
        k_datamodule.setup()
        wandb_logger = WandbLogger(project=project_name, name=f"{k}th_fold")
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            max_epochs=cfg.train.max_epoch,
            log_every_n_steps=1,
            logger=wandb_logger,
            callbacks=[
                utils.early_stop(
                    monitor=utils.monitor_config[cfg.utils.monitor]["monitor"],
                    patience=cfg.utils.patience,
                    mode=utils.monitor_config[cfg.utils.monitor]["mode"],
                ),
                utils.best_save(
                    save_path=cfg.path.save_path + f"{cfg.model.model_name}/",
                    top_k=cfg.utils.top_k,
                    monitor=utils.monitor_config[cfg.utils.monitor]["monitor"],
                    mode=utils.monitor_config[cfg.utils.monitor]["mode"],
                    filename="{epoch}-{step}-{val_pearson}",  # best 모델 저장시에 filename 설정
                ),
            ],
        )

        trainer.fit(model=Kmodel, datamodule=k_datamodule)
        score = trainer.test(model=Kmodel, datamodule=k_datamodule)
        wandb.finish()
        results.extend(score)
        save_model = f"{cfg.path.save_path}{cfg.model.model_name}_fold_{k}_epoch_{cfg.train.max_epoch}_batchsize_{cfg.train.batch_size}.pt"
        torch.save(Kmodel, save_model)

    result = [x["test_pearson"] for x in results]
    score = sum(result) / num_folds
    print(score)


def sweep(cfg, exp_count):  # 메인에서 받아온 args와 실험을 반복할 횟수를 받아옵니다
    project_name = re.sub(
        "/",
        "_",
        f"{cfg.model.model_name}_epoch_{cfg.train.max_epoch}_batchsize_{cfg.train.batch_size}",
    )
    project_name = cfg.wandb.project + project_name

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
            cfg.model.model_name,
            cfg.train.batch_size,
            cfg.data.train_ratio,
            cfg.data.shuffle,
            cfg.path.train_path,
            cfg.path.test_path,
            cfg.path.predict_path,
            cfg.data.swap,
        )
        model = module_arch.Model(
            cfg.model.model_name,
            config.lr,
            config.loss,
            dataloader.new_vocab_size(),
            cfg.train.use_frozen,
        )

        wandb_logger = WandbLogger(project=project_name)
        save_path = (
            f"{cfg.path.save_path}{cfg.model.model_name}_sweep_id_{wandb.run.name}/"
        )
        trainer = pl.Trainer(
            gpus=1,
            max_epochs=cfg.train.max_epoch,
            logger=wandb_logger,
            log_every_n_steps=1,
            callbacks=[
                utils.early_stop(
                    monitor=utils.monitor_config[cfg.utils.monitor]["monitor"],
                    patience=cfg.utils.patience,
                    mode=utils.monitor_config[cfg.utils.monitor]["mode"],
                ),
                utils.best_save(
                    save_path=save_path,
                    top_k=cfg.utils.top_k,
                    monitor=utils.monitor_config[cfg.utils.monitor]["monitor"],
                    mode=utils.monitor_config[cfg.utils.monitor]["mode"],
                    filename="{epoch}-{step}-{val_pearson}",  # best 모델 저장시에 filename 설정
                ),
            ],
        )
        trainer.fit(model=model, datamodule=dataloader)
        trainer.test(model=model, datamodule=dataloader)

    sweep_id = wandb.sweep(
        sweep=sweep_config,  # config 딕셔너리를 추가합니다.
        project=project_name,  # project의 이름을 추가합니다.
    )

    wandb.agent(sweep_id=sweep_id, function=sweep_train, count=exp_count)  # 실험할 횟수 지정
