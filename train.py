import re

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger

import model.model as module_arch
import utils.utils as utils
import wandb
from data_loader.data_loaders import Dataloader, KfoldDataloader

# train.train(conf)
def train(conf):
    project_name = re.sub(
        "/",
        "_",
        f"{conf.model.model_name}_epoch_{conf.train.max_epoch}_batchsize_{conf.train.batch_size}",
    )
    project_name = conf.wandb.project + project_name

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
    model = module_arch.Model(
        conf.model.model_name,
        conf.train.learning_rate,
        conf.train.loss,
        dataloader.new_vocab_size(),
        conf.train.use_frozen,
    )  # 새롭게 추가한 토큰 사이즈 반영

    wandb_logger = WandbLogger(project=project_name)
    save_path = f"{conf.path.save_path}{conf.model.model_name}_maxEpoch{conf.train.max_epoch}_batchSize{conf.train.batch_size}_{wandb_logger.experiment.name}/"
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=conf.train.max_epoch,
        log_every_n_steps=1,
        logger=wandb_logger,
        callbacks=[
            utils.early_stop(
                monitor=utils.monitor_config[conf.utils.monitor]["monitor"],
                patience=conf.utils.patience,
                mode=utils.monitor_config[conf.utils.monitor]["mode"],
            ),
            utils.best_save(
                save_path=save_path,
                top_k=conf.utils.top_k,
                monitor=utils.monitor_config[conf.utils.monitor]["monitor"],
                mode=utils.monitor_config[conf.utils.monitor]["mode"],
                filename="{epoch}-{step}-{val_pearson}",  # best 모델 저장시에 filename 설정
            ),
        ],
    )

    trainer.fit(model=model, datamodule=dataloader)
    trainer.test(model=model, datamodule=dataloader)

    trainer.save_checkpoint(save_path + "model.ckpt")
    torch.save(model, save_path + "model.pt")


def continue_train(args, conf):
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
    model, args, conf = load_model(
        args, conf, dataloader
    )  # train.py에 저장된 모델을 불러오는 메서드 따로 작성함

    wandb_logger = WandbLogger(project=conf.wandb.project)
    save_path = f"{conf.path.save_path}{conf.model.model_name}_maxEpoch{conf.train.max_epoch}_batchSize{conf.train.batch_size}_{wandb_logger.experiment.name}/"  # 모델 저장 디렉터리명에 wandb run name 추가
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=conf.train.max_epoch,
        log_every_n_steps=1,
        logger=wandb_logger,
        callbacks=[
            utils.early_stop(
                monitor=utils.monitor_config[conf.utils.monitor]["monitor"],
                patience=conf.utils.patience,
                mode=utils.monitor_config[conf.utils.monitor]["mode"],
            ),
            utils.best_save(
                save_path=save_path,
                top_k=conf.utils.top_k,
                monitor=utils.monitor_config[conf.utils.monitor]["monitor"],
                mode=utils.monitor_config[conf.utils.monitor]["mode"],
                filename="{epoch}-{step}-{val_pearson}",  # best 모델 저장시에 filename 설정
            ),
        ],
    )

    trainer.fit(model=model, datamodule=dataloader)
    trainer.test(model=model, datamodule=dataloader)

    trainer.save_checkpoint(save_path + "model.ckpt")
    torch.save(model, save_path + "model.pt")


def k_train(conf):
    project_name = f"_epoch_{conf.train.max_epoch}_batchsize_{conf.train.batch_size}"
    project_name = conf.wandb.project + project_name

    results = []
    num_folds = conf.k_fold.num_folds

    for k in range(num_folds):
        k_datamodule = KfoldDataloader(
            conf.model.model_name,
            conf.train.batch_size,
            conf.data.shuffle,
            k,
            conf.k_fold.num_split,
            conf.path.train_path,
            conf.path.test_path,
            conf.path.predict_path,
            conf.data.swap,
        )

        Kmodel = module_arch.Model(
            conf.model.model_name,
            conf.train.learning_rate,
            conf.train.loss,
            k_datamodule.new_vocab_size(),
            conf.train.use_frozen,
        )

        name_ = f"{k+1}th_fold"
        wandb_logger = WandbLogger(project=project_name, name=name_)
        save_path = f"{conf.path.save_path}{conf.model.model_name}_{conf.train.max_epoch}_{conf.train.batch_size}_{name_}/"  # 모델 저장 디렉터리명에 wandb run name 추가
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            max_epochs=conf.train.max_epoch,
            log_every_n_steps=1,
            logger=wandb_logger,
            callbacks=[
                utils.early_stop(
                    monitor=utils.monitor_config[conf.utils.monitor]["monitor"],
                    patience=conf.utils.patience,
                    mode=utils.monitor_config[conf.utils.monitor]["mode"],
                ),
                utils.best_save(
                    save_path=save_path,
                    top_k=conf.utils.top_k,
                    monitor=utils.monitor_config[conf.utils.monitor]["monitor"],
                    mode=utils.monitor_config[conf.utils.monitor]["mode"],
                    filename=f"{k}_best_pearson_model",
                ),
            ],
        )

        trainer.fit(model=Kmodel, datamodule=k_datamodule)
        score = trainer.test(model=Kmodel, datamodule=k_datamodule)
        wandb.finish()

        results.extend(score)
        save_model = f"{conf.path.save_path}{conf.model.model_name}_fold_{k+1}_epoch_{conf.train.max_epoch}_batchsize_{conf.train.batch_size}"
        # torch.save(Kmodel, save_model + ".pt")
        trainer.save_checkpoint(save_model + ".ckpt")

    result = [x["test_pearson"] for x in results]
    score = sum(result) / num_folds
    print(score)


def sweep(conf, exp_count):  # 메인에서 받아온 args와 실험을 반복할 횟수를 받아옵니다
    project_name = re.sub(
        "/",
        "_",
        f"{conf.model.model_name}_epoch_{conf.train.max_epoch}_batchsize_{conf.train.batch_size}",
    )
    project_name = conf.wandb.project + project_name

    sweep_config = {
        "method": "bayes",  # random: 임의의 값의 parameter 세트를 선택, #bayes : 베이지안 최적화
        "parameters": {
            "lr": {
                # parameter를 설정하는 기준을 선택합니다. uniform은 연속적으로 균등한 값들을 선택합니다.
                "distribution": "uniform",
                "min": 1e-6,  # 최소값을 설정합니다.
                "max": 9e-6,  # 최대값을 설정합니다.
            },
            "batch_size": {
                "values": [
                    64,
                ]  # 배치 사이즈 조절, OOM 안나는 선에서 할 수 있도록 실험할 때 미리 세팅해주어야 함
            },
            "loss": {
                "values": [
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
    sweep_config["metric"] = {"name": "test_pearson", "goal": "maximize"}

    def sweep_train(config=None):
        wandb.init(config=config)
        config = wandb.config

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
        model = module_arch.Model(
            conf.model.model_name,
            config.lr,
            config.loss,
            dataloader.new_vocab_size(),
            conf.train.use_frozen,
        )

        wandb_logger = WandbLogger(project=project_name)
        save_path = (
            f"{conf.path.save_path}{conf.model.model_name}_sweep_id_{wandb.run.name}/"
        )
        trainer = pl.Trainer(
            gpus=1,
            max_epochs=conf.train.max_epoch,
            logger=wandb_logger,
            log_every_n_steps=1,
            callbacks=[
                utils.early_stop(
                    monitor=utils.monitor_config[conf.utils.monitor]["monitor"],
                    patience=conf.utils.patience,
                    mode=utils.monitor_config[conf.utils.monitor]["mode"],
                ),
                utils.best_save(
                    save_path=save_path,
                    top_k=conf.utils.top_k,
                    monitor=utils.monitor_config[conf.utils.monitor]["monitor"],
                    mode=utils.monitor_config[conf.utils.monitor]["mode"],
                    filename="{epoch}-{step}-{val_pearson}",  # best 모델 저장시에 filename 설정
                ),
            ],
        )
        trainer.fit(model=model, datamodule=dataloader)
        trainer.test(model=model, datamodule=dataloader)

        trainer.save_checkpoint(save_path + "klue-roberta.ckpt")
        torch.save(model, save_path + "klue-roberta.pt")

    sweep_id = wandb.sweep(
        sweep=sweep_config,  # config 딕셔너리를 추가합니다.
        project=project_name,  # project의 이름을 추가합니다.
    )

    wandb.agent(sweep_id=sweep_id, function=sweep_train, count=exp_count)  # 실험할 횟수 지정


def load_model(
    args, conf, dataloader: Dataloader
):  # continue_train과 inference시에 모델을 불러오는 기능은 같기 때문에 메서드로 구현함
    # 불러온 모델이 저장되어 있는 디렉터리를 parsing함
    # ex) 'save_models/klue/roberta-small_maxEpoch1_batchSize32_blooming-wind-57'
    save_path = "/".join(args.saved_model.split("/")[:-1])

    # huggingface에 저장된 모델명을 parsing함
    # ex) 'klue/roberta-small'
    model_name = "/".join(args.saved_model.split("/")[1:-1]).split("_")[0]

    if args.saved_model.split(".")[-1] == "ckpt":
        model = module_arch.Model(
            conf.model.model_name,
            conf.train.learning_rate,
            conf.train.loss,
            dataloader.new_vocab_size(),
            conf.train.use_frozen,
        )  # 새롭게 추가한 토큰 사이즈 반영
        model = model.load_from_checkpoint(args.saved_model)

    elif (
        args.saved_model.split(".")[-1] == "pt"
        and args.mode != "continue train"
        and args.mode != "ct"
    ):
        model = torch.load(args.saved_model)

    else:
        exit("saved_model 파일 오류")

    conf.path.save_path = save_path + "/"
    conf.model.model_name = "/".join(model_name.split("/")[1:])
    return model, args, conf
