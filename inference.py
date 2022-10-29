import pandas as pd
import pytorch_lightning as pl
import torch

import model.model as module_arch
from data_loader.data_loaders import Dataloader, KfoldDataloader


def inference(args):
    dataloader = Dataloader(
        args.model_name,
        args.batch_size,
        args.train_ratio,
        args.shuffle,
        args.train_path,
        args.test_path,
        args.predict_path,
    )
    trainer = pl.Trainer(gpus=1, max_epochs=args.max_epoch, log_every_n_steps=1)

    if args.saved_model.split(".")[-1] == "ckpt":
        model_name = "/".join(args.saved_model.split("/")[1:3]).split("_")[
            0
        ]  # huggingface에 저장된 모델명을 parsing함
        model = module_arch.Model(
            model_name,
            args.learning_rate,
            args.loss,
            dataloader.new_vocab_size(),
            args.frozen,
        )  # 새롭게 추가한 토큰 사이즈 반영

        model = model.load_from_checkpoint(args.saved_model)
    elif args.saved_model.split(".")[-1] == "pt":
        model = torch.load(args.saved_model)
    else:
        exit("saved_model 파일 오류")

    model.eval()

    predictions = trainer.predict(model=model, datamodule=dataloader)

    predictions = list(round(float(i), 1) for i in torch.cat(predictions))
    predictions_n = [
        round(5 * x / (max(predictions) - min(predictions) + 1e-8), 1)
        for x in predictions
    ]  # Normalize
    predictions_e = [
        round((x[0] + x[1]) / 2, 1) for x in zip(predictions, predictions_n)
    ]  # Mean

    output = pd.read_csv("../data/sample_submission.csv")
    output_n = pd.read_csv("../data/sample_submission.csv")
    output_e = pd.read_csv("../data/sample_submission.csv")

    output["target"] = predictions
    output_n["target"] = predictions_n
    output_e["target"] = predictions_e

    output.to_csv("output.csv", index=False)
    output_n.to_csv("output_n.csv", index=False)
    output_e.to_csv("output_e.csv", index=False)


def kfold_inference(args, models_path, model_name):
    dataloader = Dataloader(
        args.model_name,
        args.batch_size,
        args.train_ratio,
        args.shuffle,
        args.train_path,
        args.test_path,
        args.predict_path,
    )
    trainer = pl.Trainer(gpus=1, max_epochs=args.max_epoch, log_every_n_steps=1)
    models = [model for (_, _, model) in os.walk(models_path)][0]
    predict, predict_n, predict_e = [], [], []

    for model in models:
        if model.split(".")[-1] == "ckpt":
            model_name = model_name
            model = module_arch.Model(
                model_name,
                args.learning_rate,
                args.loss,
                dataloader.new_vocab_size(),
                args.frozen,
            )  # 새롭게 추가한 토큰 사이즈 반영

            model = model.load_from_checkpoint(model)
        elif model.split(".")[-1] == "pt":
            model = torch.load(model)
        else:
            exit("saved_model 파일 오류")

        model.eval()

        predictions = trainer.predict(model=model, datamodule=dataloader)

        predictions = list(round(float(i), 1) for i in torch.cat(predictions))
        predictions_n = [
            round(5 * x / (max(predictions) - min(predictions) + 1e-8), 1)
            for x in predictions
        ]  # Normalize
        predictions_e = [
            round((x[0] + x[1]) / 2, 1) for x in zip(predictions, predictions_n)
        ]  # Mean
        predict.append(predictions)
        predict_n.append(predictions_n)
        predict_e.append(predictions_e)

    predict = [round(sum(x) / len(models), 1) for x in zip(*predict)]
    predict_n = [round(sum(x) / len(models), 1) for x in zip(*predict_n)]
    predict_e = [round(sum(x) / len(models), 1) for x in zip(*predict_e)]

    output = pd.read_csv("../data/sample_submission.csv")
    output_n = pd.read_csv("../data/sample_submission.csv")
    output_e = pd.read_csv("../data/sample_submission.csv")

    output["target"] = predict
    output_n["target"] = predict_n
    output_e["target"] = predict_e

    output.to_csv("output.csv", index=False)
    output_n.to_csv("output_n.csv", index=False)
    output_e.to_csv("output_e.csv", index=False)
