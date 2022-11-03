import pandas as pd
import pytorch_lightning as pl
import torch
import os

import train
from data_loader.data_loaders import Dataloader


def inference(args, conf):
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
    trainer = pl.Trainer(gpus=1, max_epochs=conf.train.max_epoch)

    model, _, __ = train.load_model(
        args, conf, dataloader
    )  # train.py에 저장된 모델을 불러오는 메서드 따로 작성함

    model.eval()

    predictions = trainer.predict(
        model=model,
        datamodule=dataloader,
    )
    trainer.test(model=model, datamodule=dataloader)

    predictions = list(float(i) for i in torch.cat(predictions))
    # predictions_n = [
    #     round(5 * x / (max(predictions) - min(predictions) + 1e-8), 1)
    #     for x in predictions
    # ]  # Normalize
    # predictions_e = [
    #     round((x[0] + x[1]) / 2, 1) for x in zip(predictions, predictions_n)
    # ]  # Mean

    output = pd.read_csv("../data/sample_submission.csv")
    # output_n = pd.read_csv("../data/sample_submission.csv")
    # output_e = pd.read_csv("../data/sample_submission.csv")

    output["target"] = predictions
    # output_n["target"] = predictions_n
    # output_e["target"] = predictions_e

    output.to_csv("output.csv", index=False)
    # output_n.to_csv("output_n.csv", index=False)
    # output_e.to_csv("output_e.csv", index=False)


def kfold_inference(args, conf):
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
    ### 자치구역
    models_path = "save_models"
    ###

    trainer = pl.Trainer(gpus=1, max_epochs=conf.train.max_epoch)

    models = [model for (_, _, model) in os.walk(models_path)][0]
    predict = []
    # predict_n = []
    print("Models: ", len(models))
    for model in models:
        print("Predict proceeding: ", model)
        path_ = (
            "/opt/ml/level1_semantictextsimilarity_nlp-level1-nlp-08/save_models/"
            + model
        )
        if model.split(".")[-1] == "ckpt":
            model, _, __ = train.load_model(args, conf, dataloader)
            model = model.load_from_checkpoint(path_)
        elif model.split(".")[-1] == "pt":
            model = torch.load(path_)
        else:
            exit("saved_model 파일 오류")

        model.eval()

        predictions = trainer.predict(model=model, datamodule=dataloader)

        predictions = list(float(i) for i in torch.cat(predictions))
        # predictions_n = [
        #     (5 * x / (max(predictions) - min(predictions))) for x in predictions
        # ]  # Normalize
        predict.append(predictions)
        # predict_n.append(predictions_n)

    predict = [round(sum(x) / len(models), 1) for x in zip(*predict)]
    # predict_n = [round(sum(x) / len(models), 1) for x in zip(*predict_n)]

    output = pd.read_csv("../data/sample_submission.csv")
    # output_n = pd.read_csv("../data/sample_submission.csv")

    output["target"] = predict
    # output_n["target"] = predict_n

    output.to_csv("output.csv", index=False)
    # output_n.to_csv("output_n.csv", index=False)
