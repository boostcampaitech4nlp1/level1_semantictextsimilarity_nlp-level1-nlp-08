import pandas as pd
import pytorch_lightning as pl
import torch

import train
from data_loader.data_loaders import Dataloader


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

    model = train.load_model(args, dataloader)  # train.py에 저장된 모델을 불러오는 메서드 따로 작성함

    model.eval()

    predictions = trainer.predict(model=model, datamodule=dataloader)

    predictions = list(round(float(i), 1) for i in torch.cat(predictions))
    predictions_n = [round(5 * x / (max(predictions) - min(predictions) + 1e-8), 1) for x in predictions]  # Normalize
    predictions_e = [round((x[0] + x[1]) / 2, 1) for x in zip(predictions, predictions_n)]  # Mean

    output = pd.read_csv("../data/sample_submission.csv")
    output_n = pd.read_csv("../data/sample_submission.csv")
    output_e = pd.read_csv("../data/sample_submission.csv")

    output["target"] = predictions
    output_n["target"] = predictions_n
    output_e["target"] = predictions_e

    output.to_csv("output.csv", index=False)
    output_n.to_csv("output_n.csv", index=False)
    output_e.to_csv("output_e.csv", index=False)
