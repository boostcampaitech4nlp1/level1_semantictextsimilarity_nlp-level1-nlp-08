import pandas as pd
import pytorch_lightning as pl
import torch
import create_instance


def inference(args, conf):

    trainer = pl.Trainer(gpus=1, max_epochs=conf.train.max_epoch, log_every_n_steps=1)

    dataloader, model = create_instance.new_instance(conf)  # 모듈화하여 진행
    model, _, __ = create_instance.load_model(args, conf, dataloader, model)

    model.eval()

    predictions = trainer.predict(
        model=model,
        datamodule=dataloader,
    )
    trainer.test(model=model, datamodule=dataloader)

    predictions = list(float(i) for i in torch.cat(predictions))  # 리스트화

    output = pd.read_csv("../data/sample_submission.csv")
    output["target"] = predictions
    output.to_csv("output.csv", index=False)

    # predictions_b = list(round(float(i), 1) for i in predictions)  # base postprocess
    # predictions_n = [round(5 * x / (max(predictions) - min(predictions) + 1e-8), 1) for x in predictions]  # Normalize

    # output_b = pd.read_csv("../data/sample_submission.csv")
    # output_n = pd.read_csv("../data/sample_submission.csv")

    # output_b["target"] = predictions_b
    # output_n["target"] = predictions_n

    # output_b.to_csv("output_b.csv", index=False)
    # output_n.to_csv("output_n.csv", index=False)
