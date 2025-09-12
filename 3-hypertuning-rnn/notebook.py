import marimo

__generated_with = "0.15.3"
app = marimo.App()


@app.cell
def _():
    from pathlib import Path
    import numpy as np
    import torch
    from typing import List
    from torch.nn.utils.rnn import pad_sequence
    from mltrainer import rnn_models, Trainer
    from torch import optim

    from mads_datasets import datatools
    import mltrainer
    mltrainer.__version__
    return Path, Trainer, optim, rnn_models, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # 1 Iterators
        We will be using an interesting dataset. [link](https://tev.fbk.eu/resources/smartwatch)

        From the site:
        > The SmartWatch Gestures Dataset has been collected to evaluate several gesture recognition algorithms for interacting with mobile applications using arm gestures. Eight different users performed twenty repetitions of twenty different gestures, for a total of 3200 sequences. Each sequence contains acceleration data from the 3-axis accelerometer of a first generation Sony SmartWatch™, as well as timestamps from the different clock sources available on an Android device. The smartwatch was worn on the user's right wrist. 

        """
    )
    return


@app.cell
def _():
    from mads_datasets import DatasetFactoryProvider, DatasetType
    from mltrainer.preprocessors import PaddedPreprocessor
    preprocessor = PaddedPreprocessor()

    gesturesdatasetfactory = DatasetFactoryProvider.create_factory(DatasetType.GESTURES)
    streamers = gesturesdatasetfactory.create_datastreamer(batchsize=32, preprocessor=preprocessor)
    train = streamers["train"]
    valid = streamers["valid"]
    return train, valid


@app.cell
def _(train, valid):
    len(train), len(valid)
    return


@app.cell
def _(train, valid):
    trainstreamer = train.stream()
    validstreamer = valid.stream()
    x, y = next(iter(trainstreamer))
    x.shape, y
    return trainstreamer, validstreamer, x, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Can you make sense of the shape?
        What does it mean that the shapes are sometimes (32, 27, 3), but a second time might look like (32, 30, 3)? In other words, the second (or first, if you insist on starting at 0) dimension changes. Why is that? How does the model handle this? Do you think this is already padded, or still has to be padded?


        # 2 Excercises
        Lets test a basemodel, and try to improve upon that.

        Fill the gestures.gin file with relevant settings for `input_size`, `hidden_size`, `num_layers` and `horizon` (which, in our case, will be the number of classes...)

        As a rule of thumbs: start lower than you expect to need!
        """
    )
    return


@app.cell
def _():
    from mltrainer import TrainerSettings, ReportTypes
    from mltrainer.metrics import Accuracy

    accuracy = Accuracy()
    return ReportTypes, TrainerSettings, accuracy


@app.cell
def _(rnn_models):
    model = rnn_models.BaseRNN(
        input_size=3,
        hidden_size=64,
        num_layers=1,
        horizon=20,
    )
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Test the model. What is the output shape you need? Remember, we are doing classification!
        """
    )
    return


@app.cell
def _(model, x):
    yhat = model(x)
    yhat.shape
    return (yhat,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Test the accuracy
        """
    )
    return


@app.cell
def _(accuracy, y, yhat):
    accuracy(y, yhat)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        What do you think of the accuracy? What would you expect from blind guessing?

        Check shape of `y` and `yhat`
        """
    )
    return


@app.cell
def _(y, yhat):
    yhat.shape, y.shape
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        And look at the output of yhat
        """
    )
    return


@app.cell
def _(yhat):
    yhat[0]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Does this make sense to you? If you are unclear, go back to the classification problem with the MNIST, where we had 10 classes.

        We have a classification problem, so we need Cross Entropy Loss.
        Remember, [this has a softmax built in](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) 
        """
    )
    return


@app.cell
def _(torch, y, yhat):
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(yhat, y)
    loss
    return (loss_fn,)


@app.cell
def _(torch):
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        print("Using MPS")
    elif torch.cuda.is_available():
        device = "cuda:0"
        print("using cuda")
    else:
        device = "cpu"
        print("using cpu")

    # on my mac, at least for the BaseRNN model, mps does not speed up training
    # probably because the overhead of copying the data to the GPU is too high
    # so i override the device to cpu
    device = "cpu"
    # however, it might speed up training for larger models, with more parameters
    return (device,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Set up the settings for the trainer and the different types of logging you want
        """
    )
    return


@app.cell
def _(Path, ReportTypes, TrainerSettings, accuracy, train, valid):
    settings = TrainerSettings(
        epochs=3, # increase this to about 100 for training
        metrics=[accuracy],
        logdir=Path("gestures"),
        train_steps=len(train),
        valid_steps=len(valid),
        reporttypes=[ReportTypes.TOML, ReportTypes.TENSORBOARD, ReportTypes.MLFLOW],
        scheduler_kwargs={"factor": 0.5, "patience": 5},
        earlystop_kwargs = {
            "save": False, # save every best model, and restore the best one
            "verbose": True,
            "patience": 5, # number of epochs with no improvement after which training will be stopped
            "delta": 0.0, # minimum change to be considered an improvement
        }
    )
    settings
    return (settings,)


@app.cell
def _():
    import torch.nn as nn
    from torch import Tensor
    from dataclasses import dataclass

    @dataclass
    class ModelConfig:
        input_size: int
        hidden_size: int
        num_layers: int
        output_size: int
        dropout: float = 0.0

    class GRUmodel(nn.Module):

        def __init__(self, config) -> None:
            super().__init__()
            self.config = _config
            self.rnn = nn.GRU(input_size=_config.input_size, hidden_size=_config.hidden_size, dropout=_config.dropout, batch_first=True, num_layers=_config.num_layers)
            self.linear = nn.Linear(_config.hidden_size, _config.output_size)

        def forward(self, x: Tensor) -> Tensor:
            x, _ = self.rnn(x)
            last_step = x[:, -1, :]
            yhat = self.linear(last_step)
            return yhat
    return GRUmodel, ModelConfig


@app.cell
def _(ModelConfig):
    _config = ModelConfig(input_size=3, hidden_size=64, num_layers=1, output_size=20, dropout=0.0)
    return


@app.cell
def _(
    GRUmodel,
    ModelConfig,
    Path,
    Trainer,
    device,
    loss_fn,
    optim,
    settings,
    torch,
    trainstreamer,
    validstreamer,
):
    import mlflow
    from datetime import datetime
    mlflow.set_tracking_uri('sqlite:///mlflow.db')
    mlflow.set_experiment('gestures')
    modeldir = Path('gestures').resolve()
    if not modeldir.exists():
        modeldir.mkdir(parents=True)
    with mlflow.start_run():
        mlflow.set_tag('model', 'modelname-here')
        mlflow.set_tag('dev', 'your-name-here')
        _config = ModelConfig(input_size=3, hidden_size=64, num_layers=1, output_size=20, dropout=0.1)
        model_1 = GRUmodel(config=_config)
        trainer = Trainer(model=model_1, settings=settings, loss_fn=loss_fn, optimizer=optim.Adam, traindataloader=trainstreamer, validdataloader=validstreamer, scheduler=optim.lr_scheduler.ReduceLROnPlateau, device=device)
        trainer.loop()
        if not settings.earlystop_kwargs['save']:
            tag = datetime.now().strftime('%Y%m%d-%H%M-')
            modelpath = modeldir / (tag + 'model.pt')
            torch.save(model_1, modelpath)
    return mlflow, trainer


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Try to update the code above by changing the hyperparameters.
    
        To discern between the changes, also modify the tag mlflow.set_tag("model", "new-tag-here") where you add
        a new tag of your choice. This way you can keep the models apart.
        """
    )
    return


@app.cell
def _(trainer):
    trainer.loop() # if you want to pick up training, loop will continue from the last epoch
    return


@app.cell
def _(mlflow):
    mlflow.end_run()
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
