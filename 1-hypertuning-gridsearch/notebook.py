import marimo

__generated_with = "0.15.3"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Excercises 
    # 1. Tune the network
    Run the experiment below, explore the different parameters (see suggestions below) and study the result with tensorboard. 
    Make a single page (1 a4) report of your findings. Use your visualisation skills to communicate your most important findings.
    """
    )
    return


@app.cell
def _():
    from mads_datasets import DatasetFactoryProvider, DatasetType

    from mltrainer.preprocessors import BasePreprocessor
    from mltrainer import imagemodels, Trainer, TrainerSettings, ReportTypes, metrics

    import torch.optim as optim
    from torch import nn
    from tomlserializer import TOMLSerializer

    import marimo as mo
    return (
        BasePreprocessor,
        DatasetFactoryProvider,
        DatasetType,
        ReportTypes,
        TOMLSerializer,
        Trainer,
        TrainerSettings,
        metrics,
        mo,
        nn,
        optim,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We will be using `tomlserializer` to easily keep track of our experiments, and to easily save the different things we did during our experiments.
    It can export things like settings and models to a simple `toml` file, which can be easily shared, checked and modified.

    First, we need the data.
    """
    )
    return


@app.cell
def _(BasePreprocessor, DatasetFactoryProvider, DatasetType):
    fashionfactory = DatasetFactoryProvider.create_factory(DatasetType.FASHION)
    preprocessor = BasePreprocessor()
    streamers = fashionfactory.create_datastreamer(batchsize=64, preprocessor=preprocessor)
    train = streamers["train"]
    valid = streamers["valid"]
    trainstreamer = train.stream()
    validstreamer = valid.stream()
    return train, trainstreamer, valid, validstreamer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We need a way to determine how well our model is performing. We will use accuracy as a metric.""")
    return


@app.cell
def _(metrics):
    accuracy = metrics.Accuracy()
    return (accuracy,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    You can set up a single experiment.

    - We will show the model batches of 64 images, 
    - and for every epoch we will show the model 100 batches (trainsteps=100).
    - then, we will test how well the model is doing on unseen data (teststeps=100).
    - we will report our results during training to tensorboard, and report all configuration to a toml file.
    - we will log the results into a directory called "modellogs", but you could change this to whatever you want.
    """
    )
    return


@app.cell
def _(ReportTypes, TrainerSettings, accuracy):
    import torch
    loss_fn = torch.nn.CrossEntropyLoss()

    settings = TrainerSettings(
        epochs=3,
        metrics=[accuracy],
        logdir="modellogs",
        train_steps=100,
        valid_steps=100,
        reporttypes=[ReportTypes.TENSORBOARD, ReportTypes.TOML],
    )
    return loss_fn, settings, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We will use a very basic model: a model with three linear layers.""")
    return


@app.cell
def _(nn, torch):
    class NeuralNetwork(nn.Module):
        def __init__(self, num_classes: int, units1: int, units2: int) -> None:
            super().__init__()
            self.num_classes = num_classes
            self.units1 = units1
            self.units2 = units2
            self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(28 * 28, units1),
                nn.ReLU(),
                nn.Linear(units1, units2),
                nn.ReLU(),
                nn.Linear(units2, num_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits

    model = NeuralNetwork(
        num_classes=10, units1=256, units2=256)
    return NeuralNetwork, model


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    I developped the `tomlserializer` package, it is a useful tool to save configs, models and settings as a tomlfile; that way it is easy to track what you changed during your experiments.

    This package will 1. check if there is a `__dict__` attribute available, and if so, it will use that to extract the parameters that do not start with an underscore, like this:
    """
    )
    return


@app.cell
def _(model):
    {k: v for k, v in model.__dict__.items() if not k.startswith("_")}
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    This means that if you want to add more parameters to the `.toml` file, eg `units3`, you can add them to the class like this:

    ```python
    class NeuralNetwork(nn.Module):
        def __init__(self, num_classes: int, units1: int, units2: int, units3: int) -> None:
            super().__init__()
            self.num_classes = num_classes
            self.units1 = units1
            self.units2 = units2
            self.units3 = units3  # <-- add this line
    ```

    And then it will be added to the `.toml` file. Check the result for yourself by using the `.save()` method of the `TomlSerializer` class like this:
    """
    )
    return


@app.cell
def _(TOMLSerializer, model, settings):
    tomlserializer = TOMLSerializer()
    tomlserializer.save(settings, "settings.toml")
    tomlserializer.save(model, "model.toml")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Check the `settings.toml` and `model.toml` files to see what is in there.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""You can use the `Trainer` class from my `mltrainer` module to train your model. It has the TOMLserializer integrated, so it will automatically save the settings and model to a toml file if you have added `TOML` as a reporttype in the settings.""")
    return


@app.cell
def _(Trainer, loss_fn, model, optim, settings, trainstreamer, validstreamer):
    _trainer = Trainer(model=model, settings=settings, loss_fn=loss_fn, optimizer=optim.Adam, traindataloader=trainstreamer, validdataloader=validstreamer, scheduler=optim.lr_scheduler.ReduceLROnPlateau)
    _trainer.loop()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Now, check in the modellogs directory the results of your experiment.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We can now loop this with a naive approach, called a grid-search (why do you think i call it naive?).""")
    return


@app.cell
def _():
    _units = [256, 128, 64]
    for _unit1 in _units:
        for _unit2 in _units:
            print(f'Units: {_unit1}, {_unit2}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Of course, this might not be the best way to search for a model; some configurations will be better than others (can you predict up front what will be the best configuration?).

    So, feel free to improve upon the gridsearch by adding your own logic.
    """
    )
    return


@app.cell
def _(
    NeuralNetwork,
    ReportTypes,
    Trainer,
    TrainerSettings,
    accuracy,
    optim,
    torch,
    train,
    trainstreamer,
    valid,
    validstreamer,
):
    _units = [256, 128, 64]
    loss_fn_1 = torch.nn.CrossEntropyLoss()
    settings_1 = TrainerSettings(epochs=3, metrics=[accuracy], logdir='modellogs', train_steps=len(train), valid_steps=len(valid), reporttypes=[ReportTypes.TENSORBOARD, ReportTypes.TOML])
    for _unit1 in _units:
        for _unit2 in _units:
            model_1 = NeuralNetwork(num_classes=10, units1=_unit1, units2=_unit2)
            _trainer = Trainer(model=model_1, settings=settings_1, loss_fn=loss_fn_1, optimizer=optim.Adam, traindataloader=trainstreamer, validdataloader=validstreamer, scheduler=optim.lr_scheduler.ReduceLROnPlateau)
            _trainer.loop()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Because we have set the ReportType to TOML, you will find in every log dir a model.toml and settings.toml file.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Run the experiment, and study the result with tensorboard. 

    Locally, it is easy to do that with VS code itself. On the server, you have to take these steps:

    - in the terminal, `cd` to the location of the repository
    - activate the python environment for the shell. Note how the correct environment is being activated.
    - run `tensorboard --logdir=modellogs` in the terminal
    - tensorboard will launch at `localhost:6006` and vscode will notify you that the port is forwarded
    - you can either press the `launch` button in VScode or open your local browser at `localhost:6006`
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r""" """)
    return


if __name__ == "__main__":
    app.run()
