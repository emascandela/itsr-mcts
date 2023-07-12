import os

import fire
import utils


def train(conf_name: str, load_weights: bool = False, step: int = 0, **kwargs):
    model = utils.load_model(conf_name=conf_name, step=step)

    if load_weights:
        print("Loading!")
        model.load_weights()

    model.train(**kwargs)


if __name__ == "__main__":
    fire.Fire(train)
