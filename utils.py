import os
import logging
from rich.logging import RichHandler
import importlib


class Factory:
    def __init__(self):
        self.__classes = {}

    def register(self, cls, name):
        if name in self.__classes:
            raise Exception()

        self.__classes[name] = cls

    def get(self, name: str, *args, **kwargs):
        return self.__classes[name](*args, **kwargs)


def load_model(conf_name: str, step: int):
    conf_file = importlib.import_module(f"configs.{conf_name}")
    conf = conf_file.conf

    model_class = conf["model_class"]
    model = model_class(
        name=conf_name,
        step=step,
        arch=conf["arch"],
        backbone=conf["backbone"],
        generator_class=conf["generator_class"],
        generator_params=conf["generator_params"],
        preprocessor_class=conf["preprocessor_class"],
        preprocessor_params=conf["preprocessor_params"],
        train_config=conf["train_config"],
    )

    return model


def get_logger(filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    logger = logging.getLogger()
    datetime_format = "[%Y/%m/%d %H:%M:%S]"
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)-8s %(message)s", datetime_format
    )
    rich_formatter = logging.Formatter("%(message)s")

    rich_handler = RichHandler(log_time_format=datetime_format)
    rich_handler.setFormatter(rich_formatter)
    rich_handler.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(filename)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    logger.addHandler(rich_handler)
    logger.addHandler(file_handler)

    logger.debug("Logger initialized")
    logger.setLevel(logging.INFO)
    return logger
