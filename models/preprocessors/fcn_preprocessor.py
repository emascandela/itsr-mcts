from typing import Dict, List, Callable

import numpy as np

# import tensorflow as tf
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from generators.generator import Generator
from generators.image import ImagePair

from .preprocessor import Preprocessor

from torchvision import transforms


class ITSRDataset(Dataset):
    def __init__(
        self,
        generator: Generator,
        n_samples: int,
        transform: Callable,
    ):
        self.generator = generator
        self.n_samples = n_samples
        self.transform = transform

    def __getitem__(self, idx):
        pair = self.generator.get_random_pair()
        source, target = pair.numpy()
        source = self.transform(source)
        target = self.transform(target)
        label = pair.target_image.applied_transformations[0]
        label = self.generator.transformations.index(label)

        return {"source_image": source, "target_image": target}, label

    def __len__(self):
        return self.n_samples


class FCNPreprocessor(Preprocessor):
    def get_dataset(self, generator: Generator, batch_size: int, n_samples: int):
        dataset = ITSRDataset(
            generator=generator,
            n_samples=n_samples,
            transform=self.transform,
        )
        dataloader = DataLoader(dataset, batch_size=batch_size)

        return dataloader
