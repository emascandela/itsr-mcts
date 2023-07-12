import numpy as np
import torch
from typing import List
from torch.utils.data import Dataset, DataLoader
from typing import Callable

from generators.generator import Generator

from .preprocessor import Preprocessor


class ITSRDataset(Dataset):
    def __init__(
        self,
        generator: Generator,
        data: List,
        transform: Callable,
        # repeat: int = 1,
    ):
        self.generator = generator
        self.data = [[*p.numpy(), policy, value] for p, policy, value in data]
        self.transform = transform
        # self.repeat = repeat

        # idx = 0
        # source, target, policy, value = self.data[idx]
        # # source, target = pair.numpy()
        # source = self.transform(source)
        # target = self.transform(target)
        # policy = torch.Tensor(policy).to(torch.float32)
        # value = torch.Tensor([value]).to(torch.float32)

        # self.source = source
        # self.target = target
        # self.policy = policy
        # self.value = value

    def __getitem__(self, idx):
        # idx = idx % len(self.data)
        source, target, policy, value = self.data[idx]
        # source, target = pair.numpy()
        source = self.transform(source)
        target = self.transform(target)
        policy = torch.Tensor(policy)
        value = torch.Tensor([value])
        # print("s", source)
        # print("t", target)
        # print("p", policy)
        # print("v", value)

        return source, target, policy, value

        # return {"source_image": source, "target_image": target}, {
        #     "policy": policy,
        #     "value": value,
        # }

    def __len__(self):
        return len(self.data)


class MCTSPreprocessor(Preprocessor):
    def get_dataset(self, generator, data: List, batch_size: int):
        dataset = ITSRDataset(
            generator=generator,
            data=data,
            transform=self.transform,
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        return dataloader
