from __future__ import annotations
import torch
from torch.utils.data import Sampler

class DeterministicEpochSampler(Sampler[int]):

    def __init__(self, data_source, *, base_seed: int, shuffle: bool=True) -> None:
        self.data_source = data_source
        self.base_seed = int(base_seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        count = len(self.data_source)
        if not self.shuffle:
            return iter(range(count))
        generator = torch.Generator(device='cpu')
        generator.manual_seed(self.base_seed + self.epoch * 1000003)
        return iter(torch.randperm(count, generator=generator).tolist())

    def __len__(self) -> int:
        return len(self.data_source)
