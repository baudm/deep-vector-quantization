import itertools
from typing import List, Iterator

import torch
from torch.utils.data import Sampler, DistributedSampler


class BatchBalancedSampler(Sampler[List[int]]):

    def __init__(self, samplers: List[Sampler[int]], batch_ratios: List[float], batch_size: int, drop_last: bool) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.samplers = samplers
        self.batch_size = batch_size
        self.drop_last = drop_last
        # Make sure that the ratios add up to 1
        batch_ratios = torch.as_tensor(batch_ratios, dtype=torch.float)
        # Divide the batch size according to the given ratios
        props = (batch_size * batch_ratios / batch_ratios.sum()).round().int()
        # Make sure that the proportions add up to batch_size
        diff = batch_size - props.sum().item()
        props[props.argmin()] += diff
        self.batch_props = props
        num_samples = [s.total_size if isinstance(s, DistributedSampler) else len(s) for s in self.samplers]
        self.offsets = [0] + num_samples[:-1]
        self.offsets = list(itertools.accumulate(self.offsets))
        # Get the sampler with the most number of iterations. It will determine when the loop stops.
        self.ref_sampler_idx = (torch.as_tensor(num_samples) / props).argmax().item()

    def __iter__(self) -> Iterator[List[int]]:
        # Wrap the reference sampler in an iterator,
        # and non-reference samplers in a cycle() iterator
        samplers = []
        for i, s in enumerate(self.samplers):
            s = iter(s) if i == self.ref_sampler_idx else itertools.cycle(s)
            samplers.append(s)

        batch = []
        while True:
            try:
                for i, s in enumerate(samplers):
                    for _ in range(self.batch_props[i]):
                        batch.append(next(s) + self.offsets[i])
            except StopIteration:
                break
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        sampler = self.samplers[self.ref_sampler_idx]
        batch_prop = self.batch_props[self.ref_sampler_idx]
        if self.drop_last:
            return len(sampler) // batch_prop
        else:
            return (len(sampler) + batch_prop - 1) // batch_prop
