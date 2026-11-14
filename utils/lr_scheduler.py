import numpy as np
import torch

class LR_Scheduler(object):
    def __init__(self, base_lr, num_epochs, mode='poly'):
        self.mode = mode
        self.lr = base_lr
        self.num_epochs = num_epochs
        if mode != 'poly':
            raise ValueError('Only the poly learning-rate schedule is supported')

    def __call__(self, optimizer, epoch):
        if self.mode == 'poly':
            # Keep optimizer metadata serializable by torch.load(weights_only=True).
            now_lr = float(round(self.lr * np.power(1 - np.float32(epoch)/np.float32(self.num_epochs), 0.9), 8))
        self._adjust_learning_rate(optimizer, now_lr)
        return now_lr

    def _adjust_learning_rate(self, optimizer, lr):
        optimizer.param_groups[0]['lr'] = lr

class MultiEpochsDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()
    def __len__(self):
        return len(self.batch_sampler.sampler)
    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)

class _RepeatSampler(object):
    def __init__(self, sampler):
        self.sampler = sampler
    def __iter__(self):
        while True:
            yield from iter(self.sampler)
