import random
import numpy as np
import torch
M = 2**32 - 1

def init_fn(worker):
    seed = torch.LongTensor(1).random_().item()
    seed = (seed + worker) % M
    np.random.seed(seed)
    random.seed(seed)
