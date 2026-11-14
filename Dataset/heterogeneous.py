import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from .heterogeneous_sampling import SamplingConfig, sample_views
from .transforms import Compose, Identity, NumpyType, RandomIntensityChange

class HeterogeneousMIL(Dataset):
    def __init__(self, root, data_file, training=False, config=None,
                 transforms='', num_instances=4, **unused):
        if num_instances != 4:
            raise ValueError('Exactly four views are required')
        self.root, self.training = root, training
        self.config = config or SamplingConfig()
        frame = pd.read_excel(os.path.join(root, data_file), dtype={'patient_id':str})
        if not {'patient_id', 'label'}.issubset(frame.columns) or frame.empty:
            raise ValueError('Case table must contain nonempty patient_id and label columns')
        if frame.patient_id.isna().any() or frame.patient_id.duplicated().any():
            raise ValueError('Patient IDs must be present and unique within each split')
        if not frame.label.isin([0, 1, 2]).all():
            raise ValueError('Disease labels must use the existing 0/1/2 encoding')
        self.names = frame.patient_id.tolist()
        self.labels = frame.label.tolist()
        self.transforms = eval(transforms or 'Identity()', {'Compose':Compose, 'Identity':Identity,
                                  'NumpyType':NumpyType, 'RandomIntensityChange':RandomIntensityChange,
                                  'np':np})

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        image = np.load(os.path.join(self.root, 'vol', name+'_vol.npy'), allow_pickle=False)
        mask = np.load(os.path.join(self.root, 'seg', name+'_seg.npy'), allow_pickle=False)
        views, _ = sample_views(image, mask, self.config, self.training)
        tensors = []
        for image_view, mask_view in views:
            x, y = self.transforms([image_view[None].copy(), mask_view[None].copy()])
            tensor = np.concatenate([x[0].transpose(3,0,1,2), y.astype(np.float32)], axis=0)
            tensors.append(torch.from_numpy(np.ascontiguousarray(tensor)).float())
        return {'views':tensors}, torch.tensor(self.labels[index],dtype=torch.long), name

def Teeth_load_all_train_mil(train_file='train.xlsx', **kwargs):
    return HeterogeneousMIL(data_file=train_file, training=True, **kwargs)

def Teeth_load_all_test_mil(test_file='test.xlsx', **kwargs):
    return HeterogeneousMIL(data_file=test_file, training=False, **kwargs)
