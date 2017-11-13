import os
from keras import backend as K
import numpy as np
from src.processing.folders import Folders
K.set_image_data_format('channels_last')

class DataLoader(object):

    @classmethod
    def load(cls, dataset='ds-lymphoma', set='training', records = -1):
        raw = np.load(Folders.data_folder() + '{0}-{1}.npz'.format(dataset, set), mmap_mode='r')
        if records > 0:
            # additional logic for efficient caching of small subsets of records
            raw_trunc = Folders.data_folder() + '{0}-{1}-n{2}.npz'.format(dataset, set, records)
            if os.path.isfile(raw_trunc):
                raw_n = np.load(raw_trunc, mmap_mode='r')
                return raw_n['data'], raw_n['labels'][:, 0, ...], raw_n['labels'][:, 1, ...]
            else :
                data, labels =  raw['data'][0:records, ...], raw['labels'][0:records, ...]
                np.savez(raw_trunc, data=data, labels=labels)
                return data, labels[:, 0, ...], labels[:, 1, ...]
        else:
            return raw['data'], raw['labels'][:, 0, ...], raw['labels'][:, 1, ...]

    @classmethod
    def load_training(self, dataset='ds-lymphoma', records=-1):
        return DataLoader.load(dataset=dataset, set='training', records=records)

    @classmethod
    def load_testing(self, dataset='ds-lymphoma', records=-1):
        return DataLoader.load(dataset=dataset, set='test', records=records)

# Load just 64 training records
# DataLoader.load_training(records=64)