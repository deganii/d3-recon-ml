from keras import backend as K
import numpy as np
from src.processing.folders import Folders
K.set_image_data_format('channels_last')


class DataLoader(object):

    @classmethod
    def load(cls, dataset='ds-lymphoma', set='training', records = -1):
        raw = np.load(Folders.data_folder() + '{0}-{1}.npz'.format(dataset, set), mmap_mode='r')
        if records > 0:
            return raw['data'][0:records, ...], \
                raw['labels'][0:records, 0, ...], \
                raw['labels'][0:records, 1, ...]
        else:
            return raw['data'], raw['labels'][:, 0, ...], \
                   raw['labels'][:, 1, ...]

    @classmethod
    def load_training(self, dataset='ds-lymphoma', records=-1):
        return DataLoader.load(dataset=dataset, set='training', records=records)

    @classmethod
    def load_testing(self, dataset='ds-lymphoma', records=-1):
        return DataLoader.load(dataset=dataset, set='test', records=records)
