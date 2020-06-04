import os
from keras import backend as K
import numpy as np
from src.processing.folders import Folders
K.set_image_data_format('channels_last')



class DataLoader(object):

    @classmethod
    def load(cls, dataset='ds-lymphoma', set='training', records = -1, separate=True):
        data_path = Folders.data_folder() + '{0}-{1}.npz'.format(dataset, set)
        if os.path.isfile(data_path):
            raw = np.load(data_path, mmap_mode='r')
        if records > 0:
            # additional logic for efficient caching of small subsets of records
            raw_trunc = Folders.data_folder() + '{0}-{1}-n{2}.npz'.format(dataset, set, records)
            if os.path.isfile(raw_trunc):
                raw_n = np.load(raw_trunc, mmap_mode='r')

                if separate:
                    return raw_n['data'], raw_n['labels'][:, ..., 0], raw_n['labels'][:, ..., 1]
                else:
                    if len(raw_n['labels'].shape) == 3:
                        return raw_n['data'][..., np.newaxis], raw_n['labels'][..., np.newaxis]
                    else:
                        return raw_n['data'], raw_n['labels'] # np.moveaxis(np.squeeze(), 1, -1)
            else:
                if records > raw['data'].shape[0]:
                    records = raw['data'].shape[0]
                data, labels = raw['data'][0:records, ...], raw['labels'][0:records, ...]
                np.savez(raw_trunc, data=data, labels=labels)

                if separate:
                    return data, labels[..., 0], labels[..., 1]
                else:
                    if len(labels.shape) == 3:
                        return data[..., np.newaxis], labels[..., np.newaxis]
                    else:
                        return data, labels #np.moveaxis(np.squeeze(labels), 1, -1)
        else:
            if separate:
                return raw['data'], raw['labels'][..., 0], raw['labels'][..., 1]
            else:
                if len(raw['labels'].shape) == 3:
                    return raw['data'][..., np.newaxis], raw['labels'][..., np.newaxis]
                else:
                    return raw['data'], raw['labels'] #np.moveaxis(np.squeeze(raw['labels']), 1, -1)

    @classmethod
    def load_training(cls, dataset='ds-lymphoma', records=-1, separate=True):
        return DataLoader.load(dataset=dataset, set='training', records=records, separate=separate)

    @classmethod
    def load_testing(cls, dataset='ds-lymphoma', records=-1, separate=True):
        return DataLoader.load(dataset=dataset, set='test', records=records, separate=separate)

    @classmethod
    def batch_data(cls, train_data, train_labels, batch_size):
        """ Simple sequential chunks of data """
        for batch in range(int(np.ceil(train_data.shape[0] / batch_size))):
            start = batch_size * batch
            end = start + batch_size
            if end > train_data.shape[0]:
                yield train_data[-batch_size:, ...], \
                        train_labels[-batch_size:, ...]
            else:
                yield train_data[start:end, ...], \
                        train_labels[start:end, ...]

# Load just 64 training records
# DataLoader.load_training(records=64)

# DataLoader.load_training(records=64)
