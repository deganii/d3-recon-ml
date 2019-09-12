# generator of labeled lymphoma test images
import os, glob
import h5py
import numpy as np
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.processing.folders import Folders


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cmocean
import cmocean.cm

class CommonGenerator(object):
    @classmethod
    def partitionTrainingAndTestSet(cls, set_name='ds-lymphoma'):
        data_folder = Folders.data_folder()
        data_npz = np.load(data_folder + set_name + '-all.npz')
        all_data, all_labels = data_npz['data'], data_npz['labels']
        np.random.seed(0)
        indices = np.random.permutation(all_data.shape[0])
        test_count = int(np.floor(all_data.shape[0] * 0.2))
        test_idx, training_idx = indices[:test_count], indices[test_count:]
        training_data, test_data = all_data[training_idx, :], all_data[test_idx, :]
        training_labels, test_labels = all_labels[training_idx, :, :], all_labels[test_idx, :, :]
        np.savez(os.path.join(data_folder, set_name + '-training.npz'), data=training_data, labels=training_labels)
        np.savez(os.path.join(data_folder, set_name + '-test.npz'), data=test_data, labels=test_labels)
