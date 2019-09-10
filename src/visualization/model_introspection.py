import warnings
import matplotlib
import numpy as np
import scipy.stats
import scipy.stats as st
from scipy.stats import norm
from scipy.stats import pearson3
from scipy.stats import skewnorm
from src.processing.folders import Folders
from tqdm import tqdm
import keras.models

matplotlib.use('Agg')
import matplotlib.pyplot as plt

class ModelIntrospection(object):

    @classmethod
    def unet_filters(cls, model_name=None, model=None, weights_file='weights.h5',):
        if model is None:
            model = keras.models.load_model(Folders.models_folder() + model_name + '/' + weights_file)

        for layer in model.layers:
            if layer.name.startswith('conv2d'):
                num_filters = layer.weights[0].shape[-1]
                filter_dim = layer.weights[0].shape[0:1]


        model.layers[1].weights[0][:, :, 0, 0]
        return model.layers

if __name__ == "__main__":
    # Test case
    model_name = 'unet_6-3_mse_mnist-3750'
    ModelIntrospection.unet_filters(model_name)


