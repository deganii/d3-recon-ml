import numpy as np
import scipy
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras import backend as K
import keras.layers.advanced_activations as A
from src.metrics.entropy import ImageComparator

K.set_image_data_format('channels_last')


def get_holo_transfer(img_rows, img_cols, filter_size=32, conv_depth=1,
             optimizer=Adam(lr=1e-4), loss='mean_squared_error', metrics=None,
             output_depth=1, activation: object='relu'):

    advanced_activation = activation
    inputs = Input((img_rows, img_cols, 1))
    conv = Conv2D(conv_depth, (filter_size, filter_size),
                  activation=activation, padding='same')(inputs)
    conv = advanced_activation()(conv)

    model = Model(inputs=[inputs], outputs=[conv])
    if metrics is None:
        metrics = ['accuracy'] #ImageComparator.ssim_metric]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


# test case
# modelr = get_unet(img_rows, img_cols, num_layers=6, filter_size=3,
#                   optimizer=Adam(lr=1e-4), loss='mean_squared_error')
