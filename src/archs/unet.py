import numpy as np
import scipy
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras import backend as K

from src.metrics.entropy import ImageComparator

K.set_image_data_format('channels_last')


def get_unet(img_rows, img_cols, num_layers = 4, filter_size=3, conv_depth=32,
             optimizer=Adam(lr=1e-4), loss='mean_squared_error', metrics=None):
    inputs = Input((img_rows, img_cols, 1))
    last_in= inputs
    conv_dict = {}
    for i in range(num_layers):
        conv = Conv2D(conv_depth*2**i, (filter_size, filter_size),
                      activation='relu', padding='same')(last_in)
        conv = Conv2D(conv_depth*2**i, (filter_size, filter_size),
                      activation='relu', padding='same')(conv)
        conv_dict[i] = conv
        if i < num_layers:
            pool = MaxPooling2D(pool_size=(2, 2))(conv)
            last_in = pool

    last_in = conv_dict[i]
    for i in range(num_layers-1, 0, -1):
        up = concatenate([Conv2DTranspose(conv_depth*2**i, (2, 2),
                        strides=(2, 2), padding='same')(last_in), conv_dict[i-1]], axis=3)
        conv = Conv2D(conv_depth*2**i, (filter_size, filter_size),
                      activation='relu', padding='same')(up)
        last_in = Conv2D(conv_depth*2**i, (filter_size, filter_size),
                         activation='relu', padding='same')(conv)

    conv_last = Conv2D(1, (1, 1), activation='relu', padding='same')(last_in)
    model = Model(inputs=[inputs], outputs=[conv_last])

    if metrics is None:
        metrics = ['accuracy'] #ImageComparator.ssim_metric]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


# test case
# modelr = get_unet(img_rows, img_cols, num_layers=6, filter_size=3,
#                   optimizer=Adam(lr=1e-4), loss='mean_squared_error')
