import numpy as np
import scipy
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras import backend as K
import keras.layers.advanced_activations as A
from src.metrics.entropy import ImageComparator

K.set_image_data_format('channels_last')


def get_unet(img_rows, img_cols, num_layers = 4, filter_size=3, conv_depth=32,
             optimizer=Adam(lr=1e-4), loss='mean_squared_error', metrics=None,
             output_depth=1, activation: object='relu', advanced_activations=False,
             last_activation='relu'):

    advanced_activation = activation
    if advanced_activations:
        activation = None
        last_activation = None

    inputs = Input((img_rows, img_cols, 1))
    last_in= inputs
    conv_dict = {}
    for i in range(num_layers):
        conv = Conv2D(conv_depth*2**i, (filter_size, filter_size),
                      activation=activation, padding='same')(last_in)
        if advanced_activations:
            conv = advanced_activation()(conv)
        conv = Conv2D(conv_depth*2**i, (filter_size, filter_size),
                      activation=activation, padding='same')(conv)
        if advanced_activations:
            conv = advanced_activation()(conv)

        conv_dict[i] = conv
        if i < num_layers:
            pool = MaxPooling2D(pool_size=(2, 2))(conv)
            last_in = pool

    last_in = conv_dict[i]
    for i in range(num_layers-1, 0, -1):
        up = concatenate([Conv2DTranspose(conv_depth*2**i, (2, 2),
                        strides=(2, 2), padding='same')(last_in), conv_dict[i-1]], axis=3)
        conv = Conv2D(conv_depth*2**i, (filter_size, filter_size),
                      activation=activation, padding='same')(up)
        if advanced_activations:
            conv = advanced_activation()(conv)
        last_in = Conv2D(conv_depth*2**i, (filter_size, filter_size),
                         activation=activation, padding='same')(conv)
        if advanced_activations:
            last_in = advanced_activation()(last_in)

    # if output_depth == 2:
    #     for i in range(2):
    #         last_in = Conv2D(conv_depth, (filter_size, filter_size),
    #           activation=activation, padding='same')(last_in)

    conv_last = Conv2D(output_depth, (1, 1), activation=last_activation,
                       padding='same')(last_in)
    if advanced_activations:
        conv_last = advanced_activation()(conv_last)
    model = Model(inputs=[inputs], outputs=[conv_last])

    if metrics is None:
        metrics = ['accuracy'] #ImageComparator.ssim_metric]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


# test case
# modelr = get_unet(img_rows, img_cols, num_layers=6, filter_size=3,
#                   optimizer=Adam(lr=1e-4), loss='mean_squared_error')
