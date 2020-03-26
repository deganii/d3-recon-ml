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
             output_depth=1, activation: object='relu',
             advanced_activations=False, extra_phase_layers=0):

    # advanced_activation = activation
    # if advanced_activations:
    #     activation = None

    inputs = Input((img_rows, img_cols, 1))
    conv = Conv2D(conv_depth, (filter_size, filter_size), name='mag',
                  activation=activation, padding='same')(inputs)
    # if advanced_activations:
    #     conv = advanced_activation()(conv)

    if output_depth == 3:
        # feed the magnitude output into the phase locator
        phase_input = concatenate([conv, inputs], axis=3)
        phase_r = Conv2D(conv_depth, (filter_size, filter_size), name='p_r0',
                      activation=activation, padding='same')(phase_input)
        phase_i = Conv2D(conv_depth, (filter_size, filter_size), name='p_i0',
                      activation=None, padding='same')(phase_input)
        phase_i = A.PReLU()(phase_i)
        # if advanced_activations:
        #     phase_r = advanced_activation()(phase_r)
        #     phase_i = advanced_activation()(phase_i)

        ## Add n more layers to phase detection...
        for el in range(extra_phase_layers):
            phase_r = Conv2D(conv_depth, (filter_size, filter_size), name='p_r{0}'.format(el+1),
                             activation=activation, padding='same')(phase_r)

            # give the imaginary component more help from the magnitude...
            # i_input = concatenate([conv, phase_i], axis=3)
            phase_i = Conv2D(conv_depth, (filter_size, filter_size), name='p_i{0}'.format(el+1),
                          activation=None, padding='same')(phase_i)
            phase_i = A.PReLU()(phase_i)
            # if advanced_activation:
            #     phase_r = advanced_activation()(phase_r)
            #     phase_i = advanced_activation()(phase_i)

        mag_phase_output = concatenate([conv, phase_r, phase_i], axis=3)
        model = Model(inputs=[inputs], outputs=[mag_phase_output])
    else:
        model = Model(inputs=[inputs], outputs=[conv])
    if metrics is None:
        metrics = ['accuracy'] #ImageComparator.ssim_metric]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


# test case
# modelr = get_unet(img_rows, img_cols, num_layers=6, filter_size=3,
#                   optimizer=Adam(lr=1e-4), loss='mean_squared_error')
