from keras.models import Model
from keras.layers.core import Flatten, Dense, Dropout, Activation, Lambda, Reshape
from keras.layers.convolutional import Conv2D, Deconv2D, ZeroPadding2D, UpSampling2D
from keras.layers import Input, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
import keras.backend as K
import numpy as np


def DCGAN_discriminator(img_dim, conv_depth=8, nb_filters=64,
                        model_name="DCGAN_discriminator"):
    """
    Discriminator model of the DCGAN
    args : img_dim (tuple of int)  height, width, channels
    returns : model (keras NN) the Neural Net model
    """
    # channel last
    bn_axis = -1

    # number of convs is the number of times the image can be halved
    nb_conv = int(np.floor(np.log(img_dim[1]) / np.log(2)))
    # just a list of how many filters to have at each convolutional layer
    list_filters = [nb_filters * min(conv_depth, (2 ** i)) for i in range(nb_conv)]

    # First conv
    x_input = Input(shape=img_dim, name="discriminator_input")
    x = Conv2D(list_filters[0], (3, 3), strides=(2, 2), name="disc_conv2d_1", padding="same")(x_input)
    x = BatchNormalization(axis=bn_axis)(x)
    x = LeakyReLU(0.2)(x)

    # Next convs
    for i, f in enumerate(list_filters[1:]):
        name = "disc_conv2d_%s" % (i + 2)
        x = Conv2D(f, (3, 3), strides=(2, 2), name=name, padding="same")(x)
        x = BatchNormalization(axis=bn_axis)(x)
        x = LeakyReLU(0.2)(x)

    x_flat = Flatten()(x)
    x_out = Dense(2, activation="softmax", name="disc_dense")(x_flat)
    discriminator_model = Model(inputs=x_input, outputs=x_out, name=model_name)
    return discriminator_model


def DCGAN(generator, discriminator_model, img_dim):
    gen_input = Input(shape=img_dim, name="DCGAN_input")
    generated_image = generator(gen_input)
    DCGAN_output = discriminator_model(generated_image)
    DCGAN = Model(inputs=[gen_input],
                  outputs=[generated_image, DCGAN_output],
                  name="DCGAN")
    return DCGAN
