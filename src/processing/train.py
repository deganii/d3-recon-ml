import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
from keras_contrib.losses import DSSIMObjective
from keras.layers import Activation
from src.data.loader import DataLoader
from keras.optimizers import Adam
from src.arch.unet import get_unet
from src.arch.dcgan import DCGAN, DCGAN_discriminator
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard
import keras.backend as K
from src.processing.folders import Folders
from src.visualization.fit_plotter import FitPlotter
from keras.utils import generic_utils
import pandas as pd
import inspect
import csv
import keras.layers.advanced_activations as A
import keras.backend

def get_callbacks(model_name, batch_size = 32, save_best_only = True):
    models_folder = Folders.models_folder()
    file_suffix = '_{epoch:02d}.h5'
    if save_best_only:
        file_suffix = '.h5'

    model_checkpoint = ModelCheckpoint(models_folder + "{0}/weights".format(model_name) + file_suffix,
                                       monitor='val_loss', save_best_only=save_best_only)
    csv_logger = CSVLogger(models_folder + "{0}/perflog.csv".format(model_name),
                                            separator=',', append=False)
    callbacks = [model_checkpoint, csv_logger]
    if keras.backend.backend() == 'tensoflow':
        tensorboard = TensorBoard(log_dir=models_folder + model_name, histogram_freq=0,
                              batch_size=batch_size, write_graph=True, write_grads=False,
                              write_images=True, embeddings_freq=0,
                              embeddings_layer_names=None, embeddings_metadata=None)
        callbacks.append(tensorboard)
    return callbacks


def train(model_name, model, data, labels, epochs, save_summary=True,
          batch_size=32, save_best_only=True, model_metadata=None):
    """ Train a generic model and save relevant data """
    models_folder = Folders.models_folder()
    os.makedirs(models_folder + model_name, exist_ok=True)

    if save_summary:
        def summary_saver(s):
            with open(models_folder + model_name + '/summary.txt', 'a+') as f:
                print(s, file=f)
        model.summary(print_fn=summary_saver)

    if model_metadata is not None:
        # save to a csv
        with open(models_folder + model_name +'/metadata.csv', 'w') as f:
            w = csv.writer(f)
            w.writerows(model_metadata.items())

    # Step 2: train and save best weights for the given architecture
    print('-' * 30)
    print('Fitting model {0}...'.format(model_name))
    print('-' * 30)
    history = model.fit(data, labels, batch_size=batch_size, epochs=epochs, verbose=1, shuffle=True,
             validation_split=0.2, callbacks=get_callbacks(model_name, batch_size=batch_size, save_best_only=save_best_only))

    # Step 3: Plot the validation results of the model, and save the performance data
    FitPlotter.save_plot(history.history, '{0}/train_validation.png'.format(model_name))

    val_loss = np.asarray(history.history['val_loss'])
    min_loss_epoch = np.argmin(val_loss)
    min_train_loss = history.history['loss'][min_loss_epoch]

    return min_loss_epoch, min_train_loss, val_loss[min_loss_epoch]


    # (TODO) Step 3: Save other visuals


def train_unet(descriptive_name, dataset='ds-lymphoma',
               num_layers=6, filter_size=3, conv_depth=32,
               learn_rate=1e-4, epochs=18, loss='mse', records=-1,
               separate=True,  batch_size=32, activation: object='relu',
               last_activation: object='relu', advanced_activations=False,
               a_only=False, b_only=False, output_depth=2, save_best_only=True):
    """ Train a unet model and save relevant data """

    loss_abbrev = loss
    if isinstance(loss, DSSIMObjective):
        loss_abbrev = 'dssim'

    # gather up the params
    frame = inspect.currentframe()
    _, _, _, values = inspect.getargvalues(frame)

    # Step 1: load data
    d_raw = DataLoader.load_training(dataset=dataset, records=records, separate=separate)

    # Step 2: Configure architecture
    # Step 3: Configure Training Parameters and Train

    if separate:
        suffix_a, suffix_b = 'real', 'imag'
        if 'magphase' in dataset:
            suffix_a, suffix_b = 'magnitude', 'phase'

        train_data, train_label_a, train_label_b = d_raw
        img_rows, img_cols = train_data.shape[1], train_data.shape[2]

        model_name_a = model_name_b = ''
        epoch_a = epoch_b = 0
        train_loss_a = val_loss_a = train_loss_b = val_loss_b = 0.0

        if not b_only:
            modela = get_unet(img_rows, img_cols, num_layers=num_layers, filter_size=filter_size,
                              conv_depth=conv_depth, optimizer=Adam(lr=learn_rate), loss=loss,
                              last_activation=last_activation, activation=activation,
                              advanced_activations=advanced_activations, output_depth=1)
            model_name_a = 'unet_{0}-{1}_{2}_{3}_{4}'.format(num_layers, filter_size,
                loss_abbrev, descriptive_name, suffix_a)
            epoch_a, train_loss_a, val_loss_a = train(model_name_a, modela,
                train_data, train_label_a, epochs, model_metadata=values,
                batch_size=batch_size, save_best_only=save_best_only)

        if not a_only:
            model_name_b = 'unet_{0}-{1}_{2}_{3}_{4}'.format(num_layers, filter_size,
                loss_abbrev, descriptive_name, suffix_b)
            # output_depth = 2 if split_b else 1
            modelb = get_unet(img_rows, img_cols, num_layers=num_layers, filter_size=filter_size,
                               conv_depth=conv_depth, optimizer=Adam(lr=learn_rate), loss=loss,
                              last_activation=last_activation, activation=activation,
                              advanced_activations=advanced_activations, output_depth=output_depth)
            epoch_b, train_loss_b, val_loss_b = train(model_name_b, modelb,
                train_data, train_label_b, epochs, model_metadata=values,
                batch_size=batch_size, save_best_only=save_best_only)

        return model_name_a, epoch_a, train_loss_a, val_loss_a, \
             model_name_b, epoch_b, train_loss_b, val_loss_b
    else:
        train_data, train_label = d_raw
        img_rows, img_cols = train_data.shape[1], train_data.shape[2]

        model = get_unet(img_rows, img_cols, num_layers=num_layers, filter_size=filter_size,
                         conv_depth=conv_depth, optimizer=Adam(lr=learn_rate), loss=loss,
                         output_depth=output_depth, activation=activation, advanced_activations=advanced_activations,
                         last_activation=last_activation)

        model_name = 'unet_{0}-{1}_{2}_{3}'.format(num_layers, filter_size, loss_abbrev, descriptive_name)

        epoch, train_loss, val_loss = train(model_name, model, train_data,
            train_label, epochs, model_metadata=values, batch_size=batch_size,
                save_best_only=save_best_only)

        return model_name, epoch, train_loss, val_loss

# train a single unet on a small dataset
#train_unet('small-dataset-test', 6, 3, learn_rate=1e-4, epochs=2, records=64)

# train a single unet with DSSIM loss
# train_unet('dssim_test', num_layers=6, filter_size=3, learn_rate=1e-4,
#           epochs=2, loss=DSSIMObjective(), records=64)

# train a toy unet for the image evolution plot test
#train_unet('evplot', num_layers=3, filter_size=3, learn_rate=1e-4, conv_depth=1, epochs=2, records=64)

# train a toy UNET + DCGAN
#train_dcgan(num_layers=3, filter_size=3, conv_depth=2, learn_rate=1e-3, epochs=2,
#                 loss='mean_squared_error', records=64, batch_size=2)

# train a large UNET + DCGAN
# train_dcgan(num_layers=7, filter_size=3, conv_depth=32, learn_rate=1e-3, epochs=15,
#                   loss='mean_squared_error', records=-1, batch_size=32)



# train_unet('dual-test', num_layers=6, filter_size=3,
#            learn_rate=1e-4, conv_depth=32, epochs=18,
#            records=-1, separate=False, batch_size=16,
#            activation='relu', last_activation='relu')
#

# train_unet('prelu-test', num_layers=6, filter_size=3,
#            learn_rate=1e-4, conv_depth=32, epochs=25,
#            records=-1, separate=True, batch_size=16,
#            activation=A.PReLU, advanced_activations=True,
#            last_activation=A.PReLU)

# train_unet('prelu-dual-test', num_layers=6, filter_size=3,
#            learn_rate=1e-4, conv_depth=32, epochs=18,
#            records=-1, separate=False, batch_size=16,
#            activation=A.PReLU, advanced_activations=True,
#            last_activation='relu')


# train_unet('prelu-test', num_layers=6, filter_size=3,
#            learn_rate=1e-4, conv_depth=32, epochs=25,
#            records=-1, separate=True, batch_size=16,
#            activation=A.PReLU, advanced_activations=True,
#            last_activation=A.PReLU, b_only=True)

# train_unet('prelu-test-magphase', dataset='ds-lymphoma-magphase',
#            num_layers=6, filter_size=3,
#            learn_rate=1e-4, conv_depth=32, epochs=25,
#            records=-1, separate=True, batch_size=16,
#            activation=A.PReLU, advanced_activations=True,
#            last_activation=A.PReLU)


# train_unet('prelu-split-phase-only', dataset='ds-lymphoma-magphase-splitphase',
#            num_layers=6, filter_size=3,
#            learn_rate=1e-4, conv_depth=32, epochs=25,
#            records=-1, separate=False, b_only=True,
#            batch_size=16, activation=A.PReLU, advanced_activations=True,
#            last_activation=A.PReLU)
#
#


# centered text test
# train_unet('text', num_layers=6, filter_size=3,
#     dataset = 'ds-text', save_best_only=False,
#     learn_rate=1e-4, conv_depth=32, epochs=25,
#     records=-1, separate=False, batch_size=16,
#     activation=A.PReLU, advanced_activations=True,
#     last_activation='sigmoid', output_depth=1)