import os
import numpy as np
from keras_contrib.losses import DSSIMObjective

from src.data.loader import DataLoader
from keras.optimizers import Adam
from src.archs.unet import get_unet
from src.archs.dcgan import DCGAN, DCGAN_discriminator
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard
import keras.backend as K
from src.processing.folders import Folders
from src.visualization.fit_plotter import FitPlotter
from keras.utils import generic_utils
import pandas as pd

def get_callbacks(model_name, batch_size = 32, save_best_only = True):
    models_folder = Folders.models_folder()
    file_suffix = '_{epoch:02d}.h5'
    if save_best_only:
        file_suffix = '.h5'
    model_checkpoint = ModelCheckpoint(models_folder + "{0}/weights".format(model_name) + file_suffix,
                                       monitor='val_loss', save_best_only=save_best_only)
    csv_logger = CSVLogger(models_folder + "{0}/perflog.csv".format(model_name),
                                            separator=',', append=False)
    tensorboard = TensorBoard(log_dir=models_folder + model_name, histogram_freq=0,
                              batch_size=batch_size, write_graph=True, write_grads=False,
                              write_images=True, embeddings_freq=0,
                              embeddings_layer_names=None, embeddings_metadata=None)
    return [model_checkpoint, csv_logger, tensorboard]


def train(model_name, model, data, labels, epochs, save_summary=True, batch_size = 32, save_best_only = True):
    """ Train a generic model and save relevant data """
    models_folder = Folders.models_folder()
    os.makedirs(models_folder + model_name, exist_ok=True)

    if save_summary:
        def summary_saver(s):
            with open(models_folder + model_name + '/summary.txt', 'a+') as f:
                print(s, file=f)
        model.summary(print_fn=summary_saver)

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


def train_unet(num_layers=5, filter_size=3, conv_depth=32, learn_rate=1e-4, epochs = 10,
               loss = 'mean_squared_error', records = -1):
    """ Train a unet model and save relevant data """
    # Step 1: load data
    train_data, train_label_r, train_label_i = DataLoader.load_training(records=records)
    img_rows, img_cols = train_data.shape[1], train_data.shape[2]

    # Step 2: Configure architecture
    modelr = get_unet(img_rows, img_cols, num_layers=num_layers, filter_size=filter_size,
                      conv_depth=conv_depth, optimizer=Adam(lr=learn_rate), loss=loss)
    # modeli = get_unet(img_rows, img_cols, num_layers=num_layers, filter_size=filter_size,
    #                   conv_depth=conv_depth, optimizer=Adam(lr=learn_rate), loss=loss)

    if loss == 'mean_squared_error':
        loss_abbrev = 'msq'
    elif isinstance(loss, DSSIMObjective):
        loss_abbrev = 'dssim'

    # Step 3: Configure Training Parameters and Train
    model_name_r = 'unet_{0}_layers_{1}_lr_{2}px_filter_{3}_convd_loss_{4}_r'.format(
        num_layers, learn_rate, filter_size, conv_depth, loss_abbrev)
    epoch_r, train_loss_r, val_loss_r = train(model_name_r, modelr, train_data, train_label_r, epochs)

    # model_name_i = 'unet_{0}_layers_{1}_lr_{2}px_filter_{3}_convd_loss_{4}_i'.format(
    #     num_layers, learn_rate, filter_size, conv_depth, loss_abbrev)
    # epoch_i, train_loss_i, val_loss_i = train(model_name_i, modeli, train_data, train_label_r, epochs)

    # (TODO) Step 4: Evaluate on Test Set
    #test_data, test_label_r, test_label_i = DataLoader.load_testing(records=records)
    return model_name_r, epoch_r, train_loss_r, val_loss_r, \
        'NoRun', 0, 0.0, 0.0
           # model_name_i, epoch_i, train_loss_i, val_loss_i


def l1_loss(y_true, y_pred):
    return K.sum(K.abs(y_pred - y_true), axis=-1)

def train_dcgan(num_layers=5, filter_size=3, conv_depth=32, learn_rate=1e-3, epochs = 10,
               loss = 'mean_squared_error', records = -1, batch_size = 32):

        # Load data
        train_data, train_label_r, train_label_i = DataLoader.load_training(records=records)
        img_rows, img_cols = train_data.shape[1], train_data.shape[2]

        # Create optimizers
        opt_dcgan = Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        opt_discriminator = Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

        # Create generator model (U-NET)
        generator_modelr = get_unet(img_rows, img_cols, num_layers=num_layers, filter_size=filter_size,
                      conv_depth=conv_depth, optimizer=Adam(lr=learn_rate), loss=loss)

        # Create discriminator model
        img_dim = (img_rows, img_cols, 1)
        discriminator_model = DCGAN_discriminator(img_dim)

        generator_modelr.compile(loss='mae', optimizer=opt_discriminator)
        discriminator_model.trainable = False
        dcgan_model = DCGAN(generator_modelr, discriminator_model, img_dim)

        loss = [l1_loss, 'binary_crossentropy']
        loss_weights = [1E1, 1]
        dcgan_model.compile(loss=loss, loss_weights=loss_weights, optimizer=opt_dcgan
                            )

        discriminator_model.trainable = True
        discriminator_model.compile(loss='binary_crossentropy', optimizer=opt_discriminator
                                    )

        print(generator_modelr.summary())
        print(discriminator_model.summary())
        print(dcgan_model.summary())

        models_folder = Folders.models_folder()
        model_name_prefix = models_folder + 'dcgan_{0}_layers_{1}_lr_{2}px_filter_{3}_convd_r'.format(
            num_layers, learn_rate, filter_size, conv_depth)

        os.makedirs(model_name_prefix, exist_ok=True)
        # train various unets on the full dataset
        df = pd.DataFrame(columns=['Epoch', 'Batch', 'D logloss', "G tot", "G L1", "G logloss"])


        # Start training
        for e in range(epochs):
            # shuffle the deck
            p = np.random.permutation(train_data.shape[0])
            train_data, train_label_r = train_data[p], train_label_r[p]

            batchGenerator = DataLoader.batch_data(train_data, train_label_r, batch_size)
            progbar = generic_utils.Progbar(train_data.shape[0])
            fake = True
            batch_counter = 0
            for (data_batch, label_r_batch) in batchGenerator:

                # Create a "real" or "fake" batch to feed the discriminator model
                if fake:  # Produce a forgery
                    X_disc = generator_modelr.predict(data_batch)
                    y_disc = np.zeros((X_disc.shape[0], 2), dtype=np.uint8)
                    y_disc[:, 0] = 1
                else:  # Supply the real thing
                    X_disc = label_r_batch
                    y_disc = np.zeros((X_disc.shape[0], 2), dtype=np.uint8)
                fake = not fake

                # Update the discriminator
                disc_loss = discriminator_model.train_on_batch(X_disc, y_disc)

                try:  # Get a batch to feed the generator model
                    X_gen, X_gen_target = next(batchGenerator)
                except StopIteration:
                    break

                y_gen = np.zeros((X_gen.shape[0], 2), dtype=np.uint8)
                y_gen[:, 1] = 1

                # Freeze the discriminator
                discriminator_model.trainable = False
                gen_loss = dcgan_model.train_on_batch(X_gen, [X_gen_target, y_gen])
                # Unfreeze the discriminator
                discriminator_model.trainable = True

                progbar.add(2*batch_size, values=[("D logloss", disc_loss),
                                                ("G tot", gen_loss[0]),
                                                ("G L1", gen_loss[1]),
                                                ("G logloss", gen_loss[2])])
                # write to CSV
                batch_counter += 2*batch_size
                pd.DataFrame(columns=['Epoch', 'Batch', 'D logloss', "G tot", "G L1", "G logloss"])
                df.loc[len(df)] = [e,batch_counter,disc_loss,gen_loss[0],gen_loss[1], gen_loss[2]]
                df.to_csv(model_name_prefix + '/perflog.csv')

            if e % 2 == 0:
                # save the generator weights
                gen_weights_path = model_name_prefix + '/gen_{0}_epochs.h5'.format(e)
                generator_modelr.save_weights(gen_weights_path, overwrite=True)

                disc_weights_path = model_name_prefix + '/disc_{0}_epochs.h5'.format(e)
                discriminator_model.save_weights(disc_weights_path, overwrite=True)

                dcgan_weights_path = model_name_prefix + '/dcgan_{0}_epochs.h5'.format(e)
                dcgan_model.save(dcgan_weights_path, overwrite=True)

        return model_name_prefix


# train a single unet on a small dataset
#train_unet(6, 3, learn_rate=1e-4, epochs=2, records=64)

# train a single unet with DSSIM loss
# train_unet(num_layers=6, filter_size=3, learn_rate=1e-4,
#           epochs=2, loss=DSSIMObjective(), records=64)

# train a toy unet for the image evolution plot test
#train_unet(num_layers=3, filter_size=3, learn_rate=1e-4, conv_depth=1, epochs=2, records=64)

# train a toy UNET + DCGAN
#train_dcgan(num_layers=3, filter_size=3, conv_depth=2, learn_rate=1e-3, epochs=2,
#                 loss='mean_squared_error', records=64, batch_size=2)

# train a large UNET + DCGAN
# train_dcgan(num_layers=7, filter_size=3, conv_depth=32, learn_rate=1e-3, epochs=15,
#                   loss='mean_squared_error', records=-1, batch_size=32)


# train_unet(num_layers=6, filter_size=3, learn_rate=1e-4, conv_depth=32, epochs=18, records=-1)
