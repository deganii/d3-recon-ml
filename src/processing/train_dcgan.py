import os
import numpy as np

from src.data.loader import DataLoader
from keras.optimizers import Adam
from src.arch.unet import get_unet
from src.arch.dcgan import DCGAN, DCGAN_discriminator
import keras.backend as K
from src.processing.folders import Folders
from keras.utils import generic_utils
import pandas as pd

def l1_loss(y_true, y_pred):
    return K.sum(K.abs(y_pred - y_true), axis=-1)

def train_dcgan(num_layers=5, filter_size=3, conv_depth=32, learn_rate=1e-3, epochs = 10,
               loss='mse', records = -1, batch_size = 32):

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
