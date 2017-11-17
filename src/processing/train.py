import os
import numpy as np
from keras_contrib.losses import DSSIMObjective

from src.data.loader import DataLoader
from keras.optimizers import Adam
from src.archs.unet import get_unet
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard

from src.processing.folders import Folders
from src.visualization.fit_plotter import FitPlotter


def train(model_name, model, data, labels, epochs, save_summary=True):
    """ Train a generic model and save relevant data """
    # Step 1: define all callbacks and data to log
    models_folder = Folders.models_folder()
    model_checkpoint = ModelCheckpoint(models_folder + "{0}/weights.h5".format(model_name),
                                       monitor='val_loss', save_best_only=True)
    csv_logger = CSVLogger(models_folder + "{0}/perflog.csv".format(model_name),
                                            separator=',', append=False)

    os.makedirs(models_folder + model_name, exist_ok=True)
    tensorboard = TensorBoard(log_dir=models_folder + model_name, histogram_freq=0,
                              batch_size=32, write_graph=True, write_grads=False,
                              write_images=False, embeddings_freq=0,
                              embeddings_layer_names=None, embeddings_metadata=None)

    if save_summary:
        def summary_saver(s):
            with open(models_folder + model_name + '/summary.txt', 'a+') as f:
                print(s, file=f)
        model.summary(print_fn=summary_saver)

    # Step 2: train and save best weights for the given architecture
    print('-' * 30)
    print('Fitting model {0}...'.format(model_name))
    print('-' * 30)
    history = model.fit(data, labels, batch_size=32, epochs=epochs, verbose=1, shuffle=True,
             validation_split=0.2, callbacks=[model_checkpoint, csv_logger, tensorboard])

    # Step 3: Plot the validation results of the model, and save the performance data
    FitPlotter.save_plot(history.history, '{0}/train_validation.png'.format(model_name))

    val_loss = np.asarray(history.history['val_loss'])
    min_loss_epoch = np.argmin(val_loss)
    min_train_loss = history.history['loss'][min_loss_epoch]

    return min_loss_epoch, min_train_loss, val_loss[min_loss_epoch]


    # (TODO) Step 3: Save other visuals


def train_unet(num_layers=5, filter_size=3, conv_depth=32, learn_rate=1e-4, epochs = 10,
               loss = 'mean_squared_error', records = -1, ):
    """ Train a unet model and save relevant data """
    # Step 1: load data
    train_data, train_label_r, train_label_i = DataLoader.load_training(records=records)
    img_rows, img_cols = train_data.shape[1], train_data.shape[2]

    # Step 2: Configure architecture
    modelr = get_unet(img_rows, img_cols, num_layers=num_layers, filter_size=filter_size,
                      conv_depth=conv_depth, optimizer=Adam(lr=learn_rate), loss=loss)
    modeli = get_unet(img_rows, img_cols, num_layers=num_layers, filter_size=filter_size,
                      conv_depth=conv_depth, optimizer=Adam(lr=learn_rate), loss=loss)

    # Step 3: Configure Training Parameters and Train
    model_name_r = 'unet_{0}_layers_{1}_lr_{2}px_filter_{3}_convd_r'.format(
        num_layers, learn_rate, filter_size, conv_depth)
    epoch_r, train_loss_r, val_loss_r = train(model_name_r, modelr, train_data, train_label_r, epochs)

    model_name_i = 'unet_{0}_layers_{1}_lr_{2}px_filter_{3}_convd_i'.format(
        num_layers, learn_rate, filter_size, conv_depth)
    epoch_i, train_loss_i, val_loss_i = train(model_name_i, modeli, train_data, train_label_r, epochs)

    # (TODO) Step 4: Evaluate on Test Set
    #test_data, test_label_r, test_label_i = DataLoader.load_testing(records=records)
    return model_name_r, epoch_r, train_loss_r, val_loss_r, \
           model_name_i, epoch_i, train_loss_i, val_loss_i

# train a single unet on a small dataset
#train_unet(6, 3, learn_rate=1e-4, epochs=2, records=64)

# train a single unet with DSSIM loss
#train_unet(6, 3, 1e-4, epochs=2, loss=DSSIMObjective(), records=64)

# train a toy unet for the image evolution plot test
#train_unet(num_layers=3, filter_size=3, learn_rate=1e-4, conv_depth=1, epochs=2, records=64)



