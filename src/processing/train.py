import os
from src.data.loader import DataLoader
from keras.optimizers import Adam
from src.archs.unet import get_unet
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard

from src.processing.folders import Folders
from src.visualization.fit_plotter import FitPlotter


def train(model_name, model, data, labels, epochs, debug=False):
    """ Train a generic model and save relevant data """
    # Step 1: define all callbacks and data to log
    models_folder = Folders.models_folder()
    model_checkpoint = ModelCheckpoint(models_folder + "{0}.h5".format(model_name),
                                       monitor='val_loss', save_best_only=True)
    csv_logger = CSVLogger(models_folder + "{0}.csv".format(model_name),
                                            separator=',', append=False)

    os.makedirs(models_folder + model_name, exist_ok=True)
    tensorboard = TensorBoard(log_dir=models_folder + model_name, histogram_freq=0,
                              batch_size=32, write_graph=True, write_grads=False,
                              write_images=False, embeddings_freq=0,
                              embeddings_layer_names=None, embeddings_metadata=None)

    if debug:
        model.summary()

    # Step 2: train and save best weights for the given architecture
    print('-' * 30)
    print('Fitting model {0}...'.format(model_name))
    print('-' * 30)
    history = model.fit(data, labels, batch_size=32, epochs=epochs, verbose=1, shuffle=True,
             validation_split=0.2, callbacks=[model_checkpoint, csv_logger, tensorboard])

    # Step 3: Plot the validation results of the model, and save the performance data
    FitPlotter.save_plot(history.history, '{0}.png'.format(model_name))

    # (TODO) Step 3: Save other visuals



def train_unet(num_layers, filter_size, learn_rate, epochs = 10, records = -1):
    """ Train a unet model and save relevant data """
    # Step 1: load data
    train_data, train_label_r, train_label_i = DataLoader.load_training(records=records)
    img_rows, img_cols = train_data.shape[1], train_data.shape[2]

    # Step 2: Configure architecture
    modelr = get_unet(img_rows, img_cols, num_layers=num_layers, filter_size=3,
                      optimizer=Adam(lr=learn_rate), loss='mean_squared_error')
    modeli = get_unet(img_rows, img_cols, num_layers=num_layers, filter_size=3,
                      optimizer=Adam(lr=learn_rate), loss='mean_squared_error')

    # Step 3: Configure Training Parameters and Train
    model_name = 'unet_r_{0}_layers_{1}_lr_{2}px_filter'.format(num_layers, learn_rate, filter_size)
    train(model_name, modelr, train_data, train_label_r, epochs)

    model_name = 'unet_i_{0}_layers_{1}_lr_{2}px_filter'.format(num_layers, learn_rate, filter_size)
    train(model_name, modeli, train_data, train_label_r, epochs)

    # (TODO) Step 4: Evaluate on Test Set
    test_data, test_label_r, test_label_i = DataLoader.load_testing(records=records)

# train a single unet on a small dataset
train_unet(6, 3, 1e-4, epochs=2, records=64)

# train various unets
# for lr in [1e-3, 1e-4, 1e-5]:
#     for layers in [4,5,6]:
#         for filters in [2,3,4]:
#             train_unet(layers, filters, lr)
