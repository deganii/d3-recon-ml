from keras_contrib.losses import DSSIMObjective

from src.data.loader import DataLoader
from src.processing.folders import Folders
from src.processing.train import train
import keras.models
from keras.optimizers import Adam
import keras.backend as K

#from keras_contrib.losses import DSSIMObjective
#keras.losses.dssim = DSSIMObjective()

def l2_loss(y_true, y_pred):
    return K.sum(K.square(y_pred - y_true), axis=-1)

def retrain(model_name, new_model_name, data, labels, learn_rate=1e-4,
            epochs=10, loss='mean_squared_error', recompile=False):
    # Step 1: Load the model
    model = keras.models.load_model(Folders.models_folder() + model_name + '/weights.h5')

    # Step 2: Recompile with the new loss function / learn rate
    # TODO:
    # if loss == 'dssim':
    #     #loss = [l2_loss, DSSIMObjective()]
    #     #loss_weights = [1E1, 1]
    #     model.compile(optimizer=Adam(lr=learn_rate), loss=loss,
    #                   metrics=[keras.metrics.mean_absolute_error])
    # else:
    if recompile:
        model.compile(optimizer=Adam(lr=learn_rate), loss=loss,
                      metrics=[keras.metrics.mean_squared_error])

    # Step 3: Configure Training Parameters and Train
    epoch, train_loss, val_loss = train(new_model_name, model, data, labels, epochs)

    return model_name, epoch, train_loss, val_loss

# load and re-train the large u-net
train_data, train_label_r, train_label_i = DataLoader.load_training()
model_name = 'unet_6_layers_1e-05_lr_3px_filter_32_convd_i'

m, e, t, v = retrain(model_name, model_name+'_retrain_50_epoch_mse',
                    train_data, train_label_i,
                    loss='mean_squared_error',
                    learn_rate=1e-4, epochs=50)
