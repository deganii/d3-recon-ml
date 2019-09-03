
from keras.utils.generic_utils import get_custom_objects
from keras_contrib.losses import DSSIMObjective
from keras import backend as K
from keras import losses

import keras_contrib.backend as KC

# an average of DSSIM and MSE
def mse_ssim_loss():
    dssim = DSSIMObjective()
    mse = losses.mean_squared_error

    # Create a loss function that adds the MSE loss to SSIM loss
    def loss(y_true, y_pred):
        #return K.mean(mse(y_true, y_pred), dssim(y_true,y_pred), axis=-1)
        return mse(y_true, y_pred) + dssim(y_true, y_pred)

    # Return a function
    return loss


get_custom_objects().update({"mse_ssim": mse_ssim_loss()})
