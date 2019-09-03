import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import scipy.sparse.linalg
from sys import platform

# on windows box (with AMD GPU) use keras plaidml backend...
if platform == "win32":
    os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
    # import plaidml.keras
    # plaidml.keras.install_backend()


import keras.layers.advanced_activations as A

from src.processing.train import train_unet
from src.loss.avg import mse_ssim_loss
from src.processing.predict import prediction
from src.processing.predict import prediction_with_merge
from src.data.loader import DataLoader

# all directories are relative to the src folder

# Run 100 Epochs
# train_unet('nucleus-4dirs', dataset='0129-2dirs',
#            num_layers=6, filter_size=3,
#            learn_rate=1e-4, conv_depth=32, epochs=100,
#            records=-1, batch_size=16, activation=A.PReLU,
#            advanced_activations=True, last_activation=A.PReLU)


# Run 25 epochs, saving every epoch's weights (if an evolution plot is desired)
# train_unet('nucleus-25-epochs', dataset='nucleus',
#             num_layers=6, filter_size=3, save_best_only=False,
#             learn_rate=1e-4, conv_depth=32, epochs=25,
#             records=-1, batch_size=16, activation=A.PReLU,
#             advanced_activations=True, last_activation=A.PReLU)

# # run 5 epochs with small 32-image test dataset (make sure NN architecture works)
# train_unet('nicha_sarvesh_test', dataset='test-mydataset-dapi',
#           num_layers=6, filter_size=3,
#           learn_rate=1e-4, conv_depth=32, epochs=5,
#           records=-1, batch_size=16, activation=A.PReLU,
#           advanced_activations=True, last_activation=A.PReLU)


# run 25 epochs with custom loss function
#train_unet('nucleus-custom-loss', dataset='0129-2dirs',
#           num_layers=6, filter_size=3, loss='mse_ssim',
#           learn_rate=1e-4, conv_depth=32, epochs=25,
#           records=-1, batch_size=16, activation=A.PReLU,
#           advanced_activations=True, last_activation=A.PReLU)


# Run 25 epochs, saving every 5 epoch's weights (if an evolution plot is desired)
# train_unet('nucleus-25-epochs', dataset='nucleus',
#             num_layers=6, filter_size=3, save_best_only=False,
#             learn_rate=1e-4, conv_depth=32, epochs=25,
#             records=-1, batch_size=16, activation=A.PReLU,
#             advanced_activations=True, last_activation=A.PReLU, period=5)





#load a folder of full-sized images, tile them, run the model, and then merge the predictions
# patches, padded_results = prediction_with_merge('unet_6-3_mse_ssim_test_031418_100epochs',
#                '/home/ubuntu/nucleus-prediction/data/smooth_merge_test/input/',
#                '/home/ubuntu/nucleus-prediction/data/smooth_merge_test/output/',
#                weights_file='weights_95.h5', debug_images=True, high_quality=False)

# to shutdown AWS instance automatically
# import subprocess
# subprocess.call(['sudo','shutdown','-h','0'])


# train_unet('text-full', dataset='ds-text', records=-1,
#            num_layers=6, filter_size=3,
#            learn_rate=1e-4, conv_depth=32, epochs=100,
#            batch_size=16, activation=A.PReLU,
#            separate=False, advanced_activations=True,
#            last_activation='sigmoid', output_depth=1)
# data, label_text = DataLoader.load_testing(records=-1, separate = False,
#             dataset='ds-text')
# ssim = prediction('unet_6-3_mse_text-full', data, label_text, transpose=False)

# train_unet('mnist-3750', dataset='mnist-diffraction', records=-1,
#            num_layers=6, filter_size=3,
#            learn_rate=1e-4, conv_depth=32, epochs=12,
#            batch_size=16, activation=A.PReLU,
#            separate=False, advanced_activations=True,
#            last_activation='sigmoid', output_depth=1)
data, label_text = DataLoader.load_testing(records=-1, separate = False,
            dataset='mnist-diffraction')
ssim = prediction('unet_6-3_mse_mnist-3750', data, label_text, transpose=False)


# TODO: Create a simple folder that can be uploaded to the "Experiments" dropbox
# TODO: Add platform and GPU to the model metadata file
# TODO: Change model name to remove "unet_6_3" etc. This is now in metadata.csv
# TODO: Add a text descrption the "train_unet" function which gets stored in metadata.csv
# TODO: Add model train/predict timings for easy performance comparison
# TODO: Organize SSIM best to worst, and create a "sample" folder with some examples
# TODO: Predict on every epoch for small subset and save to an evolution folder
# TODO: update train/validation curve on each epoch for live monitoring
# TODO: Add rotation/stretching and partial predictions (train on whole)
# TODO: vectorize graphs so they are easily incorporated into paper.