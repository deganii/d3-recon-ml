import keras.models
import keras.losses
import scipy.misc
import numpy as np
import skimage.measure
import os
from keras_contrib.losses import DSSIMObjective
keras.losses.dssim = DSSIMObjective()

# from keras.utils.generic_utils import get_custom_objects
#
# loss = DSSIMObjective()
# get_custom_objects().update({"dssim": loss})


from src.data.loader import DataLoader
from src.processing.folders import Folders
from PIL import Image
from src.visualization.ssim_plotter import SSIMPlotter

def format_and_save(img_array, output_file, normalize=None):
    img_array = img_array.reshape([192, 192])
    if normalize is not None:
        # img_array = img_array + normalize
        img_array = img_array + np.min(img_array)
    img = Image.fromarray(np.transpose(np.uint8(255.0 * img_array / np.max(img_array))))
    scipy.misc.imsave(output_file, img)


def prediction(model_name, data, labels, weights_file='weights.h5'):
    from src.processing.train import get_unet
    from keras.optimizers import Adam
    #model = get_unet(192, 192, num_layers=6, filter_size=3,
    #                            conv_depth=32, optimizer=Adam(lr=1e-3), loss='mse')
    #model.load_weights(Folders.models_folder() + model_name + '/' + weights_file)
    model = keras.models.load_model(Folders.models_folder() + model_name + '/' + weights_file)
    mp_folder = Folders.predictions_folder() + model_name + '/'
    os.makedirs(mp_folder, exist_ok=True)
    predictions = model.predict(data, batch_size=32, verbose=0)
    predictions=predictions.astype(np.float64)
    predictions=predictions.reshape([data.shape[0],data.shape[1], data.shape[2]])
    labels=labels.reshape([labels.shape[0],labels.shape[1],labels.shape[2]])
    ssim = np.empty([predictions.shape[0]])
    for i in range(predictions.shape[0]):
        file_prefix = mp_folder + '{0:05}-'.format(i)
        # calculate the structural similarity index (SSIM) between prediction and source
        ssim[i] = skimage.measure.compare_ssim(predictions[i], labels[i])
        format_and_save(data[i], file_prefix + 'input.png')
        normalize = np.abs(min(np.min(predictions[i]), np.min(labels[i])))
        format_and_save(predictions[i], file_prefix + 'pred.png', normalize)
        format_and_save(labels[i], file_prefix + 'label.png', normalize)

    # calculate and save statistics over SSIM
    header = 'Structural Similarity Indices for {0}\n'.format(model_name)
    header += 'N:     {0}\n'.format(ssim.shape[0])
    header += 'Mean:  {0}\n'.format(np.mean(ssim))
    header += 'STDEV: {0}\n'.format(np.mean(ssim))
    header += 'MIN:   {0}, Record ({1}\n'.format(np.min(ssim), np.argmin(ssim))
    header += 'MAX:   {0}, Record ({1}\n'.format(np.max(ssim), np.argmax(ssim))
    np.savetxt(mp_folder + 'stats.txt', ssim, header=header)
    SSIMPlotter.save_plot(model_name, ssim)
    return ssim

#data, label_r, label_i = DataLoader.load_testing(records=-1)
# prediction('unet_6_layers_1e-05_lr_3px_filter_32_convd_i_retrain_50_epoch_mse', data, label_i)
#ssim_r = prediction('dcgan_6_layers_0.001_lr_3px_filter_32_convd_r', data, label_r, weights_file='gen_4_epochs.h5')
#print(np.mean(ssim_r))
#ssim_r=prediction('unet_5_layers_0.0001_lr_4px_filter_32_convd_loss_msq_r', data, label_r)