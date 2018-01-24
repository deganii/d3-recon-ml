import keras.models
import keras.losses
import scipy.misc
import numpy as np
import skimage.measure
import os
from keras_contrib.losses import DSSIMObjective
keras.losses.dssim = DSSIMObjective()

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cmocean
import cmocean.cm


# from keras.utils.generic_utils import get_custom_objects
#
# loss = DSSIMObjective()
# get_custom_objects().update({"dssim": loss})


from src.data.loader import DataLoader
from src.processing.folders import Folders
from PIL import Image
from src.visualization.ssim_plotter import SSIMPlotter

def format_and_save(img_array, output_file, dmin=None, dmax=None):
    img_array = img_array.reshape([192, 192])
    if dmin is not None and dmax is not None:
        img_array = (img_array + dmin) / (dmax + dmin)
    else:
        img_array = img_array / np.max(img_array)
    img = Image.fromarray(np.transpose(np.uint8(255.0 * img_array)))
    #print('Norm: {0}, Max: {1}\n'.format(dmin, dmax))
    scipy.misc.imsave(output_file, img)

def format_and_save_phase(img_array, output_file):
    plt.imsave(output_file, img_array, cmap=cmocean.cm.phase,  vmin=-np.pi, vmax=np.pi)

def prediction(model_name, data, labels, save_err_img = False,
               phase_mapping= False, weights_file='weights.h5'):
    from src.processing.train import get_unet
    from keras.optimizers import Adam
    #model = get_unet(192, 192, num_layers=6, filter_size=3,
    #                            conv_depth=32, optimizer=Adam(lr=1e-3), loss='mse')
    #model.load_weights(Folders.models_folder() + model_name + '/' + weights_file)
    model = keras.models.load_model(Folders.models_folder() + model_name + '/' + weights_file)
    mp_folder = Folders.predictions_folder() + model_name + '-n{0}/'.format(data.shape[0])
    os.makedirs(mp_folder, exist_ok=True)
    predictions = model.predict(data, batch_size=32, verbose=0)

    predictions = predictions.astype(np.float64)
    # check if the network predicts the complex valued image or just one component
    complex_valued = predictions.shape[-1] == 2


    # we are color-mapping a phase pre
    if phase_mapping and complex_valued:
        predictions_complex = predictions[..., 0]  + 1j * predictions[..., 1]
        predictions = np.angle(predictions_complex)
        labels_complex = labels[..., 0] + 1j * labels[..., 1]
        labels = np.angle(labels_complex)

    # update for phase flattening
    complex_valued = predictions.shape[-1] == 2

    if complex_valued:
        ssim = np.empty([predictions.shape[0], predictions.shape[-1]])
        ms_err = np.empty([predictions.shape[0], predictions.shape[-1]])
    else:
        predictions=predictions.reshape([data.shape[0],data.shape[1], data.shape[2]])
        labels=labels.reshape([labels.shape[0],labels.shape[1],labels.shape[2]])
        ssim = np.empty([predictions.shape[0]])
        ms_err = np.empty([predictions.shape[0]])

    for i in range(predictions.shape[0]):
        file_prefix = mp_folder + '{0:05}-'.format(i)
        format_and_save(data[i], file_prefix + 'input.png')

        # calculate the structural similarity index (SSIM) between prediction and source
        if complex_valued:
            for j,name in enumerate(['real', 'imag']):
                ssim[i, j] = skimage.measure.compare_ssim(
                    predictions[i, ..., j], labels[i, ..., j])

                # todo: check for min > 0 (don't need to add)
                # todo: check for max <= 0?
                dmin = np.abs(min(0, np.min(predictions[i, ..., j]),
                                       np.min(labels[i, ..., j])))
                dmax = max(np.max(predictions[i, ..., j]),
                        np.max(labels[i, ..., j]))

                format_and_save(predictions[i, ..., j],
                    file_prefix + 'pred-{0}.png'.format(name), dmin, dmax)
                format_and_save(labels[i, ..., j],
                    file_prefix + 'label-{0}.png'.format(name), dmin, dmax)
        else:
            ssim[i] = skimage.measure.compare_ssim(predictions[i], labels[i])

            sq_err = np.square(predictions[i] - labels[i])

            dmin = np.abs(min(np.min(predictions[i]), np.min(labels[i])))
            dmax = np.abs(max(np.max(predictions[i]), np.max(labels[i])))
            format_and_save(predictions[i], file_prefix + 'pred.png', dmin, dmax)
            format_and_save(labels[i], file_prefix + 'label.png', dmin, dmax)
            if phase_mapping:
                # mean phase error accounting for loop-over
                ms_err[i] = np.mean(np.abs(np.exp(1j * predictions[i]) -  np.exp(1j * labels[i])))
                format_and_save_phase(predictions[i], file_prefix + 'pred.png')
                format_and_save_phase(labels[i], file_prefix + 'label.png')
            else:
                ms_err[i] = np.mean(sq_err)
                format_and_save(predictions[i], file_prefix + 'pred.png', dmin, dmax)
                format_and_save(labels[i], file_prefix + 'label.png', dmin, dmax)

            if save_err_img:
                smin, smax = np.min(sq_err), np.max(sq_err)
                format_and_save(sq_err, file_prefix + 'err.png', smin, smax)

    # calculate and save statistics over SSIM
    header = 'Structural Similarity Indices for {0}\n'.format(model_name)
    header = 'Phase Mapping: {0}\n'.format(phase_mapping)
    header += 'N:     {0}\n'.format(ssim.shape[0])

    if complex_valued:
        header += 'Mean:  {0}/{1}\n'.format(np.mean(ssim[:, 0]), np.mean(ssim[:, 1]))
        header += 'STDEV: {0}/{1}\n'.format(np.std(ssim[:, 0]), np.std(ssim[:, 1]))
        header += 'MIN:   {0}/{1}, Record ({2}/{3}\n'.format(
            np.min(ssim[:, 0]), np.min(ssim[:, 1]),
            np.argmin(ssim[:, 0]), np.argmin(ssim[:, 1]))
        header += 'MAX:   {0}/{1}, Record ({2}/{3}\n\n'.format(
            np.max(ssim[:, 0]), np.max(ssim[:, 1]),
            np.argmax(ssim[:, 0]), np.argmax(ssim[:, 1]))
    else:
        header += 'SSIM Statistics :\n --------------\n'
        header += 'Mean:  {0}\n'.format(np.mean(ssim))
        header += 'STDEV: {0}\n'.format(np.std(ssim))
        header += 'MIN:   {0}, Record ({1})\n'.format(np.min(ssim), np.argmin(ssim))
        header += 'MAX:   {0}, Record ({1})\n\n'.format(np.max(ssim), np.argmax(ssim))
        header += 'MSE Statistics :\n --------------\n'
        header += 'Mean:  {0}\n'.format(np.mean(ms_err))
        header += 'STDEV: {0}\n'.format(np.std(ms_err))
        header += 'MIN:   {0}, Record ({1})\n'.format(np.min(ms_err), np.argmin(ms_err))
        header += 'MAX:   {0}, Record ({1})\n\n'.format(np.max(ms_err), np.argmax(ms_err))


    # add index to ssim
    indexed_ssim_mse = np.transpose(np.vstack((np.arange(ssim.shape[0]), ssim, ms_err)))
    # indexed_ssim_mse = np.array(indexed_ssim_mse, dtype=[("idx", int),  ("SSIM", float), ("MSE", float)])

    np.savetxt(mp_folder + 'stats.txt', indexed_ssim_mse, header=header, fmt="%i %10.5f %10.5f")
    np.savez(mp_folder + 'stats.npz', indexed_ssim_mse)
    SSIMPlotter.save_plot(model_name, ssim)
    return ssim

#data, label_r, label_i = DataLoader.load_testing(records=-1)
#prediction('unet_6_layers_1e-05_lr_3px_filter_32_convd_i_retrain_50_epoch_mse', data, label_i)
#ssim_r = prediction('dcgan_6_layers_0.001_lr_3px_filter_32_convd_r', data, label_r, weights_file='gen_4_epochs.h5')
#print(np.mean(ssim_r))
#ssim_r=prediction('unet_5_layers_0.0001_lr_4px_filter_32_convd_loss_msq_r', data, label_r)

# data, label = DataLoader.load_testing(records=64, separate=False)
# ssim = prediction('unet_6-3_mse_prelu-dual-test', data, label)

# data, label_r, label_i = DataLoader.load_testing(records=64)
# ssim = prediction('unet_6-3_mse_prelu-test_real', data, label_r)
# ssim = prediction('unet_6-3_mse_prelu-test_imag', data, label_i)
#
# data, label_mag, label_ph = DataLoader.load_testing(records=64, dataset='ds-lymphoma-magphase')
# for i in range(label_ph.shape[0]):
#     print("{2} Max: {0}, Min: {1}, Err:[3}".format(
#         np.max(label_ph[i]), np.min(label_ph[i]), i))

# ssim = prediction('unet_6-3_mse_prelu-test-magphase_magnitude', data, label_mag)
# ssim = prediction('unet_6-3_mse_prelu-test-magphase_phase', data, label_ph, save_err_img=True)


# data, label_mag, label_ph = DataLoader.load_testing(records=-1, dataset='ds-lymphoma-magphase')
# ssim = prediction('unet_6-3_mse_prelu-test-magphase_magnitude', data, label_mag)

# data, label_splitphase = DataLoader.load_testing(records=-1, separate = False,
#         dataset='ds-lymphoma-magphase-splitphase')
# ssim = prediction('unet_6-3_mse_prelu-split-phase-only', data, label_splitphase, phase_mapping=True)

