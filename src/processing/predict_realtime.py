import keras.models
import keras.losses
import imageio
import numpy as np
import skimage.measure
import os
from keras_contrib.losses import DSSIMObjective
keras.losses.dssim = DSSIMObjective()
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from timeit import default_timer as timer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cmocean
import cmocean.cm

import glob
import zipfile
import numpy as np
import scipy.signal
from tqdm import tqdm

from libtiff import TIFF

import gc
import imageio

from src.data.loader import DataLoader
from src.processing.folders import Folders
from PIL import Image
from src.visualization.ssim_plotter import SSIMPlotter

#realtime prediction stats
def prediction_realtime(model_name, data, labels, save_err_img = False,
               prediction_name=None, phase_mapping=False,
               weights_file='weights.h5', long_description=None,
               transpose=True, model=None, mp_folder=None,
               save_n=-1, zip_images=False, tiled_list=None):

    if model is None:
        model = keras.models.load_model(Folders.models_folder() + model_name + '/' + weights_file)
    if mp_folder is None:
        if prediction_name is None:
            prediction_name = model_name + '-n{0}/'.format(data.shape[0])
        mp_folder = Folders.predictions_folder() + prediction_name
    mp_images_folder = mp_folder + 'images/'

    os.makedirs(mp_folder, exist_ok=True)
    os.makedirs(mp_images_folder, exist_ok=True)

    if long_description is not None:
        with open(mp_folder+"desc.txt", "w") as text_file:
            text_file.write(long_description)

    avg_fps = 0.0
    total_time = 0.0
    for i in data.shape[0]:
        start = timer()
        model.predict(data[np.newaxis, i, ...], batch_size=1, verbose=0)
        end = timer()
        total_time += end - start
    print('Number Frames: {0}'.format(data.shape[0]))
    print('Total Time (s): {0}'.format(total_time))
    print('Average FPS: {0}'.format(data.shape[0]/total_time))


    # check if the network predicts the complex valued image or just one component
    #complex_valued = predictions.shape[-1] > 1

    # we are color-mapping a phase pre
    # if phase_mapping and complex_valued:
    #     predictions_complex = predictions[..., 1]  + 1j * predictions[..., 2]
    #     predictions = np.angle(predictions_complex)
    #     labels_complex = labels[..., 1] + 1j * labels[..., 2]
    #     labels = np.angle(labels_complex)
    #
    # # update for phase flattening
    # complex_valued = predictions.shape[-1] > 1
    #
    # if save_n < 0 or save_n > predictions.shape[0]:
    #     save_n = predictions.shape[0]
    #
    # if complex_valued:
    #     ssim = np.empty([predictions.shape[0], predictions.shape[-1]])
    #     ms_err = np.empty([predictions.shape[0], predictions.shape[-1]])
    # else:
    #     predictions=predictions.reshape([data.shape[0],data.shape[1], data.shape[2]])
    #     labels=labels.reshape([labels.shape[0],labels.shape[1],labels.shape[2]])
    #     ssim = np.empty([predictions.shape[0]])
    #     ms_err = np.empty([predictions.shape[0]])
    #
    #
    # best, worst = np.argmax(ssim), np.argmin(ssim)
    # save_list = list(range(save_n))
    # if save_n < best or save_n < worst:
    #     save_list.append(best)
    #     save_list.append(worst)
    #
    # if tiled_list is None:
    #     tiled_list = [0, 1, 2]
    # tiled_imgs = []
    # best_imgs = []
    # worst_imgs = []
    #
    # for i in range(predictions.shape[0]):
    #     file_prefix = mp_images_folder + '{0:05}-'.format(i)
    #
    #     if i < save_n or i == best or i == worst:
    #         input_img = format_and_save(data[i], file_prefix + 'input.png', transpose=transpose)
    #         update_saved_lists(i,input_img,best,best_imgs,worst,worst_imgs,tiled_imgs,tiled_list)
    #
    #
    #     # calculate the structural similarity index (SSIM) between prediction and source
    #     if complex_valued:
    #         for j,name in enumerate(['mag', 'real', 'imag']):
    #             ssim[i, j] = skimage.measure.compare_ssim(
    #                 predictions[i, ..., j], labels[i, ..., j])
    #             ms_err[i, j] = np.mean(np.square(predictions[i, ..., j] - labels[i, ..., j]))
    #
    #             # todo: check for min > 0 (don't need to add)
    #             # todo: check for max <= 0?
    #             dmin = np.abs(min(0, np.min(predictions[i, ..., j]),
    #                                    np.min(labels[i, ..., j])))
    #             dmax = max(np.max(predictions[i, ..., j]),
    #                     np.max(labels[i, ..., j]))
    #             if i < save_n or i == best or i == worst:
    #                 pred = format_and_save(predictions[i, ..., j],
    #                     file_prefix + 'pred-{0}.png'.format(name),
    #                     dmin, dmax, transpose=transpose)
    #                 label = format_and_save(labels[i, ..., j],
    #                     file_prefix + 'label-{0}.png'.format(name),
    #                     dmin, dmax, transpose=transpose)
    #
    #     else:
    #         ssim[i] = skimage.measure.compare_ssim(predictions[i], labels[i])
    #
    #         sq_err = np.square(predictions[i] - labels[i])
    #
    #         dmin = np.abs(min(np.min(predictions[i]), np.min(labels[i])))
    #         dmax = np.abs(max(np.max(predictions[i]), np.max(labels[i])))
    #         if i < save_n or i == best or i == worst:
    #             pred=format_and_save(predictions[i], file_prefix + 'pred.png',
    #                 dmin, dmax, transpose = transpose)
    #             label=format_and_save(labels[i], file_prefix + 'label.png',
    #                 dmin, dmax, transpose = transpose)
    #         if phase_mapping:
    #             # mean phase error accounting for loop-over
    #             ms_err[i] = np.mean(np.abs(np.exp(1j * predictions[i]) -  np.exp(1j * labels[i])))
    #             if i < save_n or i == best or i == worst:
    #                 pred = format_and_save_phase(predictions[i], file_prefix + 'pred.png')
    #                 label = format_and_save_phase(labels[i], file_prefix + 'label.png')
    #         else:
    #             ms_err[i] = np.mean(sq_err)
    #             if i < save_n or i == best or i == worst:
    #                 pred = format_and_save(predictions[i], file_prefix + 'pred.png',
    #                     dmin, dmax, transpose=transpose)
    #                 label = format_and_save(labels[i], file_prefix + 'label.png',
    #                     dmin, dmax, transpose=transpose)
    #
    #         if save_err_img:
    #             smin, smax = np.min(sq_err), np.max(sq_err)
    #             if i < save_n or i == best or i == worst:
    #                 err = format_and_save(sq_err, file_prefix + 'err.png',
    #                     smin, smax, transpose=transpose)
    #                 update_saved_lists(i, err, best, best_imgs, worst, worst_imgs, tiled_imgs, tiled_list)
    #
    #     if i < save_n or i == best or i == worst:
    #         update_saved_lists(i,pred,best,best_imgs,worst,worst_imgs,tiled_imgs,tiled_list)
    #         update_saved_lists(i, label, best, best_imgs, worst, worst_imgs, tiled_imgs, tiled_list)
    #
    # if zip_images:
    #     with zipfile.ZipFile(mp_folder+'images.zip', "w", zipfile.ZIP_DEFLATED) as zipf:
    #         for fp in glob.glob(os.path.join(mp_images_folder, "**/*")):
    #             base = os.path.commonpath([mp_images_folder, fp])
    #             zipf.write(fp, arcname=fp.replace(base, ""))
    #     # os.rmdir(mp_images_folder)
    #
    #
    # # calculate and save statistics over SSIM
    # header = 'Structural Similarity Indices for {0}\n'.format(model_name)
    # header += 'Phase Mapping: {0}\n'.format(phase_mapping)
    # header += 'N:     {0}\n'.format(ssim.shape[0])
    #
    # if complex_valued:
    #     header += 'Mean:  {0}/{1}\n'.format(np.mean(ssim[:, 0]), np.mean(ssim[:, 1]))
    #     header += 'STDEV: {0}/{1}\n'.format(np.std(ssim[:, 0]), np.std(ssim[:, 1]))
    #     header += 'MIN:   {0}/{1}, Record ({2}/{3}\n'.format(
    #         np.min(ssim[:, 0]), np.min(ssim[:, 1]),
    #         np.argmin(ssim[:, 0]), np.argmin(ssim[:, 1]))
    #     header += 'MAX:   {0}/{1}, Record ({2}/{3}\n\n'.format(
    #         np.max(ssim[:, 0]), np.max(ssim[:, 1]),
    #         np.argmax(ssim[:, 0]), np.argmax(ssim[:, 1]))
    # else:
    #     header += 'SSIM Statistics :\n --------------\n'
    #     header += 'Mean:  {0}\n'.format(np.mean(ssim))
    #     header += 'STDEV: {0}\n'.format(np.std(ssim))
    #     header += 'MIN:   {0}, Record ({1})\n'.format(np.min(ssim), np.argmin(ssim))
    #     header += 'MAX:   {0}, Record ({1})\n\n'.format(np.max(ssim), np.argmax(ssim))
    #     header += 'MSE Statistics :\n --------------\n'
    #     header += 'Mean:  {0}\n'.format(np.mean(ms_err))
    #     header += 'STDEV: {0}\n'.format(np.std(ms_err))
    #     header += 'MIN:   {0}, Record ({1})\n'.format(np.min(ms_err), np.argmin(ms_err))
    #     header += 'MAX:   {0}, Record ({1})\n\n'.format(np.max(ms_err), np.argmax(ms_err))
    #
    #
    # # add index to ssim
    # if complex_valued:
    #     indexed_ssim_mse = np.concatenate((ssim, ms_err), axis=1)
    #     np.savetxt(mp_folder + 'stats.txt', indexed_ssim_mse, header=header, fmt="%10.5f %10.5f %10.5f %10.5f %10.5f %10.5f")
    #     # indexed_ssim_mse = np.concatenate((np.arange(ssim.shape[0]), indexed_ssim_mse))
    # else:
    #     indexed_ssim_mse = np.transpose(np.vstack((np.arange(ssim.shape[0]), ssim, ms_err)))
    #     np.savetxt(mp_folder + 'stats.txt', indexed_ssim_mse, header=header, fmt="%i %10.5f %10.5f")
    # # indexed_ssim_mse = np.array(indexed_ssim_mse, dtype=[("idx", int),  ("SSIM", float), ("MSE", float)])
    #
    #
    # np.savez(mp_folder + 'stats.npz', indexed_ssim_mse)
    # svg_path = SSIMPlotter.save_plot(model_name, ssim[0, :], mp_folder=mp_folder, fit_type='skew')
    # return ssim, svg_path, tiled_imgs, best_imgs, worst_imgs