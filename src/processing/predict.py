import keras.models
import keras.losses
import scipy.misc
import numpy as np
import skimage.measure
import os
from keras_contrib.losses import DSSIMObjective
keras.losses.dssim = DSSIMObjective()
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

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


# from keras.utils.generic_utils import get_custom_objects
#
# loss = DSSIMObjective()
# get_custom_objects().update({"dssim": loss})


from src.data.loader import DataLoader
from src.processing.folders import Folders
from PIL import Image
from src.visualization.ssim_plotter import SSIMPlotter

def format_and_save(img_array, output_file, dmin=None, dmax=None, transpose=True):
    img_array = img_array.reshape([192, 192])
    if dmin is not None and dmax is not None:
        img_array = (img_array + dmin) / (dmax + dmin)
    else:
        img_array = img_array / np.max(img_array)
    if transpose:
        img = Image.fromarray(np.transpose(np.uint8(255.0 * img_array)))
    else:
        img = Image.fromarray(np.uint8(255.0 * img_array))
    #print('Norm: {0}, Max: {1}\n'.format(dmin, dmax))
    scipy.misc.imsave(output_file, img)
    return img

def format_and_save_phase(img_array, output_file):
    plt.imsave(output_file, img_array, cmap=cmocean.cm.phase,  vmin=-np.pi, vmax=np.pi)
    return Image.open(output_file)

def update_saved_lists(i, img, best, best_list, worst, worst_list, tiled_imgs, tiled_list):
    if i in tiled_list:
        tiled_imgs.append(img)
    if i == best:
        best_list.append(img)
    if i == worst:
        worst_list.append(img)

def prediction(model_name, data, labels, save_err_img = False,
               phase_mapping=False, weights_file='weights.h5',
               transpose=True, model=None, mp_folder=None,
               save_n=-1, zip_images=False, tiled_list=None):

    if model is None:
        model = keras.models.load_model(Folders.models_folder() + model_name + '/' + weights_file)
    if mp_folder is None:
        mp_folder = Folders.predictions_folder() + model_name + '-n{0}/'.format(data.shape[0])
    mp_images_folder = mp_folder + 'images/'

    os.makedirs(mp_folder, exist_ok=True)
    os.makedirs(mp_images_folder, exist_ok=True)

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

    if save_n < 0 or save_n > predictions.shape[0]:
        save_n = predictions.shape[0]

    best, worst = np.argmax(ssim), np.argmin(ssim)
    save_list = list(range(save_n))
    save_list.append(best)
    save_list.append(worst)

    if tiled_list is None:
        tiled_list = [0, 1, 2]
    tiled_imgs = []
    best_imgs = []
    worst_imgs = []

    for i in save_list:
        file_prefix = mp_images_folder + '{0:05}-'.format(i)
        input_img = format_and_save(data[i], file_prefix + 'input.png', transpose=transpose)
        update_saved_lists(i,input_img,best,best_imgs,worst,worst_imgs,tiled_imgs,tiled_list)


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

                pred = format_and_save(predictions[i, ..., j],
                    file_prefix + 'pred-{0}.png'.format(name),
                    dmin, dmax, transpose=transpose)
                label = format_and_save(labels[i, ..., j],
                    file_prefix + 'label-{0}.png'.format(name),
                    dmin, dmax, transpose=transpose)

        else:
            ssim[i] = skimage.measure.compare_ssim(predictions[i], labels[i])

            sq_err = np.square(predictions[i] - labels[i])

            dmin = np.abs(min(np.min(predictions[i]), np.min(labels[i])))
            dmax = np.abs(max(np.max(predictions[i]), np.max(labels[i])))
            pred=format_and_save(predictions[i], file_prefix + 'pred.png',
                dmin, dmax, transpose = transpose)
            label=format_and_save(labels[i], file_prefix + 'label.png',
                dmin, dmax, transpose = transpose)
            if phase_mapping:
                # mean phase error accounting for loop-over
                ms_err[i] = np.mean(np.abs(np.exp(1j * predictions[i]) -  np.exp(1j * labels[i])))
                pred = format_and_save_phase(predictions[i], file_prefix + 'pred.png')
                label = format_and_save_phase(labels[i], file_prefix + 'label.png')
            else:
                ms_err[i] = np.mean(sq_err)
                pred = format_and_save(predictions[i], file_prefix + 'pred.png',
                    dmin, dmax, transpose=transpose)
                label = format_and_save(labels[i], file_prefix + 'label.png',
                    dmin, dmax, transpose=transpose)

            if save_err_img:
                smin, smax = np.min(sq_err), np.max(sq_err)
                err = format_and_save(sq_err, file_prefix + 'err.png',
                    smin, smax, transpose=transpose)
                update_saved_lists(i, err, best, best_imgs, worst, worst_imgs, tiled_imgs, tiled_list)

        update_saved_lists(i,pred,best,best_imgs,worst,worst_imgs,tiled_imgs,tiled_list)
        update_saved_lists(i, label, best, best_imgs, worst, worst_imgs, tiled_imgs, tiled_list)

    if zip_images:
        with zipfile.ZipFile(mp_folder+'images.zip', "w", zipfile.ZIP_DEFLATED) as zipf:
            for fp in glob.glob(os.path.join(mp_images_folder, "**/*")):
                base = os.path.commonpath([mp_images_folder, fp])
                zipf.write(fp, arcname=fp.replace(base, ""))
        # os.rmdir(mp_images_folder)


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
    SSIMPlotter.save_plot(model_name, ssim, mp_folder=mp_folder, fit_type='best')
    return ssim, tiled_imgs, best_imgs, worst_imgs


cached_windows = dict()

def create_naive_window(window_size):
    return np.ones(window_size, window_size)

def create_linear_window(window_size,):

    return np.ones(window_size, window_size)


def create_spline_window(window_size, power=2):
    intersection = int(window_size/4)
    wind_outer = (abs(2*(scipy.signal.triang(window_size))) ** power)/2
    wind_outer[intersection:-intersection] = 0

    wind_inner = 1 - (abs(2*(scipy.signal.triang(window_size) - 1)) ** power)/2
    wind_inner[:intersection] = 0
    wind_inner[-intersection:] = 0

    wind = wind_inner + wind_outer
    wind = wind / np.average(wind)
    return wind

def spline_window_2d(window_size, power=2):
    global cached_windows
    key = "{}_{}".format(window_size, power)
    if key in cached_windows:
        window = cached_windows[key]
    else:
        window = create_spline_window(window_size, power)
        window = np.expand_dims(np.expand_dims(window, 3), 3)
        window = window * window.transpose(1, 0, 2)
        window = window.reshape(window.shape[0], window.shape[1])
        cached_windows[key] = window
    return window


def pad_image(img, window_size, overlap):
    aug = int(round(window_size * (1 - 1.0 * overlap)))
    more_borders = ((aug, aug), (aug, aug))
    ret = np.pad(img, pad_width=more_borders, mode='reflect')
    return ret


def unpad_image(padded_img, window_size, overlap):
    aug = int(round(window_size * (1 - 1.0 * overlap)))
    ret = padded_img[aug:-aug, aug:-aug]
    return ret


def rotate_and_mirror(im):
    rot_mirror = [np.array(im)]
    for rot in range(3):
        rot_mirror.append(np.rot90(np.array(im), axes=(0, 1), k=1+rot))
    im = np.array(im)[:, ::-1]
    rot_mirror.append(np.array(im))
    for rot in range(3):
        rot_mirror.append(np.rot90(np.array(im), axes=(0, 1), k=1+rot))
    return rot_mirror


def derotate_and_mirror(im_mirrs):
    origs = [np.array(im_mirrs[0])]
    for rot in range(3):
        origs.append(np.rot90(np.array(im_mirrs[1+rot]), axes=(0, 1), k=3-rot))
    origs.append(np.array(im_mirrs[4])[:, ::-1])
    for rot in range(3):
        origs.append(np.rot90(np.array(im_mirrs[5+rot]), axes=(0, 1), k=3-rot)[:, ::-1])
    return np.mean(origs, axis=0)


def create_windowed_patches(model, padded_img, window_size, overlap, batch_size=32):
    window = spline_window_2d(window_size=window_size, power=2)
    step = int(window_size * overlap)
    padx_len = padded_img.shape[0]
    pady_len = padded_img.shape[1]
    patches = []
    for i in range(0, padx_len-window_size+1, step):
        patches.append([])
        for j in range(0, pady_len-window_size+1, step):
            patch = padded_img[i:i+window_size, j:j+window_size]
            patches[-1].append(patch)
    patches = np.array(patches)
    npatches_x, npatches_y, patch_w, patch_h = patches.shape
    # Unet model expects (ntiles, tile_w, tile_h, tile_ch)
    patches = patches.reshape(npatches_x * npatches_y, patch_w, patch_h, 1)
    patches = model.predict(patches, batch_size=batch_size, verbose=0)
    patches = patches.astype(np.float64)
    # back to npatches_x, npatches_y, patch_w, patch_h
    patches = patches.reshape([npatches_x,npatches_y, patch_w, patch_h])
    windowed_patches = np.array([patch * window for patch in patches])
    gc.collect()
    return windowed_patches, patches


def merge_patches(patches, window_size, overlap, padded_out_shape):
    step = int(window_size * overlap)
    padx_len = padded_out_shape[0]
    pady_len = padded_out_shape[1]
    y = np.zeros(padded_out_shape)

    a = 0
    for i in range(0, padx_len-window_size+1, step):
        b = 0
        for j in range(0, pady_len-window_size+1, step):
            windowed_patch = patches[a, b]
            y[i:i+window_size, j:j+window_size] = \
                y[i:i+window_size, j:j+window_size] + windowed_patch
            b += 1
        a += 1
    # normalize (not strictly necessary)
    return y / ((1.0/overlap) ** 2)


def debug_save_images(output_folder, input_name, w_patches, patches, img_pass=-1):
    debug_dir = os.path.join(output_folder, 'debug')
    os.makedirs(debug_dir, exist_ok=True)
    # for idx, w_patch, patch in enumerate(zip(w_patches, patches)):
    for i in range(w_patches.shape[0]):
        for j in range(w_patches.shape[1]):
            scipy.misc.imsave(os.path.join(debug_dir,
                "{0}_patch_{1}-{2}_w{3}.png".format(input_name, i, j,
                    img_pass if img_pass > 0 else '')), w_patches[i, j])
            scipy.misc.imsave(os.path.join(debug_dir,
                "{0}_patch_{1}-{2}{3}.png".format(input_name, i, j,
                    img_pass if img_pass > 0 else '')), patches[i, j])
            w_patches[i, j,0] = 1.0
            w_patches[i, j, 191] = 1.0
            w_patches[i, j, :,0] = 1.0
            w_patches[i, j, :, 191] = 1.0


def prediction_with_merge(model_name, input_folder, output_folder, window_size=192,
                   overlap=0.5, weights_file='weights.h5', batch_size=32,
                   high_quality=True, debug_images=False):
    model = keras.models.load_model(Folders.models_folder() + model_name + '/' + weights_file)

    # assume input files are 12-bit tiff format...
    input_files = glob.glob(input_folder + '*.tif')
    os.makedirs(output_folder, exist_ok=True)

    for input_file in tqdm(input_files, desc='Overall Progress'):
        # assume input files are 12-bit tiff format...
        input_img = TIFF.open(input_file).read_image() / 4095.
        padded_img = pad_image(input_img, window_size, overlap)
        input_name = os.path.splitext(os.path.basename(input_file))[0]
        if high_quality:
            # predict multiple times on rotated/flipped tiles and average them
            padded_combos = rotate_and_mirror(padded_img)
            additional_predictions = []
            for img_pass, padded_img in enumerate(tqdm(padded_combos, desc='High-Quality Pass')):
                w_patches, patches = create_windowed_patches(model, padded_img, window_size, overlap, batch_size=batch_size)
                if debug_images:
                    debug_save_images(output_folder, input_name, w_patches, patches, img_pass=img_pass)
                additional_predictions.append(merge_patches(patches, window_size,
                                                            overlap, padded_out_shape=padded_img.shape))
            padded_results = derotate_and_mirror(additional_predictions)
        else:
            w_patches, patches = create_windowed_patches(model, padded_img, window_size, overlap, batch_size=batch_size)
            if debug_images:
                debug_save_images(output_folder, input_name, w_patches, patches)
            padded_results = merge_patches(patches, window_size, overlap, padded_out_shape=padded_img.shape)
        prd = unpad_image(padded_results, window_size, overlap)
        prd = prd[:input_img.shape[0], :input_img.shape[1]]

        # write out the prediction
        scipy.misc.imsave(os.path.join(output_folder, input_name +".pred.png"), prd)
    return w_patches, padded_results





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

# data, label_text = DataLoader.load_testing(records=-1, separate = False,
#             dataset='ds-text')
# ssim = prediction('unet_6-3_mse_text', data, label_text, transpose=False)
