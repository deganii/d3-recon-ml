import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))


import matplotlib
import keras.models
import numpy as np

from src.data.loader import DataLoader
from src.processing.folders import Folders
matplotlib.use('Agg')
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

from keras import backend as K

import scipy.misc

class ImageEvolution(object):

    @classmethod
    def format_and_return(cls, img_array, normalize=None):
        img_array = img_array.reshape([192, 192])
        if normalize is not None:
            # img_array = img_array + normalize
            img_array = img_array + np.min(img_array)

        img = Image.fromarray(np.uint8(255.0 * img_array / np.max(img_array)))
        return img

    @classmethod
    def save_epoch_evolution(cls, model_name, data, labels,
                             epochs = 18, n_columns = 6, period = 1):
        print("data.shape: {0}".format(data.shape))
        epoch_ev = np.zeros((epochs//period, data.shape[0], data.shape[1], data.shape[2]))
        for i in range(epochs//period):
            # load model
            epoch = (i+1)*period
            print('Epoch ' + str(epoch))
            model = keras.models.load_model(Folders.models_folder() + model_name + '/weights_{0:02d}.h5'.format(epoch), compile=False)
            # just predict on the selected index
            predictions = model.predict(data, batch_size=32, verbose=0)
            del model
            K.clear_session()

            predictions=predictions.astype(np.float64)
            predictions=predictions.reshape([data.shape[0],data.shape[1], data.shape[2]])
            #pred = ImageEvolution.format_and_return(predictions[idx])
            #epoch_ev.append(predictions)
            epoch_ev[i] = predictions

        # save down the predictions
        # preds_file = Folders.data_folder() + 'epoch_ev-{0}-{1}-epochs.npz'.format(model_name, epochs)
        # np.savez(preds_file, epoch_ev=epoch_ev)


        data = data.reshape([data.shape[0], data.shape[1], data.shape[2]])
        labels = labels.reshape([labels.shape[0], labels.shape[1], labels.shape[2]])
        for idx in range(data.shape[0]):
            img = ImageEvolution.format_and_return(data[idx])
            label = ImageEvolution.format_and_return(labels[idx])

            ImageEvolution.saveTiledImages(epoch_ev[:, idx, ...], '{0:05}_'.format(idx) + model_name, n_columns=n_columns)
            scipy.misc.imsave(Folders.figures_folder() + '{0:05}_'.format(idx) + model_name + '_evolution_data.png', img)
            scipy.misc.imsave(Folders.figures_folder() + '{0:05}_'.format(idx) + model_name + '_evolution_label.png', label)

    @classmethod
    def save_plot(cls, model_name, title=''):
        # load model
        model = keras.models.load_model(Folders.models_folder() + model_name + '/weights.h5')

        inp = model.input
        outputs = [layer.output for layer in model.layers]  # all layer outputs
        functors = [K.function([inp] + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions

        data, real, imag = DataLoader.load_training(records=64)
        data = data[np.newaxis, 0, ...]
        layer_outs = [func([data, 1.]) for func in functors]

        imgs = []
        for lo in layer_outs:
            for i in range(lo[0].shape[3]):
                img_array = lo[0][0, ..., i]
                #img_array = img_raw.reshape(img_raw.shape[1], img_raw.shape[2])
                img_min = np.min(img_array)
                if img_min < 0:
                    img_array = img_array + img_min
                # if transpose:
                #     img = Image.fromarray(np.transpose(np.uint8(255.0 * img_array / np.max(img_array))))
                # else:
                img = Image.fromarray(np.uint8(255.0 * img_array / np.max(img_array)))
                imgs.append(img)
        ImageEvolution.saveTiledImages(imgs, model_name, n_columns=8,cropx=10, cropy = 10)


    @classmethod
    def saveTiledImages(cls, image_data, model_name, n_columns=4, cropx=0, cropy = 0):
        print("image_data.shape: {0}".format(image_data.shape))

        images = []
        for i in range(image_data.shape[0]):
            images.append(ImageEvolution.format_and_return(image_data[i]))
        # font = ImageFont.truetype("sans-serif.ttf", 16)
        # font = ImageFont.load_default()

        # if isinstance(images[0],str):
        #     images = [Image.open(f) for f in images]

        # resize all images to the same size
        for i in range(len(images)):
            if images[i].size != images[0].size:
                images[i] = images[i].resize( images[0].size, resample=Image.BICUBIC)

        width, height = images[0].size
        width, height = width - 2*cropx, height - 2*cropy
        n_rows = int((len(images))/n_columns)

        a_height = int(height * n_rows)
        a_width = int(width * n_columns)
        image = Image.new('L', (a_width, a_height), color=255)
        draw = ImageDraw.Draw(image)
        for row in range(n_rows):
            for col in range(n_columns):
                y0 = row * height - cropy
                x0 = col * width - cropx
                tile_idx = row*n_columns+col
                tile = images[tile_idx]
                image.paste(tile, (x0,y0))
                # draw.text((x0+width - 10, y0 + height-10), tile_idx, (255, 0, 0))
        full_path = Folders.figures_folder() + model_name + '_evolution.png'
        image.save(full_path)
        # send back the tiled img
        return image


# Test case
# ImageEvolution.save_plot('unet_6_layers_1e-05_lr_3px_filter_32_convd_r_retrain_100_epoch_mse')

# data, label_r, label_i = DataLoader.load_testing(records=64)
# ImageEvolution.save_epoch_evolution('unet_6_layers_0.0001_lr_3px_filter_32_convd_loss_msq_r',
#     data, label_r, idx=3, epochs=6, n_columns=3)
#
# prediction('unet_6_layers_1e-05_lr_3px_filter_32_convd_i_retrain_50_epoch_mse', data, label_i)
# ssim_r = prediction('dcgan_6_layers_0.001_lr_3px_filter_32_convd_r', data, label_r, weights_file='gen_4_epochs.h5')
# print(np.mean(ssim_r))


# data, label = DataLoader.load_testing(dataset='nucleus', records=-1)
# ImageEvolution.save_epoch_evolution('unet_6-3_mse_nucleus-all-epochs',
#     data, label, epochs=25, n_columns=5)
#     # data[np.newaxis,-1,...], label[np.newaxis,-1,...], epochs=25, n_columns=5)

data, label = DataLoader.load_testing(dataset='test-031418-bcdapi', records=100)
ImageEvolution.save_epoch_evolution('unet_6-3_mse_test_031418_100epochs_3',
    data, label, epochs=100, n_columns=5, period = 5)
    # data[np.newaxis,-1,...], label[np.newaxis,-1,...], epochs=25, n_columns=5)
