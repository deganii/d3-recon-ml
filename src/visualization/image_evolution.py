import matplotlib
import keras.models
import numpy as np

from src.data.loader import DataLoader
from src.processing.folders import Folders
matplotlib.use('Agg')
from PIL import Image
from keras import backend as K

class ImageEvolution(object):

    @classmethod
    def save_plot(cls, model_name, title=''):
        # load model
        model = keras.models.load_model(Folders.models_folder() + model_name + '.h5')

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
                img = Image.fromarray(np.transpose(np.uint8(255.0 * img_array / np.max(img_array))))
                imgs.append(img)
        ImageEvolution.saveTiledImages(imgs, model_name, n_columns=8)


    @classmethod
    def saveTiledImages(cls, images, model_name, n_columns=4, cropx=0, cropy = 0):
        if isinstance(images[0],str):
            images = [Image.open(f) for f in images]

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

        for row in range(n_rows):
            for col in range(n_columns):
                y0 = row * height - cropy
                x0 = col * width - cropx
                tile = images[row*n_columns+col]
                image.paste(tile, (x0,y0))
        full_path = Folders.figures_folder() + model_name + '_evolution.png'
        image.save(full_path)
        # send back the tiled img
        return image


# Test case
# ImageEvolution.save_plot('unet_3_layers_0.0001_lr_3px_filter_1_convd_r')

