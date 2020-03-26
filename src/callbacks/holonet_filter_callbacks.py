from keras.callbacks import Callback

from src.data.loader import DataLoader
from src.visualization.fit_plotter import FitPlotter
from src.callbacks.model_callback import ModelCallback
from keras.layers import Conv2D
import imageio
import numpy as np
import cv2
import PIL
from PIL import Image, ImageDraw, ImageFont
import skimage
from io import BytesIO, StringIO

class HoloNetFilterCallback(ModelCallback):

    def __init__(self, model_name, experiment_id, dataset_name='hangul_1', period=5):
        super(HoloNetFilterCallback, self).__init__(model_name, experiment_id, period=period)
        self.current_epoch = 0

    def extract_model_weights(self):
        w = []
        names = []
        for layer in self.model.layers:
            if isinstance(layer, Conv2D):
                lw = layer.get_weights()[0]
                for idx in range(lw.shape[2]):
                    w.append(lw[:, :, idx])
                    names.append('{0}.{1}'.format(layer.name, idx))
        return names, np.concatenate(w, axis=2)

        # w = self.model.layers[1].get_weights()
        # for phase holonet - will break orihinal holoety654
        # w2 = self.model.layers[3].get_weights()
        # w3 = self.model.layers[4].get_weights()
        # return np.concatenate([w[0][:,:,0], w2[0][:,:,0], w3[0][:,:,0]], axis=2)
        #return np.squeeze(w[0])

    def np_to_pil_image(self, array):
        buf = BytesIO()
        imageio.imwrite(buf, array, format='PNG')
        return Image.open(buf)

    def get_model_weights_image(self):
        return self.np_to_pil_image(self.extract_model_weights()[1])

    def get_model_weights_images(self):
        names, weights = self.extract_model_weights()
        images = []
        for filter_idx in range(weights.shape[-1]):
            images.append(self.np_to_pil_image(weights[..., filter_idx]))
        return names, images

    def save_model_weights_image(self, path):
        img_path = path + 'weights.png'
        imageio.imwrite(img_path, self.extract_model_weights())
        return img_path

    def on_epoch_end(self, epoch, logs=None):
        if self.should_save() and epoch > 0:
            epoch_folder = self.get_current_epoch_folder()
            names, images = self.get_model_weights_images()
            for idx, image in enumerate(images):
                imageio.imwrite(epoch_folder + 'filter_{0}_{1}.png'.format(idx, names[idx]), image)



