import matplotlib
import keras.models

from src.processing.folders import Folders
matplotlib.use('Agg')
from PIL import Image
from keras import backend as K

class ImageEvolution(object):

    @classmethod
    def save_plot(cls, model_name, title=''):
        # load model
        model = keras.models.load_model(Folders.models_folder() + model_name)

        # plot and save to disk
        full_path = Folders.figures_folder() + model_name + '_evolution.png'

        inp = model.input
        outputs = [layer.output for layer in model.layers]  # all layer outputs
        functors = [K.function([inp] + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions

        # Testing
        # test = np.random.random(input_shape)[np.newaxis, ...]
        # layer_outs = [func([test, 1.]) for func in functors]
        # layer_outs


    @classmethod
    def saveTiledImages(cls, images, model_name, n_columns=4, cropx=0, cropy = 0):
        if isinstance(images[0],str):
            images = [Image.open(f) for f in images]

        # resize all images to the same size
        for i in len(1, images):
            if(images[i].size != images[0].size):
                images[i] = images[i].resize( images[0].size, resample = Image.BICUBIC)

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
        # image.save(destination)
        # send back the tiled img
        return image


# Test case
# ImageEvolution.save_plot('unet-test.png')

