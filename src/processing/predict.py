import keras.models
import scipy.misc
import numpy as np
from src.processing.folders import Folders
from PIL import Image


def format_and_save(img_array, output_file, normalize=False):
    img_array = img_array.reshape([192, 192])
    if normalize:
        img_array = img_array + np.abs(np.min(img_array))
    img = Image.fromarray(np.transpose(np.uint8(255.0 * img_array / np.max(img_array))))
    scipy.misc.imsave(output_file, img)


def prediction(model_name, data, labels):
    model = keras.models.load_model(Folders.models_folder() + model_name)
    mp_folder = Folders.predictions_folder() + model_name + '/'
    predictions = model.predict(data, batch_size=32, verbose=0, steps=None)
    for i in range(predictions.shape[0]):
        file_prefix = mp_folder + '{0:05}-'.format(i)
        format_and_save(data[i], file_prefix + 'input.png', False)
        format_and_save(predictions[i], file_prefix + 'pred.png', True)
        format_and_save(labels[i], file_prefix + 'label.png', True)

