from keras.callbacks import Callback
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.visualization.fit_plotter import FitPlotter
from src.processing.predict import prediction
from src.processing.folders import Folders
import time
import svgutils.transform as sg


class SSIMPlotterCallback(Callback):

    def __init__(self, model_name, experiment_id, test_data, test_labels):
        super(SSIMPlotterCallback, self).__init__()
        self.model_name = model_name
        self.experiment_id = experiment_id
        self.test_data = test_data
        self.test_labels = test_labels

    def tileImages(self, images, n_columns=4, cropx=0, cropy=0):
        if isinstance(images[0], str):
            images = [Image.open(f) for f in images]

        # resize all images to the same size
        for i in range(len(images)):
            if images[i].size != images[0].size:
                images[i] = images[i].resize(images[0].size, resample=Image.BICUBIC)

        width, height = images[0].size
        width, height = width - 2 * cropx, height - 2 * cropy
        n_rows = int((len(images)) / n_columns)

        a_height = int(height * n_rows)
        a_width = int(width * n_columns)
        image = Image.new('L', (a_width, a_height), color=255)

        for row in range(n_rows):
            for col in range(n_columns):
                y0 = row * height - cropy
                x0 = col * width - cropx
                tile = images[row * n_columns + col]
                image.paste(tile, (x0, y0))

        # send back the tiled img
        return image

    def on_epoch_end(self, epoch, logs=None):
        if self.test_data is None or self.test_labels is None:
            return

        # run for first 5 epochs where results are dramatic
        # and then only do every 5th epoch
        if epoch < 5 or epoch % 5 == 0:
            mp_folder = Folders.experiments_folder() + \
                        '{0}/Epoch_{1:04}/'. format(
                            self.experiment_id, epoch)
            # save an ssim plot on test set for this epoch
            err_img = False
            ssim, tiled_imgs, best_imgs, worst_imgs = prediction(
                self.model_name, self.test_data,
                self.test_labels, transpose=False,
                model=self.model, mp_folder=mp_folder,
                save_n=100, zip_images=True, save_err_img=err_img)

            n_columns = 3 if not err_img else 4
            # make a coherent summary from the available information
            tiled_img = self.tileImages(tiled_imgs,n_columns=n_columns)
            best_img = self.tileImages(best_imgs, n_columns=n_columns)
            worst_img = self.tileImages(worst_imgs, n_columns=n_columns)

