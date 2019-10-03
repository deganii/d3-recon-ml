from keras.callbacks import Callback

from src.data.loader import DataLoader
from src.visualization.fit_plotter import FitPlotter
from src.callbacks.model_callback import ModelCallback
import imageio
import numpy as np
import cv2
import PIL
from PIL import Image, ImageDraw, ImageFont
import skimage
from io import BytesIO, StringIO

class HoloNetCallback(ModelCallback):

    def __init__(self, model_name, experiment_id):
        super(HoloNetCallback, self).__init__(model_name, experiment_id)
        self.current_epoch = 0
        # self.test_data = test_data
        # self.test_labels = test_labels
        # self.fourcc = cv2.VideoWriter_fourcc(*'H264')
        self.fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.video = None
        self.video_frame = None
        self.video_draw = None
        self.fnt = None
        d_test_raw = DataLoader.load_testing(dataset='hangul_1', records=-1, separate=False)
        self.hangul_1_data, self.hangul_1_label = d_test_raw
        np.random.seed(0)

    def extract_model_weights(self):
        w = self.model.layers[1].get_weights()
        return np.squeeze(w[0])


    def np_to_pil_image(self, array):
        buf = BytesIO()
        imageio.imwrite(buf, array, format='PNG')
        return Image.open(buf)


    def get_model_weights_image(self):
        return self.np_to_pil_image(self.extract_model_weights())

    def save_model_weights_image(self, path):
        img_path = path + 'weights.png'
        imageio.imwrite(img_path, self.extract_model_weights())
        return img_path


    def update_common_items(self):
        model_weights_img = self.get_model_weights_image()
        filter = model_weights_img.resize((300, 300), Image.ANTIALIAS)
        self.video_frame.paste(filter, (515, 600))

        # test hangul_1 image pre-selected
        pred = self.model.predict(np.expand_dims(self.hangul_1_data[0], axis=0))
        label = self.hangul_1_label[0]

        holo = self.np_to_pil_image(np.squeeze(self.hangul_1_data[0])).resize((300, 300), Image.ANTIALIAS)
        pred = self.np_to_pil_image(np.squeeze(pred)).resize((300, 300), Image.ANTIALIAS)
        label = self.np_to_pil_image(np.squeeze(label)).resize((300, 300), Image.ANTIALIAS)

        self.video_frame.paste(holo, (60, 175))
        self.video_frame.paste(holo, (60, 600))

        self.video_frame.paste(pred, (890, 600))
        self.video_frame.paste(label, (890, 175))

        # put in a random training/test set from the current batch
        r_int = np.random.randint(0, 100)
        epoch_folder = self.get_last_epoch_folder()
        self.paste_into_frame(epoch_folder + 'images/{0:05}-input.png'
                              .format(r_int), (1300, 165))
        self.paste_into_frame(epoch_folder + 'images/{0:05}-pred.png'
                              .format(r_int), (1490, 165))
        self.paste_into_frame(epoch_folder + 'images/{0:05}-label.png'
                              .format(r_int), (1680, 165))

        self.video_draw.line((1193, 60, 1193, 1020), fill=(0, 0, 0), width=5)

    def on_batch_end(self, batch, logs=None):
        """ Gets gradient of model for given inputs and outputs for all weights"""
        if self.video_frame is not None and batch % 10 == 0:
            self.update_common_items()
            self.write_frame()

        # b_folder = self.get_current_batch_folder()



        # grads = model.optimizer.get_gradients(model.total_loss, model.trainable_weights)

        # symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
        # f = K.function(symb_inputs, grads)
        # x, y, sample_weight = model._standardize_user_data(inputs, outputs)
        # output_grad = f(x + y + sample_weight)
        # return output_grad

        # get stats for batch
        # save weights of model filter
        # save gradients for batch
        # predict on a couple of images from batch and
        # if len(self.model.history.history) > 0:
        #     FitPlotter.save_plot(
        #         self.model.history.history,
        #         '{0}/train_validation'.format(self.model_name))

    def on_train_end(self, logs=None):
        self.video.release()
        self.video = None

    def write_frame(self):
        bgr_img = cv2.cvtColor(np.array(self.video_frame), cv2.COLOR_RGB2BGR)
        self.video.write(bgr_img)

    def paste_into_frame(self, path, location, resize=None):
        image = Image.open(path)
        if resize is not None:
            image = image.resize(resize, Image.ANTIALIAS)
        self.video_frame.paste(image, location)

    def on_epoch_end(self, epoch, logs=None):
        if epoch > 0 and self.should_save():
            exp_folder = self.get_experiment_folder()
            if self.video is None:
                self.video = cv2.VideoWriter(
                    # e_folder+'weights_{0}.avi'.format(self.current_epoch),
                    exp_folder+'weights.avi',
                    self.fourcc, 30.0, (1920, 1080))#weights.shape)
            self.video_frame = Image.new('RGB', (1920, 1080), color=(255, 255, 255))
            self.video_draw = ImageDraw.Draw(self.video_frame)
            self.fnt = ImageFont.truetype("arial.ttf", 60)
            epoch_folder = self.get_current_epoch_folder()


            #
            # model_weights_img = self.get_model_weights_image()
            # self.save_model_weights_image(self.get_current_epoch_folder())
            #
            # epoch_folder = self.get_current_epoch_folder()

            # save an ssim plot on test set for this epoch
            # ssim_img = Image.open(epoch_folder+'ssim.png')
            # self.video_frame.paste(ssim_img, (1215, 725) )

            # save an epoch training plot on test set for this epoch
            self.paste_into_frame(epoch_folder+'ssim.png',
                                  (1225, 725), resize=(600,300))

            # save an epoch training plot on test set for this epoch
            self.paste_into_frame(epoch_folder+'train_validation.png',
                                  (1225, 400), resize=(600,300))

            # save the phyics-based filter
            self.paste_into_frame('F:/d3-recon-ml/-Gfp64.png',
                                  (515, 175), resize=(300,300))

            # filter = model_weights_img.resize((300, 300), Image.ANTIALIAS)
            # self.video_frame.paste(filter, (515, 600))

            # save the conv icon
            self.paste_into_frame('F:/d3-recon-ml/cnn.jpeg',(375, 700))

            # save the phyics-based filter
            self.paste_into_frame('F:/d3-recon-ml/conv.png',
                                  (370, 265), resize=(128,128))

            self.paste_into_frame('F:/d3-recon-ml/green-arrow.png',
                                  (815, 305), resize=(71,37))
            self.paste_into_frame('F:/d3-recon-ml/green-arrow.png',
                                  (815, 760), resize=(71,37))

            # # put some random training/test prediction in upper left
            # r_int = np.random.randint(0, 100)
            # self.paste_into_frame(epoch_folder + 'images/{0:05}-input.png'
            #                       .format(r_int), (1300, 165))
            # self.paste_into_frame(epoch_folder + 'images/{0:05}-label.png'
            #                       .format(r_int), (1685, 165))

            self.video_draw.text((1315, 50), "Training Epoch: {0}".format(epoch),
                                 font=self.fnt, fill=(0, 0, 0))

            self.video_draw.text((60,50), "Based on light diffraction physics",
                                 font=self.fnt, fill=(0, 0, 0))

            self.video_draw.text((60, 505), "Based on machine-learned physics",
                                 font=self.fnt, fill=(0, 0, 0))

            acc = self.model.history.history['val_loss'][-1]
            self.video_draw.text((60, 960), "Cumulative Error: {0:.2%}".format(acc),
                                 font=self.fnt, fill=(0, 0, 0))

            # self.video.write(skimage.img_as_ubyte(weights))
            # self.video.write(cv2.imread(img_path))
            self.update_common_items()


            self.write_frame()



