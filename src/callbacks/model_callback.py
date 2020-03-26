from keras.callbacks import Callback
from src.processing.folders import Folders
import os

class ModelCallback(Callback):
    # simple wrapper with convenience functions for callbacks

    def __init__(self, model_name, experiment_id,
                 period=5, save_every_epoch_until=2):
        super(Callback, self).__init__()
        self.model_name = model_name
        self.experiment_id = experiment_id
        self.current_epoch = 0
        self.current_batch = 0
        self.period = period
        self.save_every_epoch_until = save_every_epoch_until
        # TODO: REMOVE THIS HACK FOR VIDEO:
        # self.save_every_epoch_until = 75

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch

    def on_batch_begin(self, batch, logs=None):
        self.current_batch = batch

    def get_experiment_folder(self):
        exp_folder = Folders.experiments_folder() + self.experiment_id + '/'
        os.makedirs(exp_folder, exist_ok=True)
        return exp_folder

    def get_current_epoch_folder(self):
        ce_folder = Folders.experiments_folder() + \
                    '{0}/Epoch_{1:04}/'.format(
                        self.experiment_id, self.current_epoch)
        os.makedirs(ce_folder, exist_ok=True)
        return ce_folder

    def get_last_epoch_folder(self):
        ce_folder = Folders.experiments_folder() + \
                    '{0}/Epoch_{1:04}/'.format(
                        self.experiment_id, self.current_epoch-1)
        os.makedirs(ce_folder, exist_ok=True)
        return ce_folder

    def get_current_batch_folder(self):
        b_folder = Folders.experiments_folder() + \
                    '{0}/Epoch_{1:04}/Batches/Batch_{2:04}/'.format(
                        self.experiment_id, self.current_epoch,
                        self.current_batch)
        os.makedirs(b_folder, exist_ok=True)
        return b_folder


    def get_model_folder(self):
        return Folders.models_folder() + self.model_name + '/'


    def should_save(self):
        return self.current_epoch < self.save_every_epoch_until \
               or self.current_epoch % self.period == 0
