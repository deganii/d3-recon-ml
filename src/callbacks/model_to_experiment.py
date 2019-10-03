
from keras.callbacks import Callback
from src.processing.folders import Folders
import  glob, os
from shutil import copyfile
from src.callbacks.model_callback import ModelCallback
class ModelToExperiment(ModelCallback):

    def __init__(self, model_name, experiment_id):
        super(ModelToExperiment, self).__init__(model_name, experiment_id)

    def on_epoch_end(self, epoch, logs=None):
        if self.should_save():
            # copy all model data to the active experiment
            model_path = self.get_model_folder()
            exp_path = self.get_current_epoch_folder()

            for file in glob.glob(model_path + '*', recursive=True):
                _, ext = os.path.splitext(file)
                if ext is not 'h5':
                    copyfile(file,
                             os.path.join(exp_path,os.path.basename(file)))
