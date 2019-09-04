
from keras.callbacks import Callback
from src.processing.folders import Folders
import  glob, os
from shutil import copyfile

class ModelToExperiment(Callback):
    def __init__(self, model_name, experiment_id):
        super(ModelToExperiment, self).__init__()
        self.model_name = model_name
        self.experiment_id = experiment_id

    def on_epoch_end(self, epoch, logs={}):
        # copy all model data to the active experiment
        model_path = Folders.models_folder() + self.model_name
        exp_path = Folders.experiments_folder() + self.experiment_id

        for file in glob.glob(model_path + '/**/*'):
            filename, ext = os.path.splitext(file)
            if(ext is not 'h5'):
                copyfile(filename, os.path.join(exp_path,os.path.basename(filename)))
