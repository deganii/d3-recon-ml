from keras.callbacks import Callback
from src.visualization.fit_plotter import FitPlotter
from src.processing.folders import Folders
from src.callbacks.model_callback import ModelCallback
import os

class FitPlotterCallback(ModelCallback):

    def __init__(self, model_name, experiment_id):
        super(FitPlotterCallback, self).__init__(model_name, experiment_id)

    def on_epoch_end(self, epoch, logs=None):
        if self.should_save() and len(self.model.history.history) > 0:
            model_path = self.get_model_folder() + 'train_validation'
            # experiment_path = os.path.join(Folders.experiments_folder(),
            #                                '{0}/train_validation'.format(self.experiment_id))
            FitPlotter.save_plot(
                self.model.history.history, model_path)
            # FitPlotter.save_plot(
            #     self.model.history.history, experiment_path)

