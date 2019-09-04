from keras.callbacks import Callback
from src.visualization.fit_plotter import FitPlotter


class FitPlotterCallback(Callback):

    def __init__(self, model_name):
        super(FitPlotterCallback, self).__init__()
        self.model_name = model_name

    def on_epoch_end(self, epoch, logs=None):
        if len(self.model.history.history) > 0:
            FitPlotter.save_plot(
                self.model.history.history,
                '{0}/train_validation'.format(self.model_name))
