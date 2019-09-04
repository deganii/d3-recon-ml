
from keras.callbacks import Callback
from src.visualization.fit_plotter import FitPlotter
from src.processing.folders import Folders
import time

class TimeHistory(Callback):
    def __init__(self, model_name, experiment_id):
        super(TimeHistory, self).__init__()
        self.model_name = model_name
        self.experiment_id = experiment_id

    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append((epoch,time.time() - self.epoch_time_start))
        _, times = zip(*self.times)
        # write out the train times to a csv
        with open(Folders.models_folder() +
                  '{0}/performance.csv'.format(self.model_name), 'w') as f:
            f.write('Avg Time(s): {0:5.5f} \n'.format(sum(times)/len(times)))
            f.write('Epoch,Time(s)\n')
            for (ep, tim) in self.times:
                f.write('{0},{1:5.5f}\n'.format(ep, tim))
