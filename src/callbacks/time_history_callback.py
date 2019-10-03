
from keras.callbacks import Callback
from src.visualization.fit_plotter import FitPlotter
from src.processing.folders import Folders
import time
from src.callbacks.model_callback import ModelCallback
class TimeHistory(ModelCallback):
    def __init__(self, model_name, experiment_id):
        super(TimeHistory, self).__init__(model_name, experiment_id)
        self.epoch_time_start = 0

    def on_train_begin(self, logs={}):
        ModelCallback.on_train_begin(self, logs=logs)
        self.times = []

    def on_epoch_begin(self, epoch, logs=None):
        ModelCallback.on_epoch_begin(self, epoch, logs=logs)
        self.epoch_time_start = time.time()

    def write_csv(self, path, times):
        with open(path, 'w') as f:
            f.write('Avg Time(s): {0:5.5f} \n'.format(sum(times)/len(times)))
            f.write('Epoch,Time(s)\n')
            for (ep, tim) in self.times:
                f.write('{0},{1:5.5f}\n'.format(ep, tim))

    def on_epoch_end(self, epoch, logs={}):
        if self.should_save():
            self.times.append((epoch,time.time() - self.epoch_time_start))
            _, times = zip(*self.times)
            # write out the train times to a csv
            model_path = self.get_model_folder() + 'performance.csv'
            # exp_path = self.get_current_experiment_folder(epoch) + 'performance.csv'
            self.write_csv(model_path, times)
            # self.write_csv(exp_path, times)
