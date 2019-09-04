import numpy as np
import os
import matplotlib

from src.processing.folders import Folders

matplotlib.use('Agg')
import matplotlib.pyplot as plt

class FitPlotter(object):
    @classmethod
    def get_full_path(cls, file):
        return os.path.join(Folders.models_folder(), file)

    @classmethod
    def save_plot(cls, history, file, title='',
                  figsize=(4,2), y_title='Loss', linewidth=0.5):
        # plot and save to disk
        full_path = FitPlotter.get_full_path(file)

        fig = plt.figure(figsize=figsize)
        #title = "Learning Rate"
        fig.suptitle('', fontsize=10, fontweight='bold')
        ax = fig.add_subplot(111)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(y_title)

        epochs = len(history['loss'])
        epoch_dt = [i+1 for i in range(epochs)]

        ax.plot(epoch_dt, history['loss'],
                label='Train', linewidth=linewidth)
        ax.plot(epoch_dt, history['val_loss'],
                label='Validation', linewidth=linewidth)
        ax.legend()
        fig.subplots_adjust(left = 0.24)
        fig.subplots_adjust(bottom = 0.27)
        fig.subplots_adjust(right = 0.96)
        fig.subplots_adjust(top = 0.94)
        fig.canvas.draw()
        #plt.show()
        #plt.imsave(file)

        plt.savefig(full_path + '.png', format='png')
        plt.savefig(full_path + '.svg', format='svg')
        plt.close(fig)

# Test case
if __name__ == "__main__":
    root_dir = 'C:\\dev\\courses\\2.131 - Advanced Instrumentation\\data_lpf\\32x6-LSTM-46,146-Params\\'
    csv = np.genfromtxt(root_dir+'training.csv', delimiter=',')

    FitPlotter.save_plot({'loss':csv[:, 4], 'val_loss':csv[:, 2]},
                         root_dir + 'training.png', figsize=(4,2),
                         y_title='Loss %',linewidth=2.0)

    FitPlotter.save_plot({'loss':csv[:, 3], 'val_loss':csv[:, 1]},
                         root_dir + 'accuracy.png', figsize=(4,2),
                         y_title='Accuracy %', linewidth=2.0)

    #FitPlotter.save_plot({'loss':[0.237664829282,0.237664833081], 'val_loss':[0.236981078982, 0.236981078982]}, 'unet-test.png')