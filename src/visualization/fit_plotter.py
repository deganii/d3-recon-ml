import numpy as np
import os
import matplotlib.pyplot as plt

class FitPlotter(object):
    @classmethod
    def get_figures_folder(cls, ):
        figures_folder = '../../figures'
        if not os.path.exists(figures_folder):
                os.makedirs(figures_folder)
        return figures_folder

    def get_full_path(cls, file):
        return os.path.join(FitPlotter.get_figures_folder(), file)

    @classmethod
    def save_fig(cls, history, file, title=''):
        # plot and save to disk
        full_path = FitPlotter.get_full_path(file)

        fig = plt.figure(figsize=(4,2))
        #title = "Learning Rate"
        fig.suptitle('', fontsize=10, fontweight='bold')
        ax = fig.add_subplot(111)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')

        epochs = len(history.history['acc'])
        epoch_dt = [i for i in range(epochs)]

        ax.plot(epoch_dt, history.history['acc'], label='Train', linewidth=0.5)
        ax.plot(epoch_dt, history.history['val_acc'], label='Validation', linewidth=0.5)
        ax.legend()
        fig.subplots_adjust(left = 0.17)
        fig.subplots_adjust(bottom = 0.27)
        fig.subplots_adjust(right = 0.96)
        fig.subplots_adjust(top = 0.94)
        fig.canvas.draw()
        #plt.show()
        #plt.imsave(file)

        plt.savefig(file)
        plt.close(fig)