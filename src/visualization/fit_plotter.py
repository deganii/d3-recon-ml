import PIL

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
    def save_plot(cls, history, path,
                  figsize=(4,2), y_title='Cumulative Error',
                  linewidth=0.5, font_size=10,
                  dpi=1200, return_pil_img=False):
        # plot and save to disk
        # full_path = FitPlotter.get_full_path(file)

        fig = plt.figure(figsize=figsize)
        #title = "Learning Rate"
        fig.suptitle('', fontsize=font_size, fontweight='bold')
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
        fig.subplots_adjust(left=0.24)
        fig.subplots_adjust(bottom=0.27)
        fig.subplots_adjust(right=0.96)
        fig.subplots_adjust(top=0.94)
        fig.canvas.draw()
        #plt.show()
        #plt.imsave(file)

        if return_pil_img:
            canvas = plt.get_current_fig_manager().canvas
            img = PIL.Image.fromstring('RGB', canvas.get_width_height(),
                 canvas.tostring_rgb())
            plt.close(fig)
            return img
        else:
            plt.savefig(path + '.png', format='png', dpi=dpi)
            plt.savefig(path + '.svg', format='svg')
            plt.close(fig)



# Test case (rebuilding from perflog)
if __name__ == "__main__":
    root_dir = Folders.models_folder() + 'holo_net_64_1/'
    csv = np.genfromtxt(root_dir+'perflog.csv', delimiter=',')
    FitPlotter.save_plot({'loss':csv[:, 2], 'val_loss':csv[:, 4]},
                         root_dir + 'train_validation_big.png', figsize=(4,2),
                         y_title='Loss %',linewidth=0.8, dpi=1200)

    #FitPlotter.save_plot({'loss':[0.237664829282,0.237664833081], 'val_loss':[0.236981078982, 0.236981078982]}, 'unet-test.png')