import matplotlib
import matplotlib.mlab as mlab
import numpy as np
from scipy.stats import norm

from src.processing.folders import Folders

matplotlib.use('Agg')
import matplotlib.pyplot as plt


class SSIMPlotter(object):
    @classmethod
    def save_plot(cls, model_name, ssim):
        # plot and save to disk
        full_path = Folders.predictions_folder() + model_name + '-n{0}/ssim.png'.format(ssim.shape[0])

        fig = plt.figure(figsize=(4, 2))
        ax = fig.add_subplot(111)
        ax.set_xlabel('SSIM')
        ax.set_ylabel('Count(Prediction)')
        n, bins, patches = plt.hist(ssim, 50, normed=1, facecolor='green', alpha=0.75)

        # add a 'best fit' gaussian
        (mu, sigma) = norm.fit(ssim)
        y = mlab.normpdf(bins, mu, sigma)
        l = plt.plot(bins, y, 'r--', linewidth=2)

        fig.suptitle( model_name + '\n$\mu={0:.2f},\ \sigma={1:.2f}$'.format(mu, sigma),
                     fontsize=10, fontweight='bold')
        fig.subplots_adjust(left=0.17)
        fig.subplots_adjust(bottom=0.27)
        fig.subplots_adjust(right=0.96)
        fig.subplots_adjust(top=0.84)

        fig.canvas.draw()
        plt.grid(True)
        plt.savefig(full_path)
        plt.close(fig)

# Test case
# SSIMPlotter.save_plot('unet_3_layers_0.0001_lr_3px_filter_1_convd_i',
#                      np.asarray([0.5, 0.3, 0.4, 0.5, 0.5, 0.4, 0.8, 0.2, 0.3, 0.4, 0.4]))
