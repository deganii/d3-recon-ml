import matplotlib
import matplotlib.mlab as mlab
import numpy as np
from scipy.stats import norm
from scipy.stats import pearson3
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
        pear_args = pearson3.fit(ssim)
        (skew, mu, sigma) = pear_args

        y_pear = pearson3.pdf(bins, *pear_args[:-2], loc=pear_args[-2], scale=pear_args[-1])
        y = mlab.normpdf(bins, mu, sigma)
        #l = plt.plot(bins, y, 'r--', linewidth=2)
        l2 = plt.plot(bins, y_pear, 'r--', linewidth=2)

        fig.suptitle( model_name + '\n$\mu={0:.2f},\ \sigma={1:.2f}, skew={2:.2f}$'.format(mu, sigma, skew),
                     fontsize=10, fontweight='bold')
        fig.subplots_adjust(left=0.17)
        fig.subplots_adjust(bottom=0.27)
        fig.subplots_adjust(right=0.96)
        fig.subplots_adjust(top=0.82)

        fig.canvas.draw()
        plt.grid(True)
        plt.savefig(full_path)
        plt.close(fig)

# Test case
model_name = 'unet_6-3_mse_prelu-test-magphase_magnitude'
ssim_file = np.load(Folders.predictions_folder() + model_name + '-n1404/stats.npz')
ssim = ssim_file['arr_0'][:,1]
SSIMPlotter.save_plot(model_name,ssim)
