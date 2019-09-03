import warnings
import matplotlib
import numpy as np
import scipy.stats
import scipy.stats as st
from scipy.stats import norm
from scipy.stats import pearson3
from scipy.stats import skewnorm
from src.processing.folders import Folders
from tqdm import tqdm


matplotlib.use('Agg')
import matplotlib.pyplot as plt

IGNORE = ['levy_stable','ncf','nct']
DISTRIBUTIONS = [eval('scipy.stats.' + d) for d in dir(scipy.stats) if
    isinstance(getattr(scipy.stats, d), scipy.stats.rv_continuous) and d not in IGNORE]

class SSIMPlotter(object):
    @classmethod
    def get_best_dist(cls, data, nbins=50):
        best_distribution = scipy.stats.norm
        best_params = (0.0, 1.0)
        best_sse = np.inf
        y, x = np.histogram(data, bins=nbins, density=True)
        x = (x + np.roll(x, -1))[:-1] / 2.0

        distributions = tqdm(DISTRIBUTIONS)
        for distribution in distributions:
            distributions.set_description('{:10.10}'.format(distribution.name))
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                params = distribution.fit(data)

                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))

                if best_sse > sse > 0:
                    best_distribution = distribution
                    best_params = params
                    best_sse = sse
        return best_distribution, best_params

    @classmethod
    def save_plot(cls, model_name, ssim, fit_type='skew'):
        # plot and save to disk

        full_path = Folders.predictions_folder() + model_name + '-n{0}/ssim'.format(ssim.shape[0])
        svg_path = full_path + '.svg'
        png_path = full_path + '.png'
        fig = plt.figure(figsize=(4, 2))
        ax = fig.add_subplot(111)
        ax.set_xlabel('SSIM')
        ax.set_ylabel('Count(Prediction)')
        n, bins, patches = plt.hist(ssim, 50, density=1, facecolor='green', alpha=0.75)

        title = model_name + '\n'
        if fit_type == 'norm':
            # add a 'best fit' gaussian
            (mu, sigma) = norm.fit(ssim)
            y = norm.pdf(bins, mu, sigma)
            title += '$\mu={0:.2f},\ \sigma={1:.3f}$'.format(mu, sigma)
        elif fit_type == 'skew':
            skew_args = skewnorm.fit(ssim)
            (skew, mu, sigma) = skew_args
            y = skewnorm.pdf(bins, *skew_args[:-2], loc=skew_args[-2], scale=skew_args[-1])
            # y = skewnorm.pdf(bins, skew, mu, sigma)
            title += '$\mu={0:.2f},\ \sigma={1:.3f}, skew={2:.2f}$'.format(mu, sigma, skew)
        elif fit_type == 'pearson':
            pear_args = pearson3.fit(ssim)
            (skew, mu, sigma) = pear_args
            y = pearson3.pdf(bins, *pear_args[:-2], loc=pear_args[-2], scale=pear_args[-1])
            title += '$\mu={0:.2f},\ \sigma={1:.3f}, skew={2:.2f}$'.format(mu, sigma, skew)
        else: # best fit
            dist, dist_params = SSIMPlotter.get_best_dist(ssim)
            arg = dist_params[:-2]
            loc = dist_params[-2]
            scale = dist_params[-1]
            y = dist.pdf(bins, *arg, loc=dist_params[-2], scale=dist_params[-1])
            title += '{0}: $\mu={1:.2f},\ \sigma={2:.3f}$'.format(dist.name, loc, scale)

        plt.plot(bins, y, 'r--', linewidth=2)
        fig.suptitle(title, fontsize=10, fontweight='bold')
        fig.subplots_adjust(left=0.17)
        fig.subplots_adjust(bottom=0.27)
        fig.subplots_adjust(right=0.96)
        fig.subplots_adjust(top=0.82)

        fig.canvas.draw()
        plt.grid(True)
        plt.savefig(svg_path, format='svg')
        plt.savefig(png_path, format='png')
        plt.close(fig)


if __name__ == "__main__":
    # Test case
    model_name = 'unet_6-3_mse_mnist-3750'
    ssim_file = np.load(Folders.predictions_folder() + model_name + '-n750/stats.npz')
    ssim = ssim_file['arr_0'][:,1]
    SSIMPlotter.save_plot(model_name,ssim, fit_type='best')

