import numpy as np
import skimage.measure

class ImageComparator(object):

    @classmethod
    def ssim_metric(cls, y_true, y_pred):
        ssim_sum = 0
        for i in range(y_true.shape[0]):
            ssim_sum += skimage.measure.compare_ssim(y_true[i], y_pred[i])
        return ssim_sum / y.shape[0]

    @classmethod
    def mutual_information(cls, hgram):
         """ Mutual information for joint histogram
         """
         # Convert bins counts to probability values
         pxy = hgram / float(np.sum(hgram))
         px = np.sum(pxy, axis=1) # marginal for x over y
         py = np.sum(pxy, axis=0) # marginal for y over x
         px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
         # Now we can do the calculation using the pxy, px_py 2D arrays
         nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
         return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

