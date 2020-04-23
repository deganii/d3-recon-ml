import os
import sys
if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")

from src.data.diffraction import DiffractionGenerator

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


# input = scipy.misc.imread('../../data/ds-simulated/00017-label-sc.png').astype("float32")
# input = input / np.max(input)
# output = DiffractionGenerator.freeSpacePropagation(input, upsample=2, pad_width=0, z=5e-4, delta=2.2e-6 )
# mag = np.abs(output)
# norm_mag = 255.0 * (mag / np.max(mag))
# imageio.imwrite('../../data/ds-simulated/00017-prop.png', norm_mag)

#DiffractionGenerator.generateNewImagePair(".", 1)
#DiffractionGenerator.diffractDS1Dataset()

# test a single diffraction simulation
# holo, recon = DiffractionGenerator.generateSample()
# holo = np.abs(holo)
# holo = holo / np.max(holo)
# holo = np.reshape(holo, [1,192, 192, 1])
# # holo = np.zeros([1,192,192,1])
# model = keras.models.load_model(Folders.models_folder() +
#             'unet_6-3_mse_prelu-test-magphase_magnitude/weights.h5')
# predictions = model.predict(holo, batch_size=1, verbose=0)
# predictions = np.reshape(predictions, [192,192])
# imageio.imwrite('pred.png', predictions)


# DiffractionGenerator.generateCenteredTextSample()
DiffractionGenerator.generateTextDataset()

#LymphomaGenerator.generateImages('ds-lymphoma')
#LymphomaGenerator.partitionTrainingAndTestSet('ds-lymphoma')
# LymphomaGenerator.generateMegPhaseDataset(suffix='')
# LymphomaGenerator.generateSplitPhaseDataset(suffix='')
# LymphomaGenerator.generateSplitPhaseDataset(suffix='')
