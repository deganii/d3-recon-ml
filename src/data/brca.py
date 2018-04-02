# generator of labeled breast cancer test images
import os, glob
import h5py
import numpy as np
import scipy
from PIL import Image
from scipy import misc

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.processing.folders import Folders


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cmocean
import cmocean.cm

import scipy
import scipy.io
import scipy.misc
import scipy.ndimage

class BrCaGenerator(object):
    @classmethod
    def partitionTrainingAndTestSet(cls, set_name='ds-brca'):
        data_folder = '../../data/'
        data_npz = np.load(data_folder + set_name + '-all.npz')
        all_data, all_labels = data_npz['data'], data_npz['labels']
        np.random.seed(0)
        indices = np.random.permutation(all_data.shape[0])
        test_count = int(np.floor(all_data.shape[0] * 0.2))
        test_idx, training_idx = indices[:test_count], indices[test_count:]
        training_data, test_data = all_data[training_idx, :], all_data[test_idx, :]
        training_labels, test_labels = all_labels[training_idx, :, :], all_labels[test_idx, :, :]
        np.savez(os.path.join(data_folder, set_name + '-training.npz'), data=training_data, labels=training_labels)
        np.savez(os.path.join(data_folder, set_name + '-test.npz'), data=test_data, labels=test_labels)

    @classmethod
    def generateMagPhaseDataset(cls, set_name='ds-brca',  suffix = '-n64'):
        data_folder = Folders.data_folder()
        for partition in ['training', 'test']:
            npz = np.load(data_folder + set_name + '-{0}{1}.npz'.format(partition, suffix))
            # create data and labels with the same shape
            labels = npz['labels']
            ri = labels[:   ,0,...] + 1j * labels[:,1,...]
            mag, phase = np.abs(ri), np.angle(ri)
            mag_phase_labels = np.stack((mag, phase), 1)
            # save down
            np.savez(data_folder + set_name + '-{0}-{1}{2}.npz'.
                     format( 'magphase', partition, suffix),
                     data=npz['data'], labels=mag_phase_labels)

    @classmethod
    def generateSplitPhaseDataset(cls, set_name='ds-brca-magphase',  suffix = '-n64'):
        data_folder = Folders.data_folder()
        for partition in ['training', 'test']:
            npz = np.load(data_folder + set_name + '-{0}{1}.npz'.format(partition, suffix))
            # create data and labels with the same shape
            holo = npz['data']
            mag = npz['labels'][:, 0, ...]
            phase = npz['labels'][:,1,...]
            phase_x, phase_y = np.cos(phase), np.sin(phase)
            phase_labels = np.stack((phase_x, phase_y), 1)

            # save down images
            split_name = data_folder + set_name + '-{0}-{1}{2}'.\
                format( 'splitphase', partition, suffix)
            if not os.path.exists(split_name):
                os.makedirs(split_name)
            for i in range(phase.shape[0]):
                holoDestFilename = '{0:05}-H.png'.format(i)
                magnDestFilename = '{0:05}-M.png'.format(i)
                phasDestFilename = '{0:05}-P.png'.format(i)

                # save hologram and magnitude
                scipy.misc.imsave(os.path.join(split_name, holoDestFilename), np.squeeze(holo[i]))
                scipy.misc.imsave(os.path.join(split_name, magnDestFilename), np.squeeze(mag[i]))
                # save phase
                plt.imsave(os.path.join(split_name, phasDestFilename), np.squeeze(phase[i]),
                           cmap=cmocean.cm.phase, vmin=-np.pi, vmax=np.pi)

            # save down npz
            np.savez(split_name + '.npz', data=npz['data'], labels=phase_labels)





    @classmethod
    def generateImages(cls, set_name, stride=100, tile=(192,192),
                      input_folder='../../data/BrCa/120813_SKBR3-7umBeads_Profiling/Reconstruction/',
                      output_folder='../../data/',
                      save_npz=True):
        data = None
        labels = None
        seq = 0

        image_folder = os.path.join(output_folder, set_name)
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)

        input_files = glob.glob(input_folder + 'SKBR3_HER2_05.mat')
        num_input_files = len(input_files)
        for input_file in input_files:
            # open brca image
            sample_data = scipy.io.loadmat(input_file)
            #sample_data = h5py.File(input_file, 'r')
            input_title = os.path.splitext(os.path.basename(input_file))[0]

            # load a sample image
            subNormAmp = sample_data['NormAmp']
            irng = sample_data['Range'][0]

            # scale and crop subNormAmp
            subNormAmp = subNormAmp[irng[2]:irng[3],irng[0]:irng[1]]
            subNormAmp = scipy.ndimage.zoom(subNormAmp, 4, order=3)
            reconImage = sample_data['ReconImage']
            reconImage = reconImage[:-1,:-1]
            #reshaped = [[reconImage[m, n] for m in range(reconImage.shape[0])] for n in range(reconImage.shape[1])]
            # convert to numpy array

            subNormAmp = subNormAmp.astype(np.float32)
            viewHologram = Image.fromarray(np.transpose(np.uint8(255.0 * subNormAmp / np.max(subNormAmp))))

            # scipy.misc.imsave('./test_viewhologram.png', viewHologram)

            # convert from matlab tuples to a proper np array
            #recon = np.asarray([[[num for num in row] for row in rows] for rows in reconImage])
            reconReal = np.real(reconImage)
            reconImag = np.imag(reconImage)

            viewReconReal = reconReal + np.abs(np.min(reconReal))
            viewReconImag = reconImag + np.abs(np.min(reconImag))

            viewReconReal = Image.fromarray(np.transpose(np.uint8(255.0 * (viewReconReal) / np.max(viewReconReal))))
            viewReconImag = Image.fromarray(np.transpose(np.uint8(255.0 * (viewReconImag) / np.max(viewReconImag))))

            # scipy.misc.imsave('./test_viewreconReal.png', viewReconReal)
            # scipy.misc.imsave('./test_viewreconImag.png', viewReconImag)

            M = subNormAmp.shape[0]
            N = subNormAmp.shape[1]

            # tile the image
            last_M = int(M - (M % stride))
            last_N = int(N - (N % stride))

            M_count = int(np.floor(M/stride))
            N_count = int(np.floor(N/stride))

            if data is None:
                data = np.zeros((num_input_files * 4 * M_count * N_count, tile[0] , tile[1]))
                labels = np.zeros((num_input_files * 4 * M_count * N_count, 2, tile[0] , tile[1]))
                # print("data shape: ")
                # print(data.shape)
                # print('\n')
                # print("labels shape: ")
                # print(labels.shape)


            for rot in range(0, 360, 90):
                for m in range(0, last_M, stride):
                    for n in range(0, last_N, stride):

                        st_m, end_m, st_n, end_n = m, tile[0] + m,  n, tile[1] + n

                        if end_m >= M:
                            st_m, end_m = M - 1 - tile[0], M - 1
                        if end_n >= N:
                            st_n, end_n = N - 1 - tile[1], N - 1

                        crop_mn = [st_m, st_n, end_m, end_n]

                        holoTile = viewHologram.crop(crop_mn)
                        realTile = viewReconReal.crop(crop_mn)
                        imageTile = viewReconImag.crop(crop_mn)

                        holoTile = holoTile.rotate(rot, resample=Image.BICUBIC)
                        realTile = realTile.rotate(rot, resample=Image.BICUBIC)
                        imageTile = imageTile.rotate(rot, resample=Image.BICUBIC)

                        transformation = "tm_{0}_tn_{1}_rot_{2}".format(st_m, st_n, rot)

                        holoDestFilename = '{0:05}-H-{1}-{2}.png'.format(seq,transformation,input_title)
                        realDestFilename = '{0:05}-R-{1}-{2}.png'.format(seq,transformation,input_title)
                        imagDestFilename = '{0:05}-I-{1}-{2}.png'.format(seq,transformation,input_title)

                        scipy.misc.imsave(os.path.join(image_folder, holoDestFilename), holoTile)
                        scipy.misc.imsave(os.path.join(image_folder, realDestFilename), realTile)
                        scipy.misc.imsave(os.path.join(image_folder, imagDestFilename), imageTile)

                        # append the raw data to the
                        data[seq, :] = np.rot90(subNormAmp[st_m:end_m, st_n:end_n], int(rot / 90))#.reshape(tile[0] , tile[1], 1)
                        labels[seq, 0, :] = np.rot90(reconReal[st_m:end_m, st_n:end_n], int(rot / 90))#.reshape(tile[0] , tile[1], 1)
                        labels[seq, 1, :] = np.rot90(reconImag[st_m:end_m, st_n:end_n], int(rot / 90))#.reshape(tile[0] , tile[1])
                        seq = seq + 1
        data = data[..., np.newaxis]
        labels = labels[..., np.newaxis]
        if save_npz:
            np.savez(os.path.join(output_folder, set_name + '-all.npz'), data=data, labels=labels)



BrCaGenerator.generateImages('ds-brca-512', tile=(512,512))
#BrCaGenerator.partitionTrainingAndTestSet('ds-brca')
# BrCaGenerator.generateMegPhaseDataset(suffix='')
# BrCaGenerator.generateSplitPhaseDataset(suffix='')
# BrCaGenerator.generateSplitPhaseDataset(suffix='')
