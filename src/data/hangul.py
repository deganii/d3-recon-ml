
import os
import numpy as np
import imageio
import glob
from src.processing.folders import Folders
from src.data.diffraction import DiffractionGenerator
from src.data.common import CommonGenerator
import scipy

class HangulGenerator(object):
    @classmethod
    def generate_hangul_dataset(cls, raw_hgu1_path, set_name='hangul'):
        image_folder = Folders.data_folder() + set_name + '/'
        os.makedirs(image_folder, exist_ok=True)

        hgu1_files = glob.glob(raw_hgu1_path + '*.hgu1')
        data, labels, idx  = [], [], 0
        Gfp = None
        for f in hgu1_files:
            hgu1_arr, origin_id = HangulGenerator.read_hgu1(f, max_images=5)
            hgu1_arr = hgu1_arr.astype(np.float16) / 255.
            holo_arr = np.empty_like(hgu1_arr)

            for t_idx, img in enumerate(hgu1_arr):
                if Gfp is None:
                    Gfp = DiffractionGenerator.\
                        freeSpaceTransfer(img, z=2.5e-3,
                                          lmbda=405e-9, upsample=2)
                holo = np.abs(DiffractionGenerator.freeSpacePropagation(
                    img, z=2.5e-3, lmbda = 405e-9, upsample=2, Gfp=Gfp))
                holo_arr[t_idx] = scipy.ndimage.zoom(holo, 0.251, order=3)

                holoDestFilename = '{0:05}-H-{1}.{2}.png'.format(idx,origin_id, t_idx)
                labelDestFilename = '{0:05}-L-{1}.{2}.png'.format(idx,origin_id, t_idx)

                # save hologram and magnitude
                imageio.imwrite(image_folder +  holoDestFilename, np.squeeze(holo_arr[t_idx]))
                imageio.imwrite(image_folder +  labelDestFilename, np.squeeze(hgu1_arr[t_idx]))
                if idx % 100 == 0:
                    print("train: {0}\n".format(t_idx))
                idx += 1
            data.append(hgu1_arr)
            labels.append(hgu1_arr)

        np.savez(os.path.join(Folders.data_folder(), set_name + '-all.npz'),
                 data=np.vstack(data), labels=np.vstack(labels))
        CommonGenerator.partitionTrainingAndTestSet(set_name=set_name)


        # return ndarray corresponding to HGU1 file
    @classmethod
    def read_hgu1(cls, path, save_path=None, debug=False,
                  img_pattern='{0:05}-{1}.png', idx=0, pad=(192, 192),
                  max_images=np.inf):
        size_bytes = os.stat(path).st_size
        origin_id = os.path.splitext(os.path.basename(path))[0]

        image_arrays = []
        with open(path, "rb") as f:
            # read the HGU1 header
            header = f.read(8).decode("ascii")
            if not header.startswith('HGU1'):
                raise ValueError()

            while f.tell() < size_bytes and len(image_arrays) < max_images:
                m_code0 = int.from_bytes(f.read(1), byteorder='big')
                m_code1 = int.from_bytes(f.read(1), byteorder='big')
                # width / height have max 255
                width = int.from_bytes(f.read(1), byteorder='big')
                height = int.from_bytes(f.read(1), byteorder='big')
                m_type = int.from_bytes(f.read(1), byteorder='big')
                m_reserved = int.from_bytes(f.read(1), byteorder='big')
                img_array = np.fromfile(f, dtype=np.uint8, count=width*height)
                img_array = img_array.reshape((height, width))
                h_pad, v_pad = (pad[0] - img_array.shape[0], pad[1] - img_array.shape[1])
                pad_left, pad_top = h_pad // 2, v_pad // 2
                pad_right, pad_bottom = h_pad - pad_left, v_pad - pad_top
                img_array = np.pad(img_array,[(pad_left, pad_right),(pad_top, pad_bottom)],
                                   mode='constant', constant_values=255)
                if save_path is not None:
                    imageio.imwrite(save_path + img_pattern.format(idx, origin_id), img_array)
                if debug:
                    print('Source: {0}, Index: {1}, Size: ({2},{3}), ' \
                          'm_code: [{4},{5}], type={6}, reserved={7}\n'.format(
                            origin_id, idx, width, height, m_code0,
                            m_code1, m_type, m_reserved))

                image_arrays.append(img_array)
                idx += 1
        return np.stack(image_arrays), origin_id


if __name__ == "__main__":
    # h_array = HangulGenerator.read_hgu1('F:\\d3-recon-ml\\data\\HangulDB-master\\HanDB_train\\b0a1.hgu1',
    #                         'F:\\d3-recon-ml\\data\\HangulDB-master\\extract_test\\')
    # print(h_array.shape)
    # HangulGenerator.generate_hangul_dataset("F:/d3-recon-ml/data/HangulDB-master/tiny_test/", set_name='hangul_tiny')
    HangulGenerator.generate_hangul_dataset("F:/d3-recon-ml/data/HangulDB-master/HanDB_train/", set_name='hangul_5')
