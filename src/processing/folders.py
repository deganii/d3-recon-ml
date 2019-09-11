import os

from sys import platform



class Folders(object):

    @classmethod
    def get_base_folder(cls):
        # on local windows box (with AMD GPU) use keras plaidml backend...
        if platform == "win32":
            # keep data on separate drive
            return 'f:/d3-recon-ml/{0}/'
        return '../../{0}/'


    @classmethod
    def get_folder(cls, name):
        relative_path = Folders.get_base_folder().format(name)
        if not os.path.exists(relative_path):
            os.makedirs(relative_path)
        return relative_path

    @classmethod
    def models_folder(cls):
        return Folders.get_folder('models')

    @classmethod
    def data_folder(cls):
        return Folders.get_folder('data')

    @classmethod
    def figures_folder(cls):
        return Folders.get_folder('figures')

    @classmethod
    def predictions_folder(cls):
        return Folders.get_folder('predictions')

    @classmethod
    def experiments_folder(cls):
        return Folders.get_folder('experiments')

