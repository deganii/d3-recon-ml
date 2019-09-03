import os

class Folders(object):

    @classmethod
    def get_base_folder(cls):
        # return '../../{0}/'
        # keep data on separate drive
        return 'f:/d3-recon-ml/{0}/'

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
    def visualizations_folder(cls):
        return Folders.get_folder('visualizations')

