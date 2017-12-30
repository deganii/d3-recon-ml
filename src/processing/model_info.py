import os
import csv

class ModelInfo(object):

    @classmethod
    def get_params(cls, model_name):
        # load the info summary
        with open('mycsvfile.csv', 'wb') as f:
            w = csv.writer(f)
            w.writerows(somedict.items())

    def save_params(self, dic):
        models_folder = Folders.models_folder()


    @classmethod
    def get_name(cls, descriptive, num_layers):
        '''Generate a name for a model given params'''
        return 'unet_{0}_{1}_layers_r'.format(descriptive, num_layers)

