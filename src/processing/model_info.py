import os

class ModelInfo(object):

    @classmethod
    def get_params(cls, model_name):
        # load the info summary



    @classmethod
    def get_name(cls, num_layers, learn_rate, filter_size, conv_depth):
        '''Generate a name for a model given params'''
        return 'unet_{0}_layers_{1}_lr_{2}px_filter_{3}_convd_r'.format(
            num_layers, learn_rate, filter_size, conv_depth)
