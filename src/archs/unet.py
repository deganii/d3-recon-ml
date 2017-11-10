import numpy as np
import scipy
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import ModelCheckpoint
import os
from src.visualization.fit_plotter import FitPlotter
from PIL import Image

K.set_image_data_format('channels_last')
data_folder = '../../data/'
data_training = np.load(data_folder + 'ds-lymphoma-training.npz')
data_testing=np.load(data_folder + 'ds-lymphoma-test.npz')
train_data, train_labels = data_training['data'], data_training['labels']
test_data,test_labels=data_testing['data'], data_testing['labels']
print(test_data.shape)

img_rows, img_cols = train_data.shape[1], train_data.shape[2]

def get_unet(num_layers = 4, filter_size=3):
    inputs = Input((img_rows, img_cols, 1))
    last_in= inputs
    conv_dict = {}
    for i in range(num_layers):
        conv = Conv2D(32*2**i, (filter_size, filter_size), activation='relu', padding='same')(last_in)
        conv = Conv2D(32*2**i, (filter_size, filter_size), activation='relu', padding='same')(conv)
        conv_dict[i] = conv
        if i < num_layers:
            pool = MaxPooling2D(pool_size=(2, 2))(conv)
            last_in = pool

    last_in = conv_dict[i]
    for i in range(num_layers-1, 0, -1):
        up = concatenate([Conv2DTranspose(32*2**i, (2, 2), strides=(2, 2), padding='same')(last_in),
                          conv_dict[i-1]], axis=3)
        conv = Conv2D(32*2**i, (3, 3), activation='relu', padding='same')(up)
        last_in = Conv2D(32*2**i, (3, 3), activation='relu', padding='same')(conv)

    conv_last = Conv2D(1, (1, 1), activation='relu', padding='same')(last_in)
    model = Model(inputs=[inputs], outputs=[conv_last])

    model.compile(optimizer=Adam(lr=1e-4), loss='mean_squared_error', metrics=['accuracy'])

    return model

modelr = get_unet(6,3)
modeli = get_unet()

models_folder = '../../models/'
if not os.path.exists(models_folder):
    os.makedirs(models_folder)
modelr_checkpoint = ModelCheckpoint(models_folder + 'weights_R.h5', monitor='val_loss', save_best_only=True)
#modeli_checkpoint = ModelCheckpoint('weights_I.h5', monitor='val_loss', save_best_only=True)

print('-'*30)
print('Fitting model...')
print('-'*30)

#resize labels
label_real_tr=train_labels[:,0,:]
label_imaginary_tr=train_labels[:,1,:]
label_real_tst=test_labels[:,0,:]
label_imaginary_tst=test_labels[:,1,:]

#shorten training set
train_data_s=train_data[0:64,...]
train_labelR_s=label_real_tr[0:64,...]
train_labelI_s=label_imaginary_tr[0:64,...]

modelr.summary()

def prediction(model,TestData):
    image_folder = '../../predictions'
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    predict=model.predict(TestData, batch_size=32, verbose=0, steps=None)
    for i in range(predict.shape[0]):
        predict_shape = predict[i].reshape([192, 192])
        testdata_shape=TestData[i].reshape([192,192])
        label_real_shape=label_real_tst[i].reshape([192,192])

        min = np.abs(np.min(predict_shape))
        viewPrediction = Image.fromarray(np.transpose(np.uint8(255.0 * (predict_shape+min)/ (np.max(predict_shape)+min))))
        scipy.misc.imsave('../../predictions/{0:05}-test_viewprediction.png'.format(i),viewPrediction)

        viewTestData = Image.fromarray(np.transpose(np.uint8(255.0 * (testdata_shape) / np.max(testdata_shape))))
        scipy.misc.imsave('../../predictions/{0:05}-test_viewtestdata.png'.format(i),viewTestData)

        min=np.abs(np.min(label_real_shape))
        viewLabelReal = Image.fromarray(np.transpose(np.uint8(255.0 * (label_real_shape+min)/ (np.max(label_real_shape)+min))))
        scipy.misc.imsave('../../predictions/{0:05}-test_viewlabelreal.png'.format(i),viewLabelReal)
    return()

#history = modelr.fit(train_data, label_real_tr, batch_size=32, epochs=2, verbose=1, shuffle=True,
#         validation_split=0.2, callbacks=[modelr_checkpoint])

modelr.load_weights('../../models/weights_R.h5')

# plot and save:
#FitPlotter.save_fig(history, 'unet-6.jpg')

prediction(modelr,test_data)



#modeli.fit(train_data, label_imaginary, batch_size=32, nb_epoch=10, verbose=1, shuffle=True,
          #validation_split=0.2, callbacks=[modeli_checkpoint])

