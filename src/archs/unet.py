import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import ModelCheckpoint

K.set_image_data_format('channels_last')
data_folder = '../../data/'
data_training = np.load(data_folder + 'ds-lymphoma-training.npz')
data_testing=np.load(data_folder + 'ds-lymphoma-test.npz')
train_data, train_labels = data_training['data'], data_training['labels']
img_rows, img_cols = train_data.shape[1], train_data.shape[2]

def get_unet(num_layers = 4):
    inputs = Input((img_rows, img_cols, 1))
    last_in= inputs
    conv_dict = {}
    for i in range(num_layers):
        conv = Conv2D(32*2**i, (3, 3), activation='relu', padding='same')(last_in)
        conv = Conv2D(32*2**i, (3, 3), activation='relu', padding='same')(conv)
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

    model.compile(optimizer=Adam(lr=1e-5), loss='mean_squared_error', metrics=['accuracy'])

    return model

modelr = get_unet(4)
modeli = get_unet()
modelr_checkpoint = ModelCheckpoint('weights_R.h5', monitor='val_loss', save_best_only=True)
#modeli_checkpoint = ModelCheckpoint('weights_I.h5', monitor='val_loss', save_best_only=True)

print('-'*30)
print('Fitting model...')
print('-'*30)

#testing resize function.  Cant get to work for individual line items
label_real=train_labels[:,0,:]
label_imaginary=train_labels[:,1,:]
train_data_s=train_data[0:500,...]
train_labelR_s=label_real[0:500,...]
train_labelI_s=label_imaginary[0:500,...]

modelr.summary()

modelr.fit(train_data_s, train_labelR_s, batch_size=32, nb_epoch=10, verbose=1, shuffle=True,
          validation_split=0.2, callbacks=[modelr_checkpoint])

#modeli.fit(train_data, label_imaginary, batch_size=32, nb_epoch=10, verbose=1, shuffle=True,
          #validation_split=0.2, callbacks=[modeli_checkpoint])

