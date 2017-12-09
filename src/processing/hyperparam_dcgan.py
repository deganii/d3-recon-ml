import pandas as pd

from src.data.loader import DataLoader
from src.processing.folders import Folders
from src.processing.train import train_unet
from src.processing.predict import prediction


#train various unets on the full dataset
df = pd.DataFrame(columns=['Learning Rate','Num Layers','Filter Size','Conv Depth','Model Name R',
                           'AVG SSIM R', 'Train Loss R', 'Val Loss R','Model Name I','AVG SSIM I',
                           'Train Loss I','Val Loss I'])
for lr in [1e-5, 1e-4, 1e-3]:
    for layers in [7,6,5,4]:
        for filters in [4,3,2]:
            for convdepth in [32,2]:
                model_name_r, epoch_r, train_loss_r, val_loss_r, \
                model_name_i, epoch_i, train_loss_i, val_loss_i = \
                    train_unet(layers, filters, learn_rate=lr,
                               conv_depth=convdepth, epochs=10,
                               loss='mean_squared_error', records=-1,)
                data, label_r, label_i = DataLoader.load_testing(records=-1)
                ssim_r=prediction(model_name_r, data, label_r)
                ssim_i=prediction(model_name_i, data, label_i)
                df.loc[len(df)] = [lr,layers,filters,convdepth,
                                   model_name_r,ssim_r,train_loss_r,val_loss_r,
                                   model_name_i,ssim_i,train_loss_i,val_loss_i]
                df.to_csv(Folders.models_folder()+'Test_Results.csv')


