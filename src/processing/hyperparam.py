import pandas as pd

from src.data.loader import DataLoader
from src.processing.folders import Folders
from src.processing.train import train_unet
from src.processing.predict import prediction


#train various unets on the full dataset
df = pd.DataFrame(columns=['Learning Rate','Num Layers','Filter Size','Conv Depth','Model Name R',
                           'Model Name I','AVG SSIM R','AVG SSIM I'])
for lr in [1e-3, 1e-4, 1e-5]:
    for layers in [4,5,6]:
        for filters in [2,3,4]:
            for conv_depth in [32]:
                model_name_r,model_name_i=train_unet(layers, filters, learn_rate=lr,
                                                 conv_depth=conv_depth, epochs=10,
                                                 loss='mean_squared_error', records=-1,)
                data, label_r, label_i = DataLoader.load_testing(records=64)
                ssim_r=prediction(model_name_r, data, label_r)
                ssim_i=prediction(model_name_i, data, label_i)
                df.loc[len(df)] = [lr,layers,filters,conv_depth,model_name_r,model_name_i,ssim_r,ssim_i]
                df.to_csv(Folders.models_folder()+'Test_Results.csv')


