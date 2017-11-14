import pandas as pd

from src.data.loader import DataLoader
from src.processing.folders import Folders
from src.processing.train import train_unet
from src.processing.predict import prediction


#train various unets on the full dataset
df = pd.DataFrame(columns=[
         'Learning Rate',
         'Num Layers',
         'Filter Size',
         'Model Name R',
         'Model Name I',
         'AVG SSIM R',
         'AVG SSIM I'])
for lr in [1e-3, 1e-4, 1e-5]:
    for layers in [4,5,6]:
        for filters in [2,3,4]:
            model_name_r,model_name_i=train_unet(layers, filters, learn_rate=lr,
                                                 conv_depth=32, epochs=10,
                                                 loss='mean_squared_error', records=-1,)
            data, label_r, label_i = DataLoader.load_testing(records=64)
            ssim_r=prediction(model_name_r, data, label_r)
            ssim_i=prediction(model_name_i, data, label_i)
            df.loc[len(df)] = [lr,layers,filters,model_name_r,model_name_i,ssim_r,ssim_i]
            df.to_csv(Folders.models_folder()+'Test_Results.csv')

            #parameters+=[lr,layers,filters,model_name_r,model_name_i,predict_r,predict_i]
# df = pd.DataFrame({'Learning Rate':[item[0] for item in parameters],
#                     'Num Layers'  : [item[1] for item in parameters],
#                     'Filter Size' :[item[2] for item in parameters]
#                     'Model Name R': [item[3] for item in parameters],
#                     'Model Name I': [item[4] for item in parameters],
#                     'AVG SSIM R'  :[item[5] for item in parameters],
#                     'AVG SSIM I'  :[item[6] for item in parameters]})


