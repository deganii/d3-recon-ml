import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import pandas as pd
from pandas import DataFrame
from src.processing.folders import Folders


#Define X, Y, and Z labels from dataframe
#Define the 2 hyperparams to exclude (ex_label) and and which values to default to (ex_value)
def surface_plot(x_label='Filter Size',y_label='Num Layers',z_label='Val Loss R',
                 ex_label1='Conv Depth',ex_value1=32,ex_label2='Learning Rate',ex_value2=1e-5):

    df=DataFrame.from_csv(Folders.models_folder()+'Test_Results.csv', header=0, sep=',', index_col=0, parse_dates=True,
                   encoding=None, tupleize_cols=None, infer_datetime_format=False)
    X = np.array([])
    Y = np.array([])
    Z = np.array([])

    for i in range(df[x_label].size):
        if df[ex_label1][i] == ex_value1 and df[ex_label2][i] == ex_value2:
                X = np.append(X, df[x_label][i])
                Y = np.append(Y, df[y_label][i])
                Z = np.append(Z, df[z_label][i])

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(X, Y, Z, zdir='z', c=Z, cmap='viridis',s=20,depthshade=True)
    ax.plot_trisurf(X, Y, Z,cmap='viridis')
    return plt.show()

surface_plot(x_label='Filter Size')