from src.data.loader import DataLoader
from src.processing.folders import Folders
from src.processing.train import train
import keras.models

def retrain(model_name, new_model_name, data, labels, epochs, ):
    train_data, train_label_r, train_label_i = DataLoader.load_training(records=records)
    img_rows, img_cols = train_data.shape[1], train_data.shape[2]

    # Step 2: Configure architecture
    model = keras.models.load_model(Folders.models_folder() + model_name + '/weights.h5')


    # Step 3: Configure Training Parameters and Train
    epoch, train_loss, val_loss = train(model_name, model, train_data, train_label, epochs)

    # give it a new model name
    epoch_i, train_loss_i, val_loss_i = train(model_name, model, train_data, train_label_r, epochs)

    # (TODO) Step 4: Evaluate on Test Set
    #test_data, test_label_r, test_label_i = DataLoader.load_testing(records=records)
    return model_name, epoch, train_loss, val_loss
