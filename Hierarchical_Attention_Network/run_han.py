import pandas as pd
import numpy as np
import HAN
from sklearn.utils import shuffle
import sys

def preprocessing(df):
    df['text'] = df['headline'] + '. ' + df['short_description']
    df = df[['text', 'category']]
    df.category = df.category.map(lambda x: "WORLDPOST" if x == "THE WORLDPOST" else x)
    return df

def show_help():
    pass

def main():
    """
    A small tutorial to use HAN module
    """
    filename = './News_Category_Dataset/News_Category_Dataset.json'
    df = pd.read_json(filename, lines=True).reset_index()
    df = preprocessing(df)
    han_network = HAN.HAN(text = df.text, labels = df.category, num_categories = 30, pretrained_embedded_vector_path = './glove.6B/glove.6B.100d.txt', max_features = 200000, max_senten_len = 150, max_senten_num = 4 , embedding_size = 100, validation_split=0.2, verbose=1)
    print(han_network.get_model().summary())
    han_network.show_hyperparameters()
    ## How to change hyperparameters
    # Let's add regularizers
    # To replace a hyperparameter change the corresponding key value to the new value in set_hyperparameters
    han_network.set_hyperparameters({'l2_regulizer': 1e-13, 'dropout_regulizer': 0.5})
    han_network.show_hyperparameters()
    print(han_network.get_model().summary())
    han_network.train_model(epochs=3, batch_size=16,
                            best_model_path='./best_model.h5')



if __name__ == '__main__':
    main()
