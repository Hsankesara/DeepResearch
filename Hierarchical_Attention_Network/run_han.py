import pandas as pd
import numpy as np
import HAN
from sklearn.utils import shuffle
import sys

def preprocessing(df):
    df['text'] = df['headline'] + '. ' + df['short_description']
    df = df[['text', 'category']]
    return df

def show_help():
    pass

def main():
    if len(sys.argv) == 2:
        show_help()
    else:
        filename = sys.argv[3]
    df = shuffle(pd.read_json(
        filename, lines=True))[:300].reset_index()
    df = preprocessing(df)
    han_network = HAN()
    data, labels = han_network.preprocessing(df.text, df.category, verbose=0, 31)
    x_train, y_train, x_val, y_val = han_network(data, labels)
    han_network.train(x_train, y_train, x_val, y_val)
    han_network.plot_results()



if __name__ == '__main__':
    main()
