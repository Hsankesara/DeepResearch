import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer,  text_to_word_sequence
from keras.engine.topology import Layer
from keras import initializers as initializers, regularizers, constraints
from keras.callbacks import Callback, ModelCheckpoint
from keras.utils.np_utils import to_categorical
from keras.layers import Embedding, Input, Dense, LSTM, GRU, Bidirectional, TimeDistributed
from keras import backend as K
from keras import optimizers
from keras.models import Model
import nltk
import re
import matplotlib.pyplot as plt
import sys
from sklearn.metrics import roc_auc_score
from nltk import tokenize
from attention_with_context import AttentionWithContext
from sklearn.utils import shuffle
import re

class HAN(object):
    self.max_features = 200000
    max_senten_len = 500
    max_senten_num = 15
    embed_size = 300
    VALIDATION_SPLIT = 0.2
    GLOVE_DIR = "../input/glove6b/glove.6B.300d.txt"
    
    def __init__(self, verbose = 0):
        self.verbose = verbose
        self.model = self.set_model()

    def clean_string(self, string):
        """
        Tokenization/string cleaning for dataset
        Every dataset is lower cased except
        """
        string = re.sub(r"\\", "", string)
        string = re.sub(r"\'", "", string)
        string = re.sub(r"\"", "", string)
        return string.strip().lower()
    
    def preprocessing(self, text, categories, verbose = 0, total_categories):
        paras = []
        labels = []
        texts = []
        max_sent_len_exist = 0
        max_sent_num_exist = 0
        for idx in range(df.text.shape[0]):
            text = clean_str(df.text[idx])
            texts.append(text)
            sentences = tokenize.sent_tokenize(text)
            if max_sent_num_exist < len(sentences):
                max_sent_num_exist = len(sentences)
            for sent in sentences:
                if max_sent_len_exist < len(sent):
                    max_sent_len_exist = len(sent)
            paras.append(sentences)
        if verbose == 1:
            print('Max existing sentence len:', max_sent_len_exist)
            print('Max existant sentence num:', max_sent_num_exist)
        tokenizer = Tokenizer(num_words=max_features, oov_token=True)
        tokenizer.fit_on_texts(texts)
        data = np.zeros((len(texts), max_senten_num,
                         max_senten_len), dtype='int32')
        for i, sentences in enumerate(paras):
            for j, sent in enumerate(sentences):
                if j < max_senten_num:
                    wordTokens = text_to_word_sequence(sent)
                    k = 0
                    for _, word in enumerate(wordTokens):
                        if k < max_senten_len and tokenizer.word_index[word] < max_features:
                            data[i, j, k] = tokenizer.word_index[word]
                            k = k+1
        word_index = tokenizer.word_index
        if verbose == 1:
            print('Total %s unique tokens.' % len(word_index))
        labels = pd.get_dummies(categories)
        if verbose == 1:
            print('Shape of data tensor:', data.shape)
            print('Shape of labels tensor:', labels.shape)
        #assert (total_categories == labels.shape[0])
        #assert (data.shape[0] = labels.shape[0])
        return data, labels
    
    def split_dataset(self, data, labels, verbose = 0):
        indices = np.arange(data.shape[0])
        np.random.shuffle(indices)
        data = data[indices]
        labels = labels.iloc[indices]
        nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

        x_train = data[:-nb_validation_samples]
        y_train = labels[:-nb_validation_samples]
        x_val = data[-nb_validation_samples:]
        y_val = labels[-nb_validation_samples:]
        if verbose == 1:
            print('Number of positive and negative reviews in traing and validation set')
            print(y_train.columns.tolist())
            print(y_train.sum(axis=0).tolist())
            print(y_val.sum(axis=0).tolist())
        return x_train, y_train, x_val, y_val
    
    def get_model(self):
        return self.model
    
    def add_glove_model(self):
        embeddings_index = {}
        f = open(GLOVE_DIR)
        for line in f:
            try:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
            except:
                print(values)
                pass
        f.close()
        return embeddings_index
    
    def get_embedding_matrix(self, verbose = 0):
        embedding_matrix = np.random.random((len(word_index) + 1, embed_size))
        absent_words = 0
        embeddings_index = add_glove_model()
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
            else:
                absent_words += 1
        if verbose == 1:
            print('Total absent words are', absent_words, 'which is', "%0.2f" %
                (absent_words * 100 / len(word_index)), '% of total words')
        return embedding_matrix
    
    def get_embedding_layer(self):
        embedding_matrix = get_embedding_layer(self.verbose)
        return Embedding(len(word_index) + 1, embed_size, weights=[embedding_matrix], input_length=max_senten_len, trainable=True)

    def set_model(self):
        word_input = Input(shape=(max_senten_len,), dtype='float32')
        word_sequences = get_embedding_layer()(word_input)
        word_lstm = Bidirectional(LSTM(100, return_sequences=True))(word_sequences)
        word_dense = TimeDistributed(Dense(200))(word_lstm)
        word_att = AttentionWithContext()(word_dense)
        wordEncoder = Model(word_input, word_att)

        sent_input = Input(shape=(max_senten_num, max_senten_len), dtype='float32')
        sent_encoder = TimeDistributed(wordEncoder)(sent_input)
        sent_lstm = Bidirectional(LSTM(100, return_sequences=True))(sent_encoder)
        sent_dense = TimeDistributed(Dense(200))(sent_lstm)
        sent_att = AttentionWithContext()(sent_dense)
        preds = Dense(31)(sent_att)
        self.model = Model(sent_input, preds)
        self.model.compile(loss='categorical_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
        
        def train_model(self, save_best_model = True):
            if save_best_model:
                checkpoint = ModelCheckpoint(
                    'model-{epoch:03d}.h5', verbose=1, monitor='val_loss', save_best_only=True, mode='auto')
            self.history = model.fit(x_train, y_train, validation_data=(
                x_val, y_val), epochs=80, batch_size=86, verbose = self.verbose, callbacks = [checkpoint])
        
        def plot_results(self):
            # summarize history for accuracy
            plt.plot(history.history['acc'])
            plt.plot(history.history['val_acc'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show()
            # summarize history for loss
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show()
