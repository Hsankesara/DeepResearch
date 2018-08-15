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
    def __init__(self, text, labels, pretrained_embedded_vector_path, max_features, max_senten_len, max_senten_num, embedding_size, validation_split = 0.2, verbose = 0):
        try:
            self.max_features = max_features
            self.max_senten_len = max_senten_len
            self.max_senten_num = max_senten_num
            self.embed_size = embedding_size
            self.validation_split = validation_split
            self.embedded_dir = pretrained_embedded_vector_path
            self.text = pd.Series(text)
            self.categories = pd.Series(labels)
            self.classes = self.labels.unique().tolist()
            assert (self.text.shape[0] == self.labels.shape[0])
            self.data, self.labels = self.preprocessing()
            self.x_train, self.y_train, self.x_val, self.y_val = self.split_dataset()
            self.embedding_index = self.add_glove_model()
            self.set_model()
        except AssertionError:
            print('Input and label data must be of same size')

    def clean_string(self, string):
        """
        Tokenization/string cleaning for dataset
        Every dataset is lower cased except
        """
        string = re.sub(r"\\", "", string)
        string = re.sub(r"\'", "", string)
        string = re.sub(r"\"", "", string)
        return string.strip().lower()
    
    def preprocessing(self):
        paras = []
        labels = []
        texts = []
        max_sent_len_exist = 0
        max_sent_num_exist = 0
        for idx in range(self.text.shape[0]):
            text = self.clean_string(self.text[idx])
            texts.append(text)
            sentences = tokenize.sent_tokenize(text)
            paras.append(sentences)
        tokenizer = Tokenizer(num_words=self.max_features, oov_token=True)
        tokenizer.fit_on_texts(texts)
        data = np.zeros((len(texts), self.max_senten_num,
                         self.max_senten_len), dtype='int32')
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
        if self.verbose == 1:
            print('Total %s unique tokens.' % len(word_index))
        labels = pd.get_dummies(self.categories)
        if self.verbose == 1:
            print('Shape of data tensor:', data.shape)
            print('Shape of labels tensor:', labels.shape)
        assert (len(self.classes) == labels.shape[1])
        assert (data.shape[0] = labels.shape[0])
        return data, labels
    
    def split_dataset(self):
        indices = np.arange(data.shape[0])
        np.random.shuffle(indices)
        data = data[indices]
        labels = labels.iloc[indices]
        nb_validation_samples = int(self.validation_split * data.shape[0])

        x_train = data[:-nb_validation_samples]
        y_train = labels[:-nb_validation_samples]
        x_val = data[-nb_validation_samples:]
        y_val = labels[-nb_validation_samples:]
        if self.verbose == 1:
            print('Number of positive and negative reviews in traing and validation set')
            print(y_train.columns.tolist())
            print(y_train.sum(axis=0).tolist())
            print(y_val.sum(axis=0).tolist())
        return x_train, y_train, x_val, y_val
    
    def get_model(self):
        return self.model
    
    def add_glove_model(self):
        embeddings_index = {}
        try:
            f = open(self.glove_dir)
            for line in f:
                try:
                    values = line.split()
                    word = values[0]
                    coefs = np.asarray(values[1:], dtype='float32')
                    embeddings_index[word] = coefs
                except:
                    pass
            f.close()
        except FileNotFoundError:
            print('Embedded file does not found')
            exit()
        return embeddings_index
    
    def get_embedding_matrix(self):
        embedding_matrix = np.random.random((len(word_index) + 1, embed_size))
        absent_words = 0
        for word, i in word_index.items():
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
            else:
                absent_words += 1
        if self.verbose == 1:
            print('Total absent words are', absent_words, 'which is', "%0.2f" %
                (absent_words * 100 / len(word_index)), '% of total words')
        return embedding_matrix
    
    def get_embedding_layer(self):
        embedding_matrix = get_embedding_layer(self.verbose)
        return Embedding(len(word_index) + 1, self.embed_size, weights=[embedding_matrix], input_length=self.max_senten_len, trainable=False)

    def set_model(self):
        word_input = Input(shape=(self.max_senten_len,), dtype='float32')
        word_sequences = get_embedding_layer()(word_input)
        word_lstm = Bidirectional(LSTM(100, return_sequences=True))(word_sequences)
        word_dense = TimeDistributed(Dense(200))(word_lstm)
        word_att = AttentionWithContext()(word_dense)
        wordEncoder = Model(word_input, word_att)

        sent_input = Input(shape=(self.max_senten_num, self.max_senten_len), dtype='float32')
        sent_encoder = TimeDistributed(wordEncoder)(sent_input)
        sent_lstm = Bidirectional(LSTM(100, return_sequences=True))(sent_encoder)
        sent_dense = TimeDistributed(Dense(200))(sent_lstm)
        sent_att = AttentionWithContext()(sent_dense)
        preds = Dense(len(self.classes))(sent_att)
        self.model = Model(sent_input, preds)
        self.model.compile(loss='categorical_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
        
        def train_model(self, best_model_path = None, final_model_path = None, plot_learning_curve = True):
            if save_best_model is not None:
                checkpoint = ModelCheckpoint(best_model_path, verbose=0, monitor='val_loss', save_best_only=True, mode='auto')
            self.history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=80, batch_size=86, verbose = self.verbose, callbacks = [checkpoint])
            if plot_learning_curve:
                plot_results()
            if final_model_path is not None:
                self.model.save(final_model_path)
        
        def plot_results(self):
            # summarize history for accuracy
            plt.subplot(211)
            plt.plot(self.history.history['acc'])
            plt.plot(self.history.history['val_acc'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            
            # summarize history for loss
            plt.subplot(212)
            plt.plot(self.history.history['val_loss'])
            plt.plot(self.history.history['loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show()
