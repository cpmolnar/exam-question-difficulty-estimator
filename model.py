# Keras dependencies for training
import numpy
import keras
from keras import initializers, regularizers, constraints, optimizers, layers
from keras import metrics
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense , Input , LSTM , Embedding, Dropout , Activation, GRU, Flatten
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.layers import Convolution1D
from keras.models import Model, Sequential

import matplotlib.pyplot as plt

class lstm_model:
    def __init__(self, df, hps=(256, 32, 4, 1, 0.0), verbose = False):
        self.df = df
        self.epochs = 20
        self.batch_size = 100
        (self.max_features,
         self.maxlen,
         self.embedding_vector_length,
         self.lstm_units,
         self.dropout_val) = hps
        self.tokenizer = Tokenizer(num_words=self.max_features)
        self.tokenizer.fit_on_texts(self.df['Stem'])
        self.verbose = verbose

    def train(self):
        list_tokenized_train = self.tokenizer.texts_to_sequences(self.df['Stem'])

        X_train = pad_sequences(list_tokenized_train, maxlen=self.maxlen)
        y_train = self.df['TotalPValue']
        self.model = Sequential()
        self.model.add(Embedding(self.max_features, self.embedding_vector_length, input_length=self.maxlen))
        self.model.add(Bidirectional(LSTM(self.lstm_units, return_sequences=True)))
        self.model.add(Dropout(self.dropout_val))
        self.model.add(Flatten())
        self.model.add(Dense(1, activation='sigmoid'))
        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='mse', optimizer=sgd, metrics=['mean_absolute_error', 'mean_absolute_percentage_error', 'mean_squared_logarithmic_error', 'poisson'])
        
        self.history = self.model.fit(X_train,y_train, batch_size=self.batch_size, epochs=self.epochs, validation_split=0.5, verbose=0)

        loss = (self.history.history['val_loss'][self.epochs-1])
        return loss

    def predict(self, input):
        list_tokenized_input = self.tokenizer.texts_to_sequences(input.input)
        padded_seq = pad_sequences(list_tokenized_input, maxlen=self.maxlen)
        prediction = self.model.predict(padded_seq)
        return padded_seq, prediction

    def tune_hyperparameters(self):
        # Hyperparameter tuning
        max_features_list = [512]
        maxlen_list = [64]
        embedding_vector_length_list = [4]
        lstm_units_list = [1]
        dropout_val_list = [0.0]

        total_iterations = len(max_features_list) * len(maxlen_list) * len(embedding_vector_length_list) * len(lstm_units_list) * len(dropout_val_list)

        best_loss = 1
        for max_features in max_features_list:
            for maxlen in maxlen_list:
                for embedding_vector_length in embedding_vector_length_list:
                    for lstm_units in lstm_units_list:
                        for dropout_val in dropout_val_list:
                            sample_hps = (max_features, maxlen, embedding_vector_length, lstm_units, dropout_val)
                            sample_model = lstm_model(self.df, sample_hps)
                            sample_model.train()
                    
                            loss = numpy.amin(sample_model.history.history['val_loss'][self.epochs-1])
                            if (loss < best_loss):
                                best_loss = loss
                                self.maxlen = maxlen
                                self.max_features = max_features
                                self.embedding_vector_length = embedding_vector_length
                                self.lstm_units = lstm_units
                                self.dropout_val = dropout_val
                            pos = 1 + (max_features_list.index(max_features) * len(maxlen_list) * len(embedding_vector_length_list) * len(lstm_units_list) * len(dropout_val_list)) + (maxlen_list.index(maxlen) * len(embedding_vector_length_list) * len(lstm_units_list) * len(dropout_val_list)) + (embedding_vector_length_list.index(embedding_vector_length) * len(lstm_units_list) * len(dropout_val_list)) + (lstm_units_list.index(lstm_units) * len(dropout_val_list)) + dropout_val_list.index(dropout_val)
                            print('\nIteration: %d / %d \nval_loss: \t%f \nmax_features: \t\t%d \nmaxlen: \t\t%d \nembedding_vector_length: \t%d \nlstm_units: \t\t%d \ndropout_val: \t\t%f' % (pos, total_iterations, loss, max_features, maxlen, embedding_vector_length, lstm_units, dropout_val))
                            keras.backend.clear_session()

        if (self.verbose):
            print('\t\t\tBest val_loss: \t%f \nBest max_features: \t%d \nBest maxlen: \t%d \nBest embedding_vector_length: \t%d \nBest lstm_units: \t%d \nBest dropout_val: \t%f' % (best_loss, self.max_features, self.maxlen, self.embedding_vector_length, self.lstm_units, self.dropout_val))
        return best_loss

    def plot_loss(self):
        # Plot the data
        history_dict = self.history.history
        loss_values = history_dict['loss']
        val_loss_values = history_dict['val_loss']
        plt.plot(range(self.epochs), loss_values, 'bo', label='Training loss')
        plt.plot(range(self.epochs), val_loss_values, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()