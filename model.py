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

def lstm_model(df, epochs, hps):
    (max_features, maxlen, embedding_vector_length, lstm_units, dropout_val) = hps
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(df['Stem'])
    list_tokenized_train = tokenizer.texts_to_sequences(df['Stem'])

    X_train = pad_sequences(list_tokenized_train, maxlen=maxlen)
    y_train = df['TotalPValue']
    model = Sequential() 
    model.add(Embedding(max_features, embedding_vector_length, input_length=maxlen))
    model.add(Bidirectional(LSTM(lstm_units, return_sequences=True)))
    model.add(Dropout(dropout_val))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mse', optimizer=sgd, metrics=['mean_absolute_error', 'mean_absolute_percentage_error', 'mean_squared_logarithmic_error', 'poisson'])
    # print(model.summary())
    batch_size = 100
    history = model.fit(X_train,y_train, batch_size=batch_size, epochs=epochs, validation_split=0.5, verbose=0)
    return history

def tune_hyperparameters(df, epochs):
    # Hyperparameter tuning
    max_features_list = [512]
    maxlen_list = [64]
    embedding_vector_length_list = [4]
    lstm_units_list = [1]
    dropout_val_list = [0.0]
    batch_size = 100

    total_iterations = len(max_features_list) * len(maxlen_list) * len(embedding_vector_length_list) * len(lstm_units_list) * len(dropout_val_list)

    best_loss = 1
    for max_features in max_features_list:
        for maxlen in maxlen_list:
            for embedding_vector_length in embedding_vector_length_list:
                for lstm_units in lstm_units_list:
                    for dropout_val in dropout_val_list:
                        history = lstm_model(df, epochs, (max_features, maxlen, embedding_vector_length, lstm_units, dropout_val))
                    
                        loss = numpy.amin(history.history['val_loss'][epochs-1])
                        if (loss < best_loss):
                            best_loss = loss
                            best_maxlen = maxlen
                            best_max_features = max_features
                            best_embedding_vector_length = embedding_vector_length
                            best_lstm_units = lstm_units
                            best_dropout_val = dropout_val
                        pos = 1 + (max_features_list.index(max_features) * len(maxlen_list) * len(embedding_vector_length_list) * len(lstm_units_list) * len(dropout_val_list)) + (maxlen_list.index(maxlen) * len(embedding_vector_length_list) * len(lstm_units_list) * len(dropout_val_list)) + (embedding_vector_length_list.index(embedding_vector_length) * len(lstm_units_list) * len(dropout_val_list)) + (lstm_units_list.index(lstm_units) * len(dropout_val_list)) + dropout_val_list.index(dropout_val)
                        print('\nIteration: %d / %d \tval_loss: \t%f \tmax_features: \t\t%d \tmaxlen: \t\t%d \tembedding_vector_length: \t%d \tlstm_units: \t\t%d \tdropout_val: \t\t%f' % (pos, total_iterations, loss, max_features, maxlen, embedding_vector_length, lstm_units, dropout_val))
                        print('\t\t\tBest val_loss: \t%f \tBest max_features: \t%d \tBest maxlen: \t%d \tBest embedding_vector_length: \t%d \tBest lstm_units: \t%d \tBest dropout_val: \t%f' % (best_loss, best_max_features, best_maxlen, best_embedding_vector_length, best_lstm_units, best_dropout_val))
                        keras.backend.clear_session()

    return best_loss, (best_maxlen, best_max_features, best_embedding_vector_length, best_lstm_units, best_dropout_val)