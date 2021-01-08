import pandas as pd
import numpy
from numpy import array
import matplotlib.pyplot as plt

import en_core_web_sm

import model
from utils import clean_text, remove_tokens_on_match

# Load question data and their p-values
print("Loading the data...")
df = pd.read_csv('models/results.csv', delimiter=",").sample(frac=1)

print("Cleaning the text...")
df['Stem'] = df.Stem.apply(clean_text)

# Leveraging SpaCy for POS removal and lemmatization
print("Applying POS and dependancy tagging...")
nlp = en_core_web_sm.load()
df['Stem_parsed'] = df['Stem'].apply(nlp)

print("Removing selected POS tags...")
df['Stem'] = df['Stem_parsed'].apply(remove_tokens_on_match)
        
print("Data after lemmatization and POS removal:")
pd.options.display.max_colwidth = 100
print(df.head(30))
# print("Average number of words per question: %s" % df.Stem.apply(lambda x: len(x.split(" "))).mean())

# LSTM setup with best hyperparameters
epochs = 20
loss, hps = model.tune_hyperparameters(df, epochs)
history = model.lstm_model(df, epochs, hps)
loss = (history.history['val_loss'][epochs-1])
print('Actual loss: %f' % (loss))
    
# Plot the data
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(history_dict['loss']) + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Read inputs for testing
pd.set_option('display.max_rows', hps[0])
while 1:
    x_in = input()
    tmp = []
    tmp.append(x_in)
    df_input = pd.DataFrame(tmp, columns={'input'})
    df_input = df_input.input.apply(lambda x: clean_text(x))
    print("Cleaned input: %s" % df_input)
    #df_input = df_input.input.apply(remove_tokens_on_match)
    #print("POS removed input: %s" % df_input)
    list_tokenized_input = tokenizer.texts_to_sequences(df_input)
    X_te = pad_sequences(list_tokenized_input, maxlen=hps[0])
    prediction = model.predict(X_te)
    print("Padded, tokenized input: %s" % X_te)
    print("Prediction: %.4f" % prediction)