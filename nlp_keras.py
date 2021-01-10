import pandas as pd
import numpy
from numpy import array

import en_core_web_sm

from model import lstm_model
from nlp_utils import clean_text, remove_tokens_on_match

# Load question data and their p-values
print("Loading the data...")
df = pd.read_csv('models/results-not-deleted-RAD.csv', delimiter=",").sample(frac=1)

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
model = lstm_model(df)
print("Tuning hyperparameters...")
best_loss = model.tune_hyperparameters()
print('\nBest loss: %f' % (best_loss))
loss = model.train()
print('Actual loss: %f' % (loss))
model.plot_loss()

# Read inputs for testing
pd.set_option('display.max_rows', model.max_features)
while 1:
    print("\nInput a new question to estimate its difficulty:")
    x_in = input()
    tmp = []
    tmp.append(x_in)
    df_input = pd.DataFrame(tmp, columns={'input'})
    #df_input = df_input.input.apply(lambda x: clean_text(x))
    #df_input = df_input.input.apply(remove_tokens_on_match)
    #print("POS removed input: %s" % df_input)
    print("Cleaned input: %s" % df_input.input)
    padded_seq, prediction = model.predict(df_input)
    print("Padded, tokenized input: %s" % padded_seq)
    print("Prediction: %.4f" % prediction)