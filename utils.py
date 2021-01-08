import re
import numpy
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Natural Language Processing libraries
import spacy
from spacy.attrs import LOWER, POS, ENT_TYPE, IS_ALPHA
from spacy.tokens import Doc

# Regex to clean the data, and remove stop words
def clean_text(text):
    stop_words = set(stopwords.words("english")) 

    lemmatizer = WordNetLemmatizer()
    text = re.sub(r'(REPLACED|ITEM DELETED|Re-pilot|do not use).*','',text)     # Cleans up trailing REPLACED and ITEM DELETED and Re-pilot text
    text = re.sub(r'\s\S*\d+\S*(\W|\s)',' ',text)
    text = re.sub(r'(\?\s|\:\s|\d\.\s\s\s).*',"",text)                          # Cleans up any other trailing text after questions
    text = re.sub(r'\[.*?\]', "", text)
    text = re.sub(r'\[.*?\]', "", text)
    
    text = [lemmatizer.lemmatize(token) for token in text.split(" ")]
    text = [lemmatizer.lemmatize(token, "v") for token in text]
    
    text = [word for word in text if not word in stop_words]
    text = [x.lower() for x in text]
    text = " ".join(text)
    return text

def remove_tokens_on_match(doc):
    tokens_to_remove = [
        "NNP", "NN", "NNS", "CD", "UH", "JJ"
    ]

    indexes = []
    for index, token in enumerate(doc):
        # print(index, token.text, token.tag_, token.dep_)
        for tag in tokens_to_remove:
            if (token.tag_ == tag):
                indexes.append(index)
                # print("REMOVE: ",token.text, tag, dep)
    np_array = doc.to_array([LOWER, POS, ENT_TYPE, IS_ALPHA])
    np_array = numpy.delete(np_array, indexes, axis = 0)
    doc2 = Doc(doc.vocab, words=[t.text for i, t in enumerate(doc) if i not in indexes])
    doc2.from_array([LOWER, POS, ENT_TYPE, IS_ALPHA], np_array)
    doc2 = doc2.text
    return doc2