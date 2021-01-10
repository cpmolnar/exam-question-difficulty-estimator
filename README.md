# Exam Question Difficulty Estimation
An NLP model for estimating exam question difficulty, given its p-value and wording.  

The model leverages SpaCy and NLTK for removal of POS tagging and removal of stop-words, as well as lemmatization.  
It also attempts simple hyperparameter tuning to find the best hyperparameter values.  

The approach for this project was:  
- Clean database artifacts from the input
- Lemmatize the input, e.g. reading becomes read, to massively reduce the dictionary size
- Stop-words are then removed, as well as problematic POS
- Hyperparameter tuning is performed
- Final model is trained from the best hyperparameters

After final training, the user is encouraged to input other exam questions, for real-time exam question difficulty estimations, for purposes of checking new exam questions and testing.

# Usage  
Sadly, the database cannot be provided for confidentiality reasons.  
However, input may be of the csv format:
| p-value            | question wording   |
| ------------------ | ------------------ |
| 0.98738527         | Example #1              |
| 0.12378355         | Example #2               |
| ...         | ...              |

If a dataset is provided and the filepath is specified, the model can be run with:  
```
python nlp_keras.py
```

# Results  
<p>
  Results on our dataset resulted in overfitting, due to data constraints. While training loss approaches 0.03, validation loss approaches 0.035.
</p>
<img src="/img/sample_results.png" width="350" alt="accessibility text">

# Conclusion
<p>
  While overfitting is a problem, the model generates some interesting results within reasonable similarity to existing questions, which could provide Psychometrics researchers with a very rough difficulty estimation when re-wording exam questions.
</p>
<p>
  Unfortunately, NLP problems take massive amounts of data, which I simply didn't have access to. I had roughly 20k exam questions, and data would be heavily filtered by content category, giving roughly 200-2,000 questions per category. Additionally, a model would ideally be trained with contextual information about the topics in the questions, but the words in the dataset are highly discipline-specific and otherwise uncommon.  
</p>

# Author  
Carl Molnar  
7/27/2019
