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
However, input may be of the format:
| p-value            | question wording   |
| ------------------ | ------------------ |
| 0.98738527         | Example #1              |
| 0.12378355         | Example #2               |
| ...         | ...              |

If a dataset is provided, the model can be run with:  
```
python nlp_keras.py
```

# Results  
<p>
  Results on our dataset resulted in overfitting, but the model would be able to accurately estimate difficulty for questions similar to what exists in the database.
</p>
<img src="/img/sample_results.png" width="350" alt="accessibility text">

# Author  
Carl Molnar  
7/27/2019
