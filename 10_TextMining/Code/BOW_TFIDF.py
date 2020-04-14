import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


docs = np.array([
        'This is why I hate the Da Vinci Code, it is so boring',
        'The code is written in Python',
        'This is fucking horrible'])

## Vectorized
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
BOW = vectorizer.fit_transform(docs)
print(BOW.toarray())
print(vectorizer.vocabulary_)

## TF-IDF
from sklearn.feature_extraction.text import TfidfTransformer
np.set_printoptions(precision=1)
tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
X = tfidf.fit_transform(BOW)
print(X.toarray())

## stop word
vectorizer_stopwords = CountVectorizer(stop_words="english")
BOW_stopwords = vectorizer_stopwords.fit_transform(docs)
print(BOW_stopwords.toarray())
print(vectorizer_stopwords.vocabulary_)