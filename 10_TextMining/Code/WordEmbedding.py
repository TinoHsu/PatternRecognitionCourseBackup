from sklearn.feature_extraction.text import CountVectorizer
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

t.manual_seed(2)

docs = np.array([
        'This is why I hate the Da Vinci Code, it is so boring',
        'The code is written in Python',
        'This is fucking horrible'])

vectorizer = CountVectorizer()
BOW = vectorizer.fit_transform(docs)
print(vectorizer.vocabulary_)
word_to_ix = vectorizer.vocabulary_
# 16 words in vocab, 2 dimensional embeddings
embeds = nn.Embedding(16, 2)  
lookup_tensor = t.tensor([word_to_ix["written"]], dtype=t.long)
written_embed = embeds(lookup_tensor)
print(written_embed)