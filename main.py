
"""
Created on Thu Apr 15 18:50:20 2021

@author: uriya
"""
import os
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
word_list=[]
os.chdir(r'C:\Users\uriya\Desktop\Uriya\Methods_for_Detecting_Cyber_Attack\FraudedRawData')
with open(r'User10','r') as f:
    for line in f:
        word_list.append(line) 
        
#%%
counts = Counter(word_list)
labels, values = zip(*counts.items())
# sort your values in descending order
indSort = np.argsort(values)[::-1]
# rearrange your data
labels = np.array(labels)[indSort]
values = np.array(values)[indSort]
indexes = np.arange(len(labels))
bar_width = 0.35
plt.bar(indexes, values)
# add labels
plt.xticks(indexes + bar_width, labels)
plt.show()
#%%TF-IDF - https://towardsdatascience.com/natural-language-processing-feature-engineering-using-tf-idf-e8b9d00e7e76
uniqueWords  = list(dict.fromkeys(word_list))
word_list = np.reshape(word_list,(150,100))
print(word_list[0])
#%%-https://towardsdatascience.com/machine-learning-nlp-text-classification-using-scikit-learn-python-and-nltk-c52b92a7c73a
from sklearn.feature_extraction.text import CountVectorizer
corpus = [
'This is the first document.',
'This document is the second document.',
'And this is the third one.',
'Is this the first document?',]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names())

print(X.toarray())

vectorizer2 = CountVectorizer(analyzer='word', ngram_range=(2, 2))
X2 = vectorizer2.fit_transform(corpus)
print(vectorizer2.get_feature_names())
print(X2.toarray())
#%%
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(word_list)
print(vectorizer.get_feature_names())

print(X.toarray())