import random.randint
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

tweets = pd.read_csv("/Users/harshjhunjhunwala/Desktop/github_datasets/charlottesville-on-twitter")

# Removing stopwords:
top_N = 30
stopwords = nltk.corpus.stopwords.words('english')
RE_stopwords = r'\b(?:{})\b'.format('|'.join(stopwords))
words = (tweets['full_text']
           .str.lower()
           .replace([r'\|', RE_stopwords, r"(&amp)|,|;|\"|\.|\?|’|!|'|:|-|\\|/|https"], [' ', ' ', ' '], regex=True)
           .str.cat(sep=' ')
           .split()
)
rslt = pd.DataFrame(Counter(words).most_common(top_N),
                    columns=['Word', 'Frequency']).set_index('Word')

rslt = rslt.iloc[1:]

# Creating a barplot of the most used Twitter words
plt.rcParams["figure.figsize"] = [30.0, 20.0]
ax = sns.barplot(y=rslt.index, x="Frequency", data=rslt)
ax.tick_params(labelsize=30)

