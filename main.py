import nltk
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import seaborn as sns

tweets = pd.read_csv("/Users/harshjhunjhunwala/Desktop/github_datasets/charlottesville-on-twitter/aug15_sample.csv")

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

plt.rcParams["figure.figsize"] = [30.0, 20.0]
ax = sns.barplot(y=rslt.index, x="Frequency", data=rslt)
ax.tick_params(labelsize=30)

tags = tweets['hashtags'].str.lower().str.cat(sep=' ').split()
hashtags = pd.DataFrame(Counter(tags).most_common(top_N), columns=['Hashtags', 'Frequency']).set_index('Hashtags')
hashtags = hashtags.iloc[1:]

plt.rcParams["figure.figsize"] = [30.0, 20.0]
ax = sns.barplot(y=rslt.index, x="Frequency", data=rslt)
ax.tick_params(labelsize=30)


# Momentum of tweets:
tweets['created_at'] = pd.to_datetime(tweets['created_at'])
tweets = tweets.set_index('created_at')

df = tweets[['id']]
tweet_volume = df.resample('10min').count()

ax = sns.pointplot(x=tweet_volume.index, y='id', data=tweet_volume)
ax.tick_params(labelsize=25)
for item in ax.get_xticklabels():
    item.set_rotation(90)

# Filter out the most influential and reachable tweets

influential_tweets = tweets[['user_name', 'followers_count']]
# influential_tweets = influential_tweets.sort_values('followers_count')
influential_tweets = influential_tweets.groupby('user_name').first().sort_values('followers_count', ascending=False)[:10]
influential_tweets

# People who tweeted the most in a span of 3 hours