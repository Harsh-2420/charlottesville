import nltk
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import seaborn as sns
from nltk.corpus import stopwords

aug15 = pd.read_csv("/Users/harshjhunjhunwala/Desktop/github_datasets/charlottesville-on-twitter/aug15_sample.csv")
aug16 = pd.read_csv("/Users/harshjhunjhunwala/Desktop/github_datasets/charlottesville-on-twitter/aug16_sample.csv")
aug17 = pd.read_csv("/Users/harshjhunjhunwala/Desktop/github_datasets/charlottesville-on-twitter/aug17_sample.csv")

# Removing stopwords:
top_N = 30
stopwords = stopwords.words('english') 
RE_stopwords = r'\b(?:{})\b'.format('|'.join(stopwords))

# most used words in the tweets and plotting it on barplot
words = (aug15['full_text']
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

# most used hashtags in the tweets and plotting it on barplot
def find_hashtags(filepath):
    tags = filepath['hashtags'].str.lower().str.cat(sep=' ').split()
    hashtags = pd.DataFrame(Counter(tags).most_common(top_N), columns=['Hashtags', 'Frequency']).set_index('Hashtags')
    hashtags = hashtags.iloc[1:]
    return hashtags

hashtags15 = find_hashtags(aug15)

plt.rcParams["figure.figsize"] = [30.0, 20.0]
ax = sns.barplot(y=hashtags15.index, x="Frequency", data=hashtags15)
ax.tick_params(labelsize=30)


# Momentum of tweets:
aug15['created_at'] = pd.to_datetime(aug15['created_at'])
aug15 = aug15.set_index('created_at')

df = aug15[['id']]
tweet_volume = df.resample('10min').count()

ax = sns.pointplot(x=tweet_volume.index, y='id', data=tweet_volume)
ax.tick_params(labelsize=25)
for item in ax.get_xticklabels():
    item.set_rotation(90)

# Filter out the most influential tweets
influential_tweets = aug15[['user_name', 'followers_count']]
influential_tweets = influential_tweets.groupby('user_name').first().sort_values('followers_count', ascending=False)[:10] 

# hashtags16 = find_hashtags(aug16)
# hashtags17 = find_hashtags(aug17)
# total_hashtags = np.concat([hashtags15, hashtags16, hashtags17])

# plot total_hashtags
# Use nltk further