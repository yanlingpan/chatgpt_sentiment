import pandas as pd
from pathlib import Path
from collections import Counter

from parse import clean, normalize_font, hash_tweet

data_dir = "data/"
data = pd.read_csv(Path(data_dir, 'tweets172.csv'), sep=',',
                    encoding='utf-8',
                    converters={
                      'text': clean,
                      'user_name': normalize_font,
                      'user_description': normalize_font,
                      'user_location': normalize_font,
                    },
                    on_bad_lines='skip', engine='python',
                    dtype=str
                    ) 

# remove rows with shifted columns (i.e. NA in the specified columns)
data = data.dropna(subset=['user_name', 'user_created', 'user_followers', 
                           'user_friends', 'user_favourites', 'user_verified',
                           'text', 'date', 'source'])


# https://github.com/cardiffnlp/timelms/blob/main/scripts/preprocess.py
# remove top 1% active user
user_counter = Counter(data.user_name)
top_users = [user for user, _ in user_counter.most_common()]

n_blacklisted_users = int(len(top_users)*0.01)
blacklisted_users = set(top_users[:n_blacklisted_users])

# additional stats
n_users = len(user_counter.keys())
pct_blacklisted_users = round((n_blacklisted_users / n_users) * 100, 2)

n_blacklisted_tweets = sum([user_counter[u] for u in blacklisted_users])
pct_blacklisted_tweets = round((n_blacklisted_tweets / sum(user_counter.values())) * 100, 2)

print(f"Blacklisted\t{len(blacklisted_users):,}\tusers ({pct_blacklisted_users}%)\
      \nIgnoring\t{n_blacklisted_tweets:,}\ttweets ({pct_blacklisted_tweets}%)")

data = data[~data.user_name.isin(blacklisted_users)]

# remove duplicates & near duplicates by hashing
written_hashes = set()
discard_idx = set()
for idx, tweet in data.iterrows():
  tweet_hash = hash_tweet(tweet.text)
  if tweet_hash in written_hashes:
    discard_idx.add(idx)
  else:
    written_hashes.add(tweet_hash)
print(f"written hash: {len(written_hashes):,}\ndiscard_idx: {len(discard_idx):,}")
data = data[~data.index.isin(discard_idx)]

# tweet by date
data['date_by_day'] = pd.to_datetime(data.date).dt.date
data['date_by_time'] = pd.to_datetime(data.date).dt.time
data['date_by_date'] = pd.to_datetime(data.date)

# save cleaned data
data.to_csv(Path(data_dir, 'tweets172clean.csv'), index=True)