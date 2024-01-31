import re
import string
import pandas as pd
from pathlib import Path
from collections import defaultdict

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
StopWords = stopwords.words("english")

def tokenize_ngram(text):
  if isinstance(text, str) and len(text) > 0:
    text = text.replace("front end", "frontend").replace("back end", "backend")
    toks = word_tokenize(text)
    unigrams_ls, bigrams_ls, new_toks = [], [], []
    for tok in toks:
      # split sentence by punctuation or stopword
      if tok[0].isalpha() and tok not in StopWords:
        # remove punctuation within word
        tok = re.sub('[%s]' % re.escape(string.punctuation), '', tok)
        new_toks.append(tok)
      else: # extract ngrams of split sentence
        if len(new_toks) > 0:
          unigrams_ls += new_toks
          bigrams_ls += [" ".join(ngram) for ngram in nltk.ngrams(new_toks, 2, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>')]
          new_toks = []
    # extract ngrams of the rest of sentence
    if len(new_toks) > 0:
      unigrams_ls += new_toks
      bigrams_ls += [" ".join(ngram) for ngram in nltk.ngrams(new_toks, 2, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>')]
      new_toks = []

    # return unigrams and bigrams separated by " || "
    # each separated by " | "
    ngrams_ls = []
    if len(unigrams_ls) == 0 and len(bigrams_ls) == 0:
      return None
    for grams in [unigrams_ls, bigrams_ls]:
      if len(grams) != 0:
        ngrams_ls.append(" | ".join(grams))
    return " || ".join(ngrams_ls)#, ngrams_ls
  else:
    return None


# load curated lists
data_dir = "data/"
# title2occupation
title2occupation_df = pd.read_csv(Path(data_dir, 'title2occupation.csv')).drop_duplicates()
# titles belong to multiple occupation groups are ambiguous
ambiguous = set(title2occupation_df[title2occupation_df.key.duplicated()].key.tolist())
unambiguous = set(title2occupation_df.key.tolist()) - ambiguous

print(f'all title-occupation pair:{len(title2occupation_df)}\
      \tunambiguious titles:{len(unambiguous)}\
      \tambiguous titles:{len(ambiguous)}')

# each ambiguous titles have a default fallback occupation value, 
# in case disambiguation doesnot work
title2occupation_df = title2occupation_df.drop_duplicates(subset=['key'])
title2occupation_dict = pd.Series(title2occupation_df.occupation.values, index=title2occupation_df.key).to_dict()
soc_set_title = list(title2occupation_dict.keys())
assert (unambiguous | ambiguous) == set(soc_set_title)

print(f'\nambiguious title defaults:\n')
print(f"{'title'.ljust(15)}occupation\n{'-'*30}")
for title in ambiguous:
  print(f"{title:15}{title2occupation_dict[title]}")

# modifier2occupation
modifier2occupation = pd.read_csv(Path(data_dir, 'modifier2occupation.csv'))
modifier2occupation = dict(zip(list(modifier2occupation.modifier),list(modifier2occupation.occupation)))
print(len(modifier2occupation.keys()))
set(modifier2occupation.values())

# keyword2occupation
keyword2occupation = pd.read_csv(Path(data_dir, 'keyword2occupation.csv'))
keyword2occupation = dict(zip(list(keyword2occupation.keyword),list(keyword2occupation.occupation)))
print(len(keyword2occupation.keys()))


def disambiguate(modifier):
  try:
    return modifier2occupation[modifier]
  except KeyError:
    return None
  
def match_keyword(keyword):
  try:
    return keyword2occupation[keyword]
  except KeyError:
    return None
  
def extract_occupation(text):
  if isinstance(text, str):
    unigrams, bigrams = text.split(" || ")
  else:
    return None

  # return first matched title from bigrams and unigrams
  bigrams = bigrams.split(" | ")
  bigrams = [s.split() for s in bigrams]
  for ngram in bigrams:
    # unambiguous title in either position
    for gram in ngram:
      if gram in unambiguous: # 1st position 
        assert isinstance(title2occupation_dict[gram], str)
        print(f"unambiguous: {gram} in {ngram}")
        return (' '.join(ngram), title2occupation_dict[gram])
    # ambiguous title @ 2nd position disambiguate by 1st position
    if ngram[1] in ambiguous:
      # print(ngram[0])
      match = disambiguate(ngram[0])
      # print(f"match: {match}")
      if match is not None: # disambiguate success
        assert isinstance(match, str)
        print(f"ambiguous @ 2nd: {ngram[1]} in {ngram}")
        return (' '.join(ngram), match)
    # return ambiguous title @ 1st position, return default occupation
    if ngram[0] in ambiguous:
      assert isinstance(title2occupation_dict[ngram[0]], str)
      print(f"ambiguous @ 1st: {ngram[0]} in {ngram}")
      return (' '.join(ngram), title2occupation_dict[ngram[0]])
    
  # match keyword in unigrams, cast majority vote, resolve tight by order
  unigrams = unigrams.split(" | ")
  key_occ_ls = []
  key_occ_cnt = defaultdict(int)
  for gram in unigrams:
    match = match_keyword(gram)
    if match is not None:
      assert isinstance(match, str)
      print(f"keyword: {gram}, {match}")
      key_occ_ls.append((gram, match))
      key_occ_cnt[match] += 1

  if len(key_occ_ls) > 0:
    occ, cnt = sorted(key_occ_cnt.items(), key=lambda item: item[1], reverse=True)[0]
    return (cnt, occ)
  
  return None