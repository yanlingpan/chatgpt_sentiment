import pandas as pd
from pathlib import Path

from parse import clean_description
from occupation import tokenize_ngram, extract_occupation

data_dir = "data/"
data = pd.read_csv(Path(data_dir, 'tweets172clean.csv'), sep=',',
                    index_col=0,
                    encoding="utf-8",
                    converters={
                      'user_description': clean_description,
                    },
                    dtype=str,
                    )

# lemmatization
import spacy
nlp = spacy.load("en_core_web_sm")
lemmatized = []
for doc in nlp.pipe(data["user_description"].fillna('')):
  lemmatized.append(" ".join(token.lemma_ for token in doc))
data["user_description"] = lemmatized

# extract unigram, bigram
data['description_12gram'] = data.user_description.map(lambda x: tokenize_ngram(x))

# extract occupation
data['occupation_tuple'] = data.description_12gram.map(lambda x: extract_occupation(x))
data['occupation'] = data.occupation_tuple.map(lambda x: x[-1] if x is not None else 'NA')
data.to_csv(Path(data_dir, 'tweets172_occupation.csv'))

