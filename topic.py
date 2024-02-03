import time
import numpy as np
import pandas as pd
from pathlib import Path


"""### Data"""
f_name = "tweets172clean_hashseq_hash_demoji_http_user"
data = pd.read_csv(f_name+".csv", index_col=0)
print(f"data: {data.shape}\n{data.head()}\n{data.tail()}")

"""### Topic Modeling"""

from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from bertopic.representation import MaximalMarginalRelevance
from bertopic.vectorizers import ClassTfidfTransformer

embedding_model = SentenceTransformer("all-mpnet-base-v2")
umap_model = UMAP(n_neighbors=100, n_components=5, min_dist=0.0, metric='cosine')
hdbscan_model = HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words="english")
ctfidf_model = ClassTfidfTransformer()
representation_model = MaximalMarginalRelevance(diversity=0.3)


topic_model = BERTopic(
  embedding_model = embedding_model,           # Step 1 - Extract embeddings
  umap_model = umap_model,                     # Step 2 - Reduce dimensionality
  hdbscan_model = hdbscan_model,               # Step 3 - Cluster reduced embeddings
  vectorizer_model = vectorizer_model,         # Step 4 - Tokenize topics
  ctfidf_model = ctfidf_model,                 # Step 5 - Extract topic words
  representation_model = representation_model, # Step 6 - (Optional) Fine-tune topic represenations
  calculate_probabilities = False, 
  verbose = True,
  language = "multilingual",
  low_memory = False,
  min_topic_size = 100
)

"""#### train"""
datetime = time.strftime("%m%d_%H%M", time.localtime()) + "_"
out_name = "output/" + datetime + f_name
tic = time.time()

tweets = data.text.tolist()

topics, probs = topic_model.fit_transform(tweets)

# extract topics
def get_keywords(id):
  return " ".join([k for k,v in topic_model.get_topic(id)])
# topic frequency
freq = topic_model.get_topic_info()
freq['keywords'] = freq.Topic.map(lambda x: get_keywords(x))
freq.to_csv(out_name+'_TopicFreq.csv', index=False)
# representative docs
pd.DataFrame.from_records(topic_model.get_representative_docs()).transpose().to_csv(out_name+'_RepDoc.csv', index=True)
# document info
topic_model.get_document_info(tweets).to_csv(out_name+'_DocTopic.csv', index=True)

tok = time.time()
print(time.strftime("%Hh%Mm%Ss", time.gmtime(tok - tic)))

# save model
topic_model.save(out_name+'_model')

