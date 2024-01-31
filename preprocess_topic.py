import pandas as pd
from pathlib import Path

from parse import remove_hashseq_hash_demoji, remove_http_user

data_dir = "data/"
data = pd.read_csv(Path(data_dir, 'tweets172clean.csv'), sep=',',
                    encoding='utf-8',
                    converters={
                      'text': remove_hashseq_hash_demoji,
                    },
                    on_bad_lines='skip', engine='python',
                    dtype=str
                    ) 

data.text = data.text.map(lambda x: remove_hashseq_hash_demoji(x))
# data.text.to_csv(Path(data_dir, '/tweets172clean_hashseq_hash_demoji.csv'))
data.text = data.text.map(lambda x: remove_http_user(x))
data.text.to_csv(Path(data_dir, '/tweets172clean_hashseq_hash_demoji_http_user.csv'))