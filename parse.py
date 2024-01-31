import re
import regex
import string
import unicodedata
import contractions

import locale
locale.getpreferredencoding = lambda: "UTF-8"


def clean(text):
  """   
  1. Normalize special fonts
  2. Remove ads: token related and domain sales
  3. Replace URLs and user handles
  4. Remove short tweets (<4 words) AFTER stripping hashtags 
  """
  if text is not None:
    text = unicodedata.normalize('NFKC', text) # Normalize special fonts
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    text = check_keywords(text)
    if text is not None:
      text = replace_url(text) # replace w/ "http"
      text = replace_handle(text) # replace w/ "@user"
      # remove short tweets
      no_hash = strip_hashtag(text) # preserve hashtag
      no_hash = no_hash.replace("@user", " ")
      no_hash = no_hash.replace("http", " ")
      if len(no_hash.split()) <= 3:
        return None
    return text

def check_keywords(text):
  text_lower = text.lower()
  keywords = ['airdrop',r'\$[a-zA-Z]+', 'for sale','domain',
  '#crypto', '#token', '#btc', '#eth', '#ethereum', '#web3',
  ]
  pattern = re.compile("|".join(keywords))
  if pattern.search(text_lower):
    return None
  pattern = re.compile(r"(?<=^|(?<=[^a-zA-Z0-9-_\.]))[#＃][a-z]*coin")
  match = re.search(pattern, text_lower)
  if match is not None:
    return None
  return text

def normalize_font(text):
  if text is not None:
    return unicodedata.normalize('NFKC', text)

# https://github.com/cardiffnlp/timelms/blob/main/scripts/preprocess.py
import xxhash
from datasketch import MinHash, LeanMinHash
def hash_tweet(text, num_perm=16):
  """
  remove duplicates & near duplicates tweets by hashing 
  after 
    lowercasing, 
    stripping punctuation, 
    stripping hashtag sequence, 
    stripping user handle 
  """
  def normalize_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    text = text.lower()
    return text

  def minhash(seq):
    # https://skeptric.com/minhash/
    m = MinHash(num_perm=num_perm, hashfunc=xxhash.xxh64_intdigest)
    for s in seq:
      m.update(s.encode('utf8'))
    return LeanMinHash(m)

  text = strip_hashtag_sequence(text).replace("@user", " ") # added        
  tokens = normalize_text(text).split()  # whitespace tokenization
  return minhash(tokens)


# preoprocess for topic modeling
import demoji
def remove_emoji(text):
  return demoji.replace(text, '')

# remove hashtag symbol
def replace_hash(text):
  return re.compile(r"(?<=^|(?<=[^a-zA-Z0-9-_\.]))[#＃]([a-zA-Z0-9_]+)").sub(r'\1', text)

# remove hashtag sequence
def strip_hashtag_sequence(text):
  text = re.compile(r'((^|\s*)[#＃](\w+)){2,}').sub(r' ', text)
  return text

def remove_hashseq_hash_demoji(text):
  text = strip_hashtag_sequence(text)
  text = replace_hash(text)
  text = remove_emoji(text)
  return text

# remove http, @user
def remove_http_user(text):
  text = text.replace("@user", " ")
  text = text.replace("http", " ")
  return " ".join(text.split())


def strip_hashtag(text):
  # if text is not None:
  text = re.compile(r"(?<=^|(?<=[^a-zA-Z0-9-_\.]))[#＃]([A-Za-z]+[A-Za-z0-9-_]+)").sub(r' ', text)
  return text

def replace_hash(text):
  return re.compile(r"(?<=^|(?<=[^a-zA-Z0-9-_\.]))[#＃]([a-zA-Z0-9_]+)").sub(r'\1', text)

def strip_hashtag_sequence(text):
  text = re.compile(r'((^|\s*)[#＃](\w+)){2,}').sub(r' ', text)
  return text

def strip_url(text):
  text = re.compile(r"https?://\S+|www\.\S+").sub(r' ', text)
  return text

def replace_url(text):
  text = re.compile(r"https?://\S+|www\.\S+").sub(r' http ', text)
  return text

def strip_handle(text):
  # if text is not None:
  # https://www.nltk.org/_modules/nltk/tokenize/casual.html#TweetTokenizer
  HANDLES_RE = regex.compile(
    r"(?<![A-Za-z0-9_!@#\$%&*])@"
    r"(([A-Za-z0-9_]){15}(?!@)|([A-Za-z0-9_]){1,14}(?![A-Za-z0-9_]*@))"
  )
  text = HANDLES_RE.sub(" ", text)
  return text

def replace_handle(text):
  # https://www.nltk.org/_modules/nltk/tokenize/casual.html#TweetTokenizer
  HANDLES_RE = regex.compile(
    r"(?<![A-Za-z0-9_!@#\$%&*])@"
    r"(([A-Za-z0-9_]){15}(?!@)|([A-Za-z0-9_]){1,14}(?![A-Za-z0-9_]*@))"
  )
  text = HANDLES_RE.sub(" @user ", text)
  return text

def strip_handle_hashtag_url(text):
  if text is not None:
    text = strip_handle(text)
    text = strip_hashtag(text)
    text = strip_url(text)
  return text


def remove_digits(text):
  if isinstance(text, str):
    text = [word for word in text.split() if word.isalpha()]
    return " ".join(text)
  else:
    return None


# for user description
def clean_description(text):
  if isinstance(text, str):
    text = text.lower()
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    text = text.replace("’", "'").replace("'s", '')
    text = strip_url(text)
    text = strip_handle(text)
    text = remove_emoji(text)
    text = contractions.fix(text) # replace contraction
    text = text.split()
    if len(text) > 0:
      return " ".join(text)
    else:
      return None
  return None

