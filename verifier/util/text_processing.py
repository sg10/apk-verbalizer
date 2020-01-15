import time

from verifier.util.unstem import UnStemmer

import nltk
from nltk.corpus import stopwords

_stemmer = None
_stopwords_list = stopwords.words('english')
_stopwords_list_clean = None
_stemmed_cache = {}


nltk.download('punkt')
nltk.download('stopwords')


def should_use_token(token):
    return len(token) > 1 and \
           len([i for i in token if ord(i) < 128]) > len(token) / 2 - 1 and \
           not token.isnumeric()


def remove_trailing_leading_nonalpha_from_token(token):
    while len(token) > 0 and not token[0].isalnum():
        token = token[1:]
    while len(token) > 0 and not token[-1].isalnum():
        token = token[:-1]
    return token


def clean_token(token):
    token = remove_trailing_leading_nonalpha_from_token(token)
    token = stem_cached(token)
    token = token.lower()

    return token


def stem_cached(token):
    global _stemmer, _stemmed_cache
    if _stemmer is None:
        _stemmer = nltk.PorterStemmer()

    if token not in _stemmed_cache:
        _stemmed_cache[token] = _stemmer.stem(token)

    stemmed = _stemmed_cache[token]

    if UnStemmer.enabled:
        UnStemmer.get().add(token, stemmed)

    return stemmed


def tokenize_text(text, clean_and_stem=True):
    for replace_sign in [".", "/", ":", "*", "=", "_"]:
        text = text.replace(replace_sign, " ")

    # remove non-ascii
    text = ''.join(char for char in text if ord(char) < 128)

    #tokens_orig = nltk.tokenize.TweetTokenizer(preserve_case=True).tokenize(text)
    tokens_orig = nltk.word_tokenize(text)
    tokens = tokens_orig

    if clean_and_stem:
        tokens = [clean_token(t) for t in tokens]
        tokens = [t for t in tokens if should_use_token(t)]

    return tokens


def get_stopwords_list(clean=True):
    if clean:
        global _stopwords_list_clean
        if _stopwords_list_clean is None:
            _stopwords_list_clean = [clean_token(token) for token in _stopwords_list]

        return _stopwords_list_clean
    else:
        return _stopwords_list

