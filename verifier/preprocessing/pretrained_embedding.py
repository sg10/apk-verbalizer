import numpy as np
import os
import pickle
import pprint
import re

from verifier import config


class PreTrainedEmbeddings:
    _instance = None

    @staticmethod
    def get():
        if PreTrainedEmbeddings._instance is None:
            PreTrainedEmbeddings._instance = PreTrainedEmbeddings()
        return PreTrainedEmbeddings._instance

    TOKENS = {
        'UNKNOWN': '<unk>',
        'START': '<start>',
        'END': '<end>'
    }

    def __init__(self):
        self.dim = config.Embeddings.embedding_dim
        self.embedding_matrix = None
        self.word2index_dict = None
        self.delimiter_regex_pattern = None
        self.split_pattern_compiled = None
        self.load()

    def load(self):
        if self.is_cached():
            print("loading cached word embeddings file: ",
                  config.Embeddings.cached_embedding_file_values,
                  " -- ",
                  config.Embeddings.downloaded_embedding_file)
            self.load_cached()
        elif os.path.exists(config.Embeddings.downloaded_embedding_file):
            print("loading raw word embeddings file: ", config.Embeddings.downloaded_embedding_file)
            self.create_cached()
        else:
            raise RuntimeError("could not find raw or cached word embeddings file")

    def get_delimiter_regex_pattern(self):
        if self.delimiter_regex_pattern:
            return self.delimiter_regex_pattern

        chars = set()
        for k in self.word2index_dict.keys():
            for c in k:
                chars.add(c)

        chars_regex = "".join([re.escape(c) for c in chars])

        regex_pattern = "[^%s]" % chars_regex

        self.delimiter_regex_pattern = regex_pattern

        return regex_pattern

    def is_cached(self):
        cached_file = config.Embeddings.cached_embedding_file_values
        check_file = config.Embeddings.cached_embedding_file_check
        downloaded_file = config.Embeddings.downloaded_embedding_file

        # use size of downloaded file as check if the cached file is valid
        return os.path.exists(cached_file) and \
            os.path.exists(check_file) and \
            open(check_file, "r").read() == str(os.stat(downloaded_file).st_size)

    def create_cached(self):
        word2coeffs = {}
        word2index = {}

        embedding_dim_without_tokens = self.dim - 1

        with open(config.Embeddings.downloaded_embedding_file) as f:
            for idx, line in enumerate(f):
                try:
                    data = [x.strip().lower() for x in line.split()]
                    if len(data) < embedding_dim_without_tokens:
                        print("skipped line %d, length is only %d" % (idx, len(data)))
                        continue
                    word = data[0]
                    coeffs = np.zeros(self.dim, dtype='float32')
                    coeffs[0:self.dim - 1] = data[1:self.dim]
                    word2coeffs[word] = coeffs
                    if word not in word2index:
                        word2index[word] = len(word2index)
                except Exception as e:
                    print('Exception occurred in `load_glove_embeddings`:', e)
                    continue

        # add tokens in additional dimensionality (50->51, 300-301, etc.)
        for i, value in enumerate(PreTrainedEmbeddings.TOKENS.values()):
            coeffs = np.zeros(self.dim)
            coeffs[self.dim - 1] = -0.5 + i / (len(PreTrainedEmbeddings.TOKENS) - 1)
            word2coeffs[value] = coeffs
            word2index[value] = len(word2index)

        vocab_size = len(word2coeffs)
        embedding_matrix = np.memmap(config.Embeddings.cached_embedding_file_values,
                                     dtype='float32',
                                     mode='w+',
                                     shape=(vocab_size, self.dim))
        for word, idx in word2index.items():
            embedding_vec = word2coeffs.get(word)
            if embedding_vec is not None and embedding_vec.shape[0] == self.dim:
                embedding_matrix[idx] = np.asarray(embedding_vec)

        # embedding_matrix[len(word2index) - 1, :] = embedding_matrix[len(word2index) - 1, :] * 0

        self.word2index_dict = word2index
        self.embedding_matrix = embedding_matrix

        pickle.dump(word2index, open(config.Embeddings.cached_embedding_file_indices, "wb"))

        # for cache validity check
        open(config.Embeddings.cached_embedding_file_check, "w")\
            .write(str(os.stat(config.Embeddings.downloaded_embedding_file).st_size))

        print("saved raw embedding")

    def load_cached(self):
        self.word2index_dict = pickle.load(open(config.Embeddings.cached_embedding_file_indices, "rb"))
        shape = (len(self.word2index_dict), self.dim)
        self.embedding_matrix = np.memmap(config.Embeddings.cached_embedding_file_values,
                                          mode="r", dtype="float32", shape=shape)

    def get_unknown_idx(self):
        return self.word2index(PreTrainedEmbeddings.TOKENS['UNKNOWN'])

    def word2index(self, word, return_unknown_token=False):
        idx = self.word2index_dict.get(word)
        if idx is not None:
            return idx

        idx = self.word2index_dict.get(word.lower())
        if idx is not None:
            return idx

        if return_unknown_token:
            return self.word2index_dict.get(PreTrainedEmbeddings.TOKENS['UNKNOWN'])

        return -1

    def interpret(self, vec):
        if not isinstance(vec, np.ndarray):
            raise ValueError("not a valid vector")

        words = self.nearest_vectors(vec)

        return words[0]

    def index2word(self, search_index):
        if type(search_index) is not int and type(search_index) is not np.int64:
            raise RuntimeError("search index is not int, is %s" % type(search_index))

        for word, index in self.word2index_dict.items():
            if search_index == index:
                return word

        return None

    def nearest_vectors(self, vec, num_words=1):
        d = (np.sum(vec ** 2, ) ** (0.5))
        vec_norm = (vec.T / d).T

        dist = np.dot(self.embedding_matrix, vec_norm.T)

        a = np.argsort(-dist)[:num_words]

        words = []

        for x in a:
            words.append((x, self.embedding_matrix[x], dist[x]))

        return words

    def tokens_to_indices(self, tokens, max_tokens=None):
        tokens = list(tokens)

        embedding_indices = []

        for token in tokens:
            idx = self.word2index(token)
            if idx > -1:
                embedding_indices.append(idx)
                continue

            subtokens = re.split("[^A-z0-9]", token)
            found = False
            for subtoken in subtokens:
                idx = self.word2index(subtoken)
                if idx > -1:
                    embedding_indices.append(idx)
                    found = True

            if found:
                continue

            subtokens = re.split("[^A-z]", token)
            for subtoken in subtokens:
                idx = self.word2index(subtoken)
                if idx > -1:
                    embedding_indices.append(idx)

        embedding_indices = list(filter(lambda e: e >= 0, embedding_indices))

        if max_tokens:
            embedding_indices = embedding_indices[:max_tokens]

        return embedding_indices


def test():
    from verifier.preprocessing.samples_database import SamplesDatabase
    e = PreTrainedEmbeddings.get()
    pt = e.get_delimiter_regex_pattern()

    db = SamplesDatabase.get()
    desc = db.read('com.whatsapp', 'description_raw')
    ts = re.split(pt, desc, flags=re.IGNORECASE)

    print(desc)
    print("|".join(ts))

    ids = e.tokens_to_indices(ts)
    ewords = [e.index2word(id) for id in ids]
    print(" ".join(ewords))



if __name__ == "__main__":
    test()
