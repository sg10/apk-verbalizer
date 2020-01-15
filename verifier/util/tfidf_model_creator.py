import logging
import pickle

import numpy as np
import os
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from verifier.util.text_processing import get_stopwords_list

logging.basicConfig(level=logging.INFO)
log = logging.getLogger()


class TfIdfModelCreator:

    def __init__(self,
                 model_file, data_file,
                 files=[], file_ids=[], files_folder=None,
                 input_analyzer=None,
                 tokenize_function=None,
                 ngram_range=(1, 1),
                 max_df_pct=0.20,
                 min_df_pct=0.02):
        self.doc_ids = file_ids
        self.files = files
        if len(files) != len(file_ids):
            raise RuntimeError("#files is not equal #file_ids")
        self.model_file = model_file
        self.data_file = data_file
        self.input_analyzer = input_analyzer
        self.max_df_pct = max_df_pct
        self.min_df_pct = min_df_pct
        self.ngram_range = ngram_range

        self.docs = [VectorizerDoc(os.path.join(files_folder, f), tokenize_function) for f in files]
        self.tfidf_model = None
        self.tfidf_data = None

    def load(self, data=False, model=False):
        if model:
            self.tfidf_model = pickle.load(open(self.model_file, "rb"))
        if data:
            save_data = pickle.load(open(self.data_file, "rb"))
            self.tfidf_data = save_data['data']
            self.doc_ids = save_data['ids']

    def create_model(self):
        print("creating model")

        min_df = int(len(self.doc_ids) * self.min_df_pct)
        max_df = int(len(self.doc_ids) * self.max_df_pct)

        # way of reading input in form of {token_name: count}
        analyzer = self.input_analyzer or 'word'

        self.tfidf_model = TfidfVectorizer(tokenizer=VectorizerDoc.read_doc,
                                           min_df=min_df,
                                           max_df=max_df,
                                           analyzer=analyzer,
                                           ngram_range=self.ngram_range,
                                           lowercase=False,  # done in tokenize function
                                           stop_words=get_stopwords_list())

        VectorizerDoc.counter = 0
        self.tfidf_model.fit(self.docs)

        print("saving model: ", self.model_file)
        pickle.dump(self.tfidf_model, open(self.model_file, "wb"))
        print("saved")

    def transform_data(self):
        print("transforming data: ")

        VectorizerDoc.counter = 0
        self.tfidf_data = self.tfidf_model.transform(self.docs)

        print("saving data: ", self.data_file)
        save_data = {'ids': self.doc_ids,
                     'data': self.tfidf_data}
        pickle.dump(save_data, open(self.data_file, "wb"))
        print("saved")

    def transform_sample(self, sample):
        self.tfidf_model.set_params(analyzer=lambda x: x, tokenizer=lambda x: x)
        return self.tfidf_model.transform([sample])

    def get_tokens(self, sorted_by_df=False):
        tokens = self.tfidf_model.get_feature_names()
        if sorted_by_df:
            df = self.get_document_frequencies()
            token_and_df = list(zip(tokens, df))
            token_and_df.sort(key=lambda t: t[1], reverse=True)
            tokens = [t[0] for t in token_and_df]
        return tokens

    def get_document_frequencies(self):
        num_documents = len(self.doc_ids)
        df = num_documents / np.exp(self.tfidf_model.idf_) - np.ones(self.tfidf_model.idf_.shape)
        df = df.tolist()
        return df

    def get_inverse_document_frequencies(self):
        return self.tfidf_model.idf_.tolist()

    def info(self):
        df = self.get_document_frequencies()
        print("document frequency:")
        print("    min:                    ", np.min(df))
        print("    max:                    ", np.max(df))
        print("    mean:                   ", np.mean(df))
        print("    median                  ", np.median(df))
        print("number of features:         ", len(self.tfidf_model.get_feature_names()))
        print("number of input files:      ", len(self.files))
        if self.tfidf_data is not None:
            print("number of tfidf docs:       ", self.tfidf_data.shape[0])
            print("tfidf values:")
            print("    min:                    ", np.min(self.tfidf_data.data))
            print("    max:                    ", np.max(self.tfidf_data.data))
            print("    mean:                   ", np.mean(self.tfidf_data.data))
            print("    median                  ", np.median(self.tfidf_data.data))

            nums_per_doc = [self.tfidf_data.getrow(i_row).getnnz() for i_row in range(self.tfidf_data.shape[0])]
            print("number of values per tfidf vector (per document)")
            print("    min:                    ", np.min(nums_per_doc))
            print("    max:                    ", np.max(nums_per_doc))
            print("    median:                 ", np.median(nums_per_doc))
            num_doc_mean = np.mean(nums_per_doc)
            print("    mean                  ", num_doc_mean)
            print("    >   0:                  ", len([n for n in nums_per_doc if n > 0])/len(nums_per_doc))
            print("    >  15:                  ", len([n for n in nums_per_doc if n > 15])/len(nums_per_doc))
            print("    >  50:                  ", len([n for n in nums_per_doc if n > 50])/len(nums_per_doc))


class VectorizerDoc:

    counter = 0

    def __init__(self, file_path, tokenize_function):
        self.file_path = file_path
        self.tokenize_function = tokenize_function

    def read(self):
        VectorizerDoc.counter += 1
        if VectorizerDoc.counter % 1000 == 0:
            print("# Vectorizer docs processed:   ", VectorizerDoc.counter)

        if not os.path.exists(self.file_path):
            return []

        return self.tokenize_function(self.file_path)

    @staticmethod
    def read_doc(doc):
        return doc.read()
