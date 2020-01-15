import html
import logging
import numpy as np
import pickle
import random

from html.parser import HTMLParser
from sklearn.feature_extraction.text import TfidfVectorizer

from verifier import config
from verifier.preprocessing.pretrained_embedding import PreTrainedEmbeddings
from verifier.preprocessing.samples_database import SamplesDatabase
from verifier.util.text_processing import tokenize_text, get_stopwords_list
from verifier.util.unstem import UnStemmer

logging.basicConfig(level=logging.INFO)
log = logging.getLogger()


def meta_data_description_tokenize(package_name):
    text = SamplesDatabase.get().read(package_name, 'description_raw')
    text = html.unescape(text)
    tokens = tokenize_text(text)
    return tokens


def get_document_frequencies(tfidf_model, num_documents):
    df = num_documents / np.exp(tfidf_model.idf_) - np.ones(tfidf_model.idf_.shape)
    df = df.tolist()
    return df


def get_inverse_document_frequencies(tfidf_model):
    return tfidf_model.idf_.tolist()


def info(tfidf_model, tfidf_data):
    df = get_document_frequencies(tfidf_model, tfidf_data.shape[0])
    print("document frequency:")
    print("    min:                    ", np.min(df))
    print("    max:                    ", np.max(df))
    print("    mean:                   ", np.mean(df))
    print("    median                  ", np.median(df))
    print("number of features:         ", len(tfidf_model.get_feature_names()))
    if tfidf_data is not None:
        print("number of tfidf docs:       ", tfidf_data.shape[0])
        print("tfidf values:")
        print("    min:                    ", np.min(tfidf_data.data))
        print("    max:                    ", np.max(tfidf_data.data))
        print("    mean:                   ", np.mean(tfidf_data.data))
        print("    median                  ", np.median(tfidf_data.data))

        nums_per_doc = [tfidf_data.getrow(i_row).getnnz() for i_row in range(tfidf_data.shape[0])]
        print("number of values per tfidf vector (per document)")
        print("    min:                    ", np.min(nums_per_doc))
        print("    max:                    ", np.max(nums_per_doc))
        print("    median:                 ", np.median(nums_per_doc))
        num_doc_mean = np.mean(nums_per_doc)
        print("    mean                    ", num_doc_mean)
        print("    >   0:                  ", len([n for n in nums_per_doc if n > 0]) / len(nums_per_doc))
        print("    >  15:                  ", len([n for n in nums_per_doc if n > 15]) / len(nums_per_doc))
        print("    >  50:                  ", len([n for n in nums_per_doc if n > 50]) / len(nums_per_doc))


def create():
    # SamplesDatabase.set_file('samples_database.pk')

    db = SamplesDatabase.get()
    packages = db.filter(('lang', '==', 'en'))  # [:20000]

    print("creating model")

    min_df_pct = 0.002
    max_df_pct = 0.4

    min_df = int(len(packages) * min_df_pct)
    max_df = int(len(packages) * max_df_pct)

    UnStemmer.enabled = True

    tfidf_model = TfidfVectorizer(tokenizer=meta_data_description_tokenize,
                                  min_df=min_df,
                                  max_df=max_df,
                                  ngram_range=(1, 3),
                                  lowercase=False,  # done in tokenize function
                                  stop_words=get_stopwords_list())

    tfidf_model.fit(packages)

    u = UnStemmer.get()

    print("transforming data")

    tfidf_data = tfidf_model.transform(packages)

    print("saving model: ", config.TFIDFModels.description_model_2)
    tfidf_model.vocabulary_ = {" ".join(map(lambda x: u.resolve(x), k.split(" "))): v
                               for k, v in tfidf_model.vocabulary_.items()}
    pickle.dump(tfidf_model, open(config.TFIDFModels.description_model_2, "wb"))
    print("saved")

    print("saving data: ", config.TFIDFModels.description_data_2)
    save_data = {'ids': packages,
                 'data': tfidf_data}
    pickle.dump(save_data, open(config.TFIDFModels.description_data_2, "wb"))
    print("saved")


def metric_threshold():
    model, data = load()

    lengths = {th: [] for th in np.arange(0.05, 0.25, 0.005)}

    for doc in data['data']:
        d = doc.todense()
        for l in lengths.keys():
            lengths[l].append(d[d >= l].size)

    # find threshold for description discretization
    for l in sorted(lengths.keys()):
        print("%2.3f      mean=%3.1f      median=%3.1f" % (l, float(np.mean(lengths[l])), float(np.median(lengths[l]))))


def check():
    tfidf_model, save_data = load()

    info(tfidf_model, save_data['data'])

    print()

    sample_features = tfidf_model.get_feature_names()
    random.shuffle(sample_features)
    N = 200
    sample_features = sample_features[:N]

    for i in range(0, N, 5):
        print("%25s  %25s  %25s  %25s" % (sample_features[i],
                                              sample_features[i+1],
                                              sample_features[i+2],
                                              sample_features[i+3]))

    return tfidf_model, save_data


def load():
    tfidf_model = pickle.load(open(config.TFIDFModels.description_model_2, "rb"))
    save_data = pickle.load(open(config.TFIDFModels.description_data_2, "rb"))
    return tfidf_model, save_data


def cooccurrence():
    model, data = load()

    d = data['data']
    num_features = len(model.get_feature_names())

    occurrence = np.zeros((num_features, num_features))

    feature_names = model.get_feature_names()
    random.shuffle(feature_names)

    for ft in feature_names:
        E = PreTrainedEmbeddings.get()
        print()
        print(ft)
        print()
        print(E.nearest_words(ft)[:5])

if __name__ == "__main__":
    #create()
    #check()
    #metric_threshold()
    cooccurrence()
