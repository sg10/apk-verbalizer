import numpy as np
import keras
import tensorflow as tf

from keras.layers import *
from sklearn.metrics.pairwise import *

from verifier.preprocessing.samples_database import SamplesDatabase
from verifier.util.metrics import tfidf_accuracy, tfidf_metrics
from verifier.util.tfidf_models import get_top_terms


class TrainingPredictionsCallback(keras.callbacks.Callback):

    def __init__(self, generator, n_print, n_words_max):
        super().__init__()

        self.generator = generator
        self.desc_tfidf_model = self.generator.descriptions['model']

        self.db = SamplesDatabase.get()
        self.n_words_max = n_words_max
        self.n_print = n_print

    def on_epoch_end(self, epoch, logs=None):
        self.print_samples()

    def print_samples(self):
        print("----- PREDICTION -----")

        n_print = self.n_print
        n_features = len(self.desc_tfidf_model.get_feature_names())

        y_true_single = tf.placeholder(dtype=tf.float32, shape=(1, n_features))
        y_pred_single = tf.placeholder(dtype=tf.float32, shape=(1, n_features))
        metric_calc = tfidf_metrics(y_true_single, y_pred_single)

        for b in range(len(self.generator)):

            X, y, packages = self.generator.get_batch_and_details(b)
            p = self.model.predict(X)

            for i in range(X.shape[0]):
                if self.db.read(packages[i], 'downloads') < 1e5:
                    continue

                precision, recall, f1_score = K.get_session().run(feed_dict={y_true_single: y[i:i + 1], y_pred_single: p[i:i + 1]},
                                                    fetches=metric_calc)
                precision = 0 if np.isnan(precision) else precision * 100
                recall = 0 if np.isnan(recall) else recall * 100
                f1_score = 0 if np.isnan(f1_score) else f1_score * 100

                print(self.db.read(packages[i], 'title'), "  --  ", "f1: %3d%%, precision %3d%%, recall %3d%%" %
                      (f1_score, precision, recall))

                actual_words = get_top_terms(y[i], self.desc_tfidf_model, self.n_words_max, add_value=False)
                print("   actual:        ", "  ".join(actual_words))

                pred_words = get_top_terms(p[i], self.desc_tfidf_model, self.n_words_max, add_value=True)
                pred_words_str = ['%s:%.3f' % (word, value) for word, value in pred_words]
                print("   predict:       ", "  ".join(pred_words_str))

                n_print -= 1
                if n_print <= 0:
                    break

            if n_print <= 0:
                break
