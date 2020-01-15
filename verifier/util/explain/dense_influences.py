import numpy as np
import tensorflow as tf
import keras.backend as K
from scipy.sparse import csr_matrix


class SignificantInputWords:

    def __init__(self, model, input_tfidf_vectorizer):
        self.model = model
        self.helper_session = K.get_session() #tf.InteractiveSession()
        self.gradients_func = None
        self.input_tfidf_vectorizer = input_tfidf_vectorizer

    def get_significant_words(self, X, y):
        if type(X) is csr_matrix:
            X = X.todense()

        if np.count_nonzero(X) == 0 or np.count_nonzero(y) == 0:
            return []

        if self.gradients_func is None:
            output_tensor = self.model.output
            list_of_train_vars = self.model.trainable_weights
            self.gradients_func = K.gradients(output_tensor, list_of_train_vars)

        self.helper_session.run(tf.initialize_all_variables())

        evaluated_gradients = self.helper_session.run(self.gradients_func,
                                                      feed_dict={self.model.input: X})

        mean_gradients_first_layer = np.mean(evaluated_gradients[0], axis=1)
        mean_gradients_first_layer = np.clip(mean_gradients_first_layer,
                                             a_min=0.,
                                             a_max=9999.)
        mean_gradients_first_layer = mean_gradients_first_layer / np.max(mean_gradients_first_layer)

        top_indices = np.argsort(-mean_gradients_first_layer).tolist()

        words = []

        for ind in top_indices:
            value = mean_gradients_first_layer[ind]
            if value <= 0 or np.isnan(value):
                continue
            word = self.input_tfidf_vectorizer.get_feature_names()[ind]
            words.append((word, float(value)))

        return words
