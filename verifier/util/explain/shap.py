import datetime
import itertools
import pickle
import pprint
import random

import shap

import numpy as np
import keras as K
from collections import Counter

from sklearn.neighbors import NearestNeighbors

from verifier import config


def get_input_influences(input_tfidf_model, input_tfidf_vector,
                         output_tfidf_model, out_label_indices,
                         ml_model):
    if hasattr(output_tfidf_model, 'tfidf_model'):
        output_tfidf_model = output_tfidf_model.tfidf_model

    shap_neighborhood = create_shap_neighborhood(input_tfidf_vector, input_tfidf_model.tfidf_data, 500)

    explainer_input = (ml_model.layers[0].input, ml_model.layers[-1].output)
    explainer = shap.DeepExplainer(explainer_input, shap_neighborhood[1:], K.backend.get_session())
    shap_values, indexes = explainer.shap_values(shap_neighborhood[:1],
                                                 ranked_outputs=len(out_label_indices))

    #shap.summary_plot(shap_values, input_tfidf_model.tfidf_model.get_feature_names(),
    #                  max_display=10,
    #                  plot_type="violin",
    #                  class_names=[output_tfidf_model.get_feature_names()[i] for i in indexes.flatten().tolist()])

    influences = {}

    for shap_value_i, out_idx in zip(shap_values, indexes.flatten().tolist()):
        shap_value_i = shap_value_i.flatten()
        output_word = output_tfidf_model.get_feature_names()[out_idx]
        influences[output_word] = []
        inputs = np.argsort(-shap_value_i).flatten().tolist()[:100]
        for in_idx in inputs:
            input_word = input_tfidf_model.tfidf_model.get_feature_names()[in_idx]
            input_shap_value = shap_value_i[in_idx]
            if input_shap_value < 0.00001:
                break
            influences[output_word].append((input_word, input_shap_value))

    return influences


def create_shap_neighborhood(input_tfidf_vector, X, neighborhood_size, knn=False, n_step=5):
    if knn:

        neigh = NearestNeighbors(n_neighbors=1, n_jobs=-1)
        neigh.fit(X)

        indices = neigh.kneighbors(input_tfidf_vector,
                                   n_neighbors=neighborhood_size * n_step,
                                   return_distance=False).flatten()

        indices = indices[::n_step]

    else:
        rows, cols = X.nonzero()

        col_list_idx_relevant = np.argwhere(np.isin(cols, input_tfidf_vector.indices))
        rows_relevant = rows[col_list_idx_relevant].flatten().tolist()

        rows_relevant_sorted = list(Counter(rows_relevant).items())
        rows_relevant_sorted.sort(key=lambda x: x[1], reverse=True)

        indices = list(map(lambda x: x[0], rows_relevant_sorted))
        indices = indices[:neighborhood_size * n_step:n_step]

    return X[indices].toarray()

