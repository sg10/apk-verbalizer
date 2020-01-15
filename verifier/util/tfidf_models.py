import numpy as np
from scipy.sparse import csr_matrix


def get_top_terms(tfidf_data, model, top_k=0, add_value=True):
    features = model.get_feature_names()

    if type(tfidf_data) is csr_matrix:
        terms = [(ind[1], value) for ind, value in tfidf_data.todok().items()]
        terms.sort(key=lambda el: el[1], reverse=True)
        terms = [(features[term[0]], term[1]) for term in terms]
    else:
        indices_sorted = np.argsort(-tfidf_data, axis=None).flatten().tolist()
        if add_value:
            terms = [(features[d], tfidf_data[d]) for d in indices_sorted if tfidf_data[d] > 0]
        else:
            terms = [features[d] for d in indices_sorted if tfidf_data[d] > 0]

    if top_k > 0:
        return terms[:top_k]

    return terms
