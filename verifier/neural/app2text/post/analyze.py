import pickle

import keras
import numpy as np
import os
from keras.optimizers import Adam
from tensorflow._api.v1 import logging

from verifier import config
from verifier.neural.app2text import model as m
from verifier.neural.app2text.datagen import TFIDFGenerator
from verifier.neural.app2text.model import model_for_tf_idfs
from verifier.neural.app2text.preprocessing.descriptions import meta_data_description_tokenize
from verifier.neural.app2text.preprocessing.strings_and_ids import CountDictIterator
from verifier.neural.app2text.training import get_metrics_for_a2p_model
from verifier.preprocessing.samples_database import SamplesDatabase
from verifier.util import metrics
from verifier.util.train import train_summary


def find_good_examples_in_test_set():
    models = [
        (config.TFIDFClassifier.ModelStrings,
         config.TFIDFModels.code_stringres_data,
         config.TFIDFModels.code_stringres_model,
         config.TrainedModels.app2text_stringres),
        (config.TFIDFClassifier.ModelResourceIds,
         config.TFIDFModels.code_ids_data,
         config.TFIDFModels.code_ids_model,
         config.TrainedModels.app2text_ids),
        (config.TFIDFClassifier.ModelMethods,
         config.TFIDFModels.code_methods_data,
         config.TFIDFModels.code_methods_model,
         config.TrainedModels.app2text_methods)
    ]

    for config_class, tfidf_data, tfidf_model, model_save_file in models:
        result = test_predictions_for_submodel(config_class, tfidf_data, tfidf_model, model_save_file)
        continue

        top_packages = result['package_names'].copy()
        top_packages.sort(key=lambda p: result['metrics'][result['package_names'].index(p)][1],
                          reverse=True)  # via tfidf-fbeta-score

        for p in top_packages[10:]:
            print_sample_with_prediction(result, p)
        break


def print_sample_with_prediction(result, package_name):
    idx = result['package_names'].index(package_name)

    model = pickle.load(open(config.TFIDFModels.description_model_2, 'rb'))

    print(SamplesDatabase.get().read(package_name, 'title'))
    print("  ", list(map(lambda v: "%d%%" % int(100 * v), result['metrics'][idx][1:])))

    top_actual_desc_indices = np.argsort(-np.array(result['y'][idx]))[:10].tolist()
    top_pred_desc_indices = np.argsort(-np.array(result['predictions'][idx]))[:10].tolist()

    tokens_actual = [model.get_feature_names()[i] for i in top_actual_desc_indices]
    values_actual = [result['y'][idx][i] for i in top_actual_desc_indices]

    tokens_predicted = [model.get_feature_names()[i] for i in top_pred_desc_indices]
    values_predicted = [result['predictions'][idx][i] for i in top_pred_desc_indices]

    print(" actual:")
    for token, value in zip(tokens_actual, values_actual):
        if value >= metrics._tfidf_discretize_threshold_descriptions:
            print("   %.2f  %s" % (value, token))
    print(" predicted:")
    for token, value in zip(tokens_predicted, values_predicted):
        if value >= metrics._tfidf_discretize_threshold_descriptions:
            print("   %.2f  %s" % (value, token))

    print()


def test_predictions_for_submodel(config_class, tfidf_data, tfidf_model, model_save_file):
    print("-" * 50)
    print("-    ", config_class.__name__)
    print("-" * 50)

    m.model_config_class = config_class

    db = SamplesDatabase.get()
    package_names = db.filter(('lang', '==', 'en'),
                              ('set', '==', 'test'),
                              ('description_raw', 'len>', 100),
                              ('cross_platform', '==', False))

    keras.backend.clear_session()

    test_generator = TFIDFGenerator(tfidf_input_model_file=tfidf_model,
                                    tfidf_input_data_file=tfidf_data,
                                    package_names=package_names,
                                    batch_size=128,
                                    verbose=True)

    model = model_for_tf_idfs(test_generator.get_num_inputs(),
                              test_generator.get_num_outputs())
    model.compile(optimizer=Adam(lr=config_class.learning_rate, amsgrad=True),
                  loss='mse',
                  metrics=get_metrics_for_a2p_model())

    model.load_weights(model_save_file)

    eval_values = model.evaluate_generator(test_generator, verbose=False)
    train_summary(eval_values, model, test_generator)

    result = {
        'X': [],
        'y': [],
        'package_names': [],
        'metrics': [],
        'predictions': []
    }

    for batch_index in range(len(test_generator)):
        X, y, package_names = test_generator.get_batch_and_details(batch_index, get_packages=True)
        metrics = [model.evaluate(X[i:i + 1], y[i:i + 1], verbose=False) for i in range(X.shape[0])]
        predictions = model.predict_on_batch(X)
        result['X'] += X.tolist()
        result['y'] += y.tolist()
        result['package_names'] += package_names
        result['metrics'] += metrics
        result['predictions'] += predictions.tolist()

    return result


if __name__ == "__main__":
    assert meta_data_description_tokenize
    assert CountDictIterator
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '2'
    logging.set_verbosity(logging.ERROR)
    find_good_examples_in_test_set()
