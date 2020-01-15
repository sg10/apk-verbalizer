import keras
import numpy as np
import os
import pprint
import random

from builtins import reversed
from keras.optimizers import Adam, SGD
from sklearn.model_selection import KFold
from tensorflow._api.v1 import logging

from verifier import config
from verifier.neural.app2text import model
from verifier.neural.app2text.callbacks import TrainingPredictionsCallback
from verifier.neural.app2text.datagen import TFIDFGenerator
from verifier.neural.app2text.model import model_for_tf_idfs
from verifier.neural.app2text.preprocessing.descriptions import meta_data_description_tokenize
from verifier.neural.app2text.preprocessing.strings_and_ids import CountDictIterator
from verifier.preprocessing.samples_database import SamplesDatabase
from verifier.util import metrics
from verifier.util.metrics import get_tfidf_parameterized
from verifier.util.tfidf_model_creator import VectorizerDoc, TfIdfModelCreator
from verifier.util.train import train_summary


def train(verbose=True, reduced_dataset=False):
    if not verbose:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '2'
        logging.set_verbosity(logging.ERROR)

    db = SamplesDatabase.get()
    package_names = db.filter(('lang', '==', 'en'),
                              ('set', '==', 'train+valid'),
                              ('description_raw', 'len>', 100),
                              ('cross_platform', '==', False))
    if reduced_dataset:
        random.seed(42)
        random.shuffle(package_names)
        package_names = package_names[:11000]
        random.seed(None)
    else:
        random.shuffle(package_names)

    # package_names = package_names[:3000]

    if verbose:
        print("# apps for training (db): ", len(package_names))

    performance = []

    if config.TFIDFClassifier.ModelStrings.enabled:
        performance = train_tfidf_classifier(config.TFIDFClassifier.ModelStrings,
                                             package_names,
                                             config.TFIDFModels.code_stringres_data,
                                             config.TFIDFModels.code_stringres_model,
                                             config.TrainedModels.app2text_stringres,
                                             verbose)

    if config.TFIDFClassifier.ModelResourceIds.enabled:
        performance = train_tfidf_classifier(config.TFIDFClassifier.ModelResourceIds,
                                             package_names,
                                             config.TFIDFModels.code_ids_data,
                                             config.TFIDFModels.code_ids_model,
                                             config.TrainedModels.app2text_ids,
                                             verbose)

    if config.TFIDFClassifier.ModelMethods.enabled:
        performance = train_tfidf_classifier(config.TFIDFClassifier.ModelMethods,
                                             package_names,
                                             config.TFIDFModels.code_methods_data,
                                             config.TFIDFModels.code_methods_model,
                                             config.TrainedModels.app2text_methods,
                                             verbose)

    return performance


def train_tfidf_classifier(model_config_class, package_names, tfidf_data, tfidf_model, model_save_file, verbose):
    keras.backend.clear_session()

    model.model_config_class = model_config_class

    if verbose:
        print("---------------------------------------------------")
        print(tfidf_model)
        print(model_save_file)
        print("---------------------------------------------------")

    k_fold_split = KFold(n_splits=int(1 / config.TFIDFClassifier.validation_split),
                         shuffle=True)
    train_indices, valid_indices = next(k_fold_split.split(package_names))

    train_generator = TFIDFGenerator(tfidf_input_model_file=tfidf_model,
                                     tfidf_input_data_file=tfidf_data,
                                     package_names=np.array(package_names)[train_indices].tolist(),
                                     verbose=verbose)
    valid_generator = TFIDFGenerator(tfidf_input_model_file=tfidf_model,
                                     tfidf_input_data_file=tfidf_data,
                                     package_names=np.array(package_names)[valid_indices].tolist(),
                                     batch_size=256,
                                     verbose=verbose)

    callbacks = [
        keras.callbacks.EarlyStopping(  # monitor='val_mean_squared_error',
            monitor='val_f_04',
            patience=config.TFIDFClassifier.early_stopping_patience,
            min_delta=0.0025,
            mode='max',
            verbose=verbose),
        keras.callbacks.ModelCheckpoint(filepath=model_save_file,
                                        # monitor='val_mean_squared_error',
                                        # monitor='val_tfidf_fbeta',
                                        monitor='val_f_04',
                                        mode='max',
                                        save_best_only=True,
                                        verbose=verbose)
    ]
    if verbose:
        callbacks.append(
            TrainingPredictionsCallback(valid_generator,
                                        n_print=10,
                                        n_words_max=25)
        )

    keras_model = model_for_tf_idfs(train_generator.get_num_inputs(),
                                    train_generator.get_num_outputs())
    keras_model.compile(optimizer=Adam(lr=model_config_class.learning_rate, amsgrad=True),
                        loss='mse',
                        metrics=get_metrics_for_a2p_model())

    if verbose:
        keras_model.summary()

    keras_model.fit_generator(train_generator,
                              validation_data=valid_generator,
                              epochs=config.TFIDFClassifier.max_train_epochs,
                              callbacks=callbacks,
                              verbose=verbose)

    keras_model.load_weights(model_save_file)

    eval_values = keras_model.evaluate_generator(valid_generator, verbose=False)

    if verbose:
        train_summary(eval_values, keras_model, valid_generator)

    return list(zip(keras_model.metrics_names, eval_values))


def get_metrics_for_a2p_model():
    return get_tfidf_parameterized(0.5, 0.03) + \
           get_tfidf_parameterized(0.5, 0.04) + \
           get_tfidf_parameterized(0.5, 0.05) + \
           get_tfidf_parameterized(0.5, 0.06) + \
           get_tfidf_parameterized(0.5, 0.07) + \
           get_tfidf_parameterized(0.5, 0.08)


def final():
    print("App2Text training of final arch started")
    train(True, False)


def random_search():
    print("App2Text random search started")

    first = True

    model_configs = [config.TFIDFClassifier.ModelResourceIds,
                     config.TFIDFClassifier.ModelStrings,
                     config.TFIDFClassifier.ModelMethods]

    for i_model in [0]:
        # for i_model in reversed(range(3)):
        model_configs[0], model_configs[1], model_configs[2] = False, False, False
        model_configs[i_model] = True

        layer_configs = []
        for _ in range(50):
            num_layers = random.randint(1, 3)
            l = [random.randint(1000, 15000) // num_layers for _ in range(num_layers)]
            d = random.randint(0, 40) / 100
            # layer_configs.append([l, d])
            layer_configs.append([l, d])

        for layer_config, dropout in layer_configs:
            model_configs[i_model] = layer_config
            model_configs[i_model] = dropout

            try:
                arch_performance = train(False, True)
            except Exception:
                arch_performance = ["error"]
            if first:
                header = ["model", "arch"] + [name for name, _ in arch_performance] + ["dropout"]
                print(";".join(header))
                first = False

            arch_str = ",".join(["%d" % l for l in layer_config])
            row = ["model %d" % i_model, arch_str] + ["%.7f" % value for _, value in arch_performance] + [
                "%.2f" % dropout]
            print(";".join(row))


if __name__ == "__main__":
    # need to load into main scope; for unpickling
    assert (VectorizerDoc is not None)
    assert (CountDictIterator is not None)
    assert (meta_data_description_tokenize is not None)
    # random_search()
    final()
    print("done.")
