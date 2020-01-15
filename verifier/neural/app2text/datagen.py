import pickle
import pprint

import keras
import numpy as np

from verifier import config
from verifier.preprocessing.samples_database import SamplesDatabase
from verifier.util.tfidf_model_creator import TfIdfModelCreator


class TFIDFGenerator(keras.utils.Sequence):

    def __init__(self,
                 tfidf_input_model_file,
                 tfidf_input_data_file,
                 package_names,
                 shuffle=True,
                 verbose=True,
                 batch_size=None):
        self.batch_size = batch_size or config.TFIDFClassifier.batch_size
        self.shuffle = shuffle

        self.descriptions = {'model': pickle.load(open(config.TFIDFModels.description_model_2, 'rb')),
                             'data': pickle.load(open(config.TFIDFModels.description_data_2, 'rb'))}

        self.inputs = TfIdfModelCreator(model_file=tfidf_input_model_file,
                                        data_file=tfidf_input_data_file)
        self.inputs.load(data=True, model=True)

        self.num_inputs = len(self.inputs.tfidf_model.get_feature_names())
        self.num_outputs = len(self.descriptions['model'].get_feature_names())

        # only use packages that occur in input data, output data and the list
        # that the generator was called with
        self.package_names = list(set(package_names)
                                  .intersection(set(self.inputs.doc_ids))
                                  .intersection(set(self.descriptions['data']['ids'])))

        self.data_samples = {}  # {'package name 1': {'input': <data>, 'output': <data>}, ...}
        for package_name in self.package_names:
            # remove all samples that don't contain enough terms
            sample_input = self.inputs.tfidf_data[self.inputs.doc_ids.index(package_name)]
            sample_output = self.descriptions['data']['data'][self.descriptions['data']['ids'].index(package_name)]

            if sample_input.getnnz() < config.TFIDFModels.min_terms_per_doc or \
                sample_output.getnnz() < config.TFIDFModels.min_terms_per_doc:
                continue

            self.data_samples[package_name] = {
                'input': sample_input,
                'output': sample_output
            }

        self.package_names = list(self.data_samples.keys())

        self.db = SamplesDatabase.get()

        if verbose:
            print("Generator contains %d samples (network: %d --> %d)" % (len(self.package_names),
                                                                          self.get_num_inputs(),
                                                                          self.get_num_outputs()))

        self.on_epoch_end()

    def get_num_inputs(self):
        return self.num_inputs

    def get_num_outputs(self):
        return self.num_outputs

    def __len__(self):
        return int(np.floor(len(self.package_names) / self.batch_size))

    def __getitem__(self, batch_index):
        X, y, _ = self.get_batch_and_details(batch_index, get_packages=False)
        return X, y

    def get_batch_and_details(self, batch_index, get_packages=True):
        sample_packages = self.package_names[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]
        X, y, package_names = self.__data_generation(sample_packages, get_packages)
        return X, y, package_names

    def __data_generation(self, sample_packages, get_packages=False):
        # Initialization
        num_samples = self.batch_size
        X = np.zeros((num_samples, self.get_num_inputs()), dtype=np.float32)
        y = np.zeros((num_samples, self.get_num_outputs()))
        package_names = []

        # Generate data
        for i_row, package_name in enumerate(sample_packages):
            sample = self.data_samples[package_name]
            X[i_row][:sample['input'].shape[1]] = np.array(sample['input'].todense())
            # if X[i_row].max() > 0:
            #    X[i_row] /= X[i_row].max()
            y[i_row] = np.array(sample['output'].todense())
            # if y[i_row].max() > 0:
            #    y[i_row] /= y[i_row].max()

            if get_packages:
                package_names.append(package_name)

        return X, y, package_names

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.package_names)
