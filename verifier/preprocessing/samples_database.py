import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import pprint
import random
import re

import langdetect
from langdetect.lang_detect_exception import LangDetectException

from verifier import config
from verifier.preprocessing.pretrained_embedding import PreTrainedEmbeddings


class SamplesDatabase:
    _instance = None
    FIELDS = ["title",
              "permissions",
              "category",
              "lang",
              "downloads",
              "description_raw",
              "description_num_tokens_glove",
              "description_num_tokens_word2vec",
              "meta_file",
              "apk_file",
              "cross_platform",
              "set"]

    _comparisons = {
        '>': lambda x, y: x > y,
        '<': lambda x, y: x < y,
        '>=': lambda x, y: x >= y,
        '<=': lambda x, y: x <= y,
        '==': lambda x, y: x == y,
        '!=': lambda x, y: x != y,
        'len>': lambda x, y: len(x) > y,
        'len<': lambda x, y: len(x) < y,
        'contains': lambda x, y: (y in x),
        'contains_insensitive': lambda x, y: (str(y).lower() in str(x).lower()),
    }

    @staticmethod
    def set_file(database_file):
        if SamplesDatabase._instance is not None:
            del SamplesDatabase._instance

        SamplesDatabase.get(database_file)

    @staticmethod
    def get(database_file=config.Samples.samples_database):
        if SamplesDatabase._instance is None:

            SamplesDatabase._instance = SamplesDatabase(database_file)
        return SamplesDatabase._instance

    def __init__(self, database_file=config.Samples.samples_database):
        self.database_file = database_file
        self.data = {}
        self.load()

    def read(self, package, key=None):
        if key:
            if key not in SamplesDatabase.FIELDS and not key.startswith("_"):
                raise RuntimeError("unknown key: %s" % key)

            if package:
                return self.data.get(package, {}).get(key, None)
            else:
                return [d.get(key) for d in self.data.values()]
        elif not key and package:
            return self.data.get(package, {})
        else:
            raise RuntimeError("no package and/or key")

    def write(self, package, **kwargs):
        if package not in self.data and (not kwargs or "create" not in kwargs):
            raise RuntimeError("package %s not found" % package)

        self.data[package] = self.data.get(package, {})

        for key, value in kwargs.items():
            if key == "create": continue

            if key in SamplesDatabase.FIELDS or \
               key.startswith("_"):  # temporary
                self.data[package][key] = value
            else:
                raise RuntimeError("unknown key: ", key)

    def filter(self, *criterions, return_value='package'):
        for criterion in criterions:
            if type(criterion) is not tuple or len(criterion) != 3:
                raise RuntimeError("criterion is not a tuple of length 3")

        filtered = []

        for package, data in self.data.items():
            match = True
            for criterion in criterions:
                sample_attribute, comparison_char, compare_value = criterion
                if sample_attribute not in SamplesDatabase.FIELDS:
                    raise RuntimeError("unknown key: %s" % sample_attribute)

                if not SamplesDatabase._comparisons[comparison_char](data.get(sample_attribute), compare_value):
                    match = False
                    break

            if match:
                if return_value == 'package':
                    filtered.append(package)
                elif return_value in SamplesDatabase.FIELDS:
                    filtered.append(data.get(return_value))
                else:
                    raise RuntimeError('unknown desired return value')

        return filtered

    def has_package(self, package):
        return package in self.data

    def save(self):
        pickle.dump(self.data, open(self.database_file, "wb"))

    def load(self):
        if not os.path.exists(self.database_file):
            print("samples database does not exist, please create first!")
        else:
            self.data = pickle.load(open(self.database_file, "rb"))


def build_database_from_meta():
    db = SamplesDatabase.get()
    db.data = {}

    for idx, file_name in enumerate(os.listdir(config.Samples.app_metadata)):
        file = os.path.join(config.Samples.app_metadata, file_name)
        if not os.path.isfile(file) or not file.endswith(".json"):
            continue

        try:
            meta = json.load(open(file, "r", encoding="utf8"))

            package = meta['docid']
            title = meta['title']
            description_raw = re.sub(r'(</?[A-z0-9]+( [a-z]+=\"\S+\")*>)', ' ', meta['description_html'])
            permissions = meta['details']['app_details'].get('permission', [])
            downloads = int(meta['details']['app_details']['num_downloads'].replace(",", "").replace("+", ""))
            category = meta['details']['app_details']['app_category'][0]
            try:
                lang = langdetect.detect(description_raw)
            except LangDetectException:
                lang = '?'

            db.write(package,
                     title=title,
                     description_raw=description_raw,
                     permissions=permissions,
                     downloads=downloads,
                     category=category,
                     lang=lang,
                     create=True)

            if idx % 1000 == 0:
                print("processed:   ", idx)

        except json.decoder.JSONDecodeError as jex:
            print("error parsing: ", file, str(jex))

    db.save()


def add_cross_platform_information():
    db = SamplesDatabase.get()

    for package in db.filter():
        db.write(package, cross_platform=False)

    cross_platform_packages = open(config.Samples.list_of_cross_platform_apps, "r").read().split("\n")
    for package in cross_platform_packages:
        package = package.rsplit("-", maxsplit=1)[0]
        if not db.has_package(package):
            print("does not exist: ", package)
            continue
        db.write(package, cross_platform=True)

    n_are = len(db.filter(('cross_platform', '==', True)))
    n_are_not = len(db.filter(('cross_platform', '==', False)))

    print("%d cross-platform apps, which is %d%%" % (n_are, n_are * 100 / n_are_not))

    db.save()


def divide_into_train_and_test_sets():
    db = SamplesDatabase.get()

    packages_test = []

    all_packages_en = db.filter(('lang', '==', 'en'))
    random.shuffle(all_packages_en)
    all_packages_en.sort(key=lambda p: db.read(p, 'downloads'), reverse=True)

    # add every 10th sample of the list sorted by downloads until enough
    for i in range(0, 5*config.Samples.num_for_test_set, 3):
        packages_test.append(all_packages_en[i])
        if len(packages_test) == 1000:
            break

    for p in all_packages_en:
        if p in packages_test:
            db.write(p, set='test')
        else:
            db.write(p, set='train+valid')

    print("# total:        %8d" % len(db.filter()))
    print("# train+valid:  %8d" % (len(all_packages_en) - len(packages_test)))
    print("# test:         %8d" % len(packages_test))

    db.save()


def save_test_set_list():
    db = SamplesDatabase.get()
    test_packages = db.filter(('set', '==', 'test'))
    pprint.pprint(test_packages)
    with open(config.Samples.test_set_save_file, "w") as fp:
        fp.write("\n".join(test_packages))


def info():

    db = SamplesDatabase.get()

    n = len(db.filter(
        ('lang', '==', 'en'),
        #('cross_platform', '==', False),
        #('description_raw', 'len>', 2)
    ))

    print("# en = ", n)

    em = PreTrainedEmbeddings.get()
    for package in db.filter(('set', '==', 'test')):
        print("--", package)
        d = db.read(package, 'downloads')
        print(d)



if __name__ == "__main__":
    #build_database_from_meta()
    #add_cross_platform_information()
    #divide_into_train_and_test_sets()
    #save_test_set_list()
    #set_embedded_tokens_length()
    #embedded_tokens_length_stats()

    info()