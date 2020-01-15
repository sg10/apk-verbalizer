import argparse
import base64
import json
import keras as K
import logging
import numpy as np
import os
import pprint
import random

from datetime import datetime

from androguard.core.androconf import show_logging
from io import BytesIO
from PIL import Image
from androguard.core.bytecodes.apk import APK

from verifier import config
from verifier.neural.app2text import model
from verifier.neural.app2text.model import model_for_tf_idfs
from verifier.neural.app2text.preprocessing.descriptions import meta_data_description_tokenize
from verifier.neural.app2text.preprocessing.strings_and_ids import CountDictIterator
from verifier.preprocessing.samples_database import SamplesDatabase
from verifier.preprocessing.string_resources import APKStringResources, APKStringResourceExtractor
from verifier.test.report_saver import ReportSaver
from verifier.util.android_slicer.app_api_call_analyzer import apk_find_api_calls
from verifier.util.explain.shap import get_input_influences
from verifier.util.strings import remove_html
from verifier.util.text_processing import tokenize_text
from verifier.util.tfidf_model_creator import TfIdfModelCreator, VectorizerDoc
from verifier.util.tfidf_models import get_top_terms


class APKRunner:
    def __init__(self, report_folder, apk=None, txt=None, package_name=None):
        self.report_folder = report_folder
        self.apk_file = apk
        self.text_file = txt
        self.n_nearest_apps = 15
        self.n_top_terms = 25
        self.load_from_preprocessed = True  # only checks package name, not version, hash, etc.

        self.app_name = "Unknown App | " + datetime.today().strftime('%Y-%m-%d %H:%M')
        self.text = None
        self.package_name = package_name
        self.result_file_name = None
        self._preloaded_input_tfidf_model = {}
        self._preloaded_descriptions_tfidf_model = None

        self.report_saver = ReportSaver(report_folder=report_folder)

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        show_logging(logging.ERROR)  # androguard

    def run(self):
        if self.apk_file:
            self.get_apk_info()
        else:
            self.get_fallback_app_info()

        if self.text_file:
            self.read_text()
        else:
            self.get_fallback_description()

        print("-" * 80)
        print(self.app_name or self.apk_file)
        print("-" * 80)

        self.process_a2t()
        self.report_saver.save()

    def get_apk_info(self):
        apk = APK(self.apk_file)
        app_icon_file = apk.get_app_icon()
        app_icon_data = apk.get_file(app_icon_file)

        size = (256, 256)

        buffered = BytesIO()
        im = Image.open(BytesIO(app_icon_data))
        im = im.resize(size, Image.ANTIALIAS)
        im.save(buffered, "PNG")

        app_icon_b64 = "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode('utf-8')

        self.package_name = apk.get_package()
        self.app_name = apk.get_app_name()

        self.report_saver.package_name = self.package_name
        self.report_saver.app_name = self.app_name
        self.report_saver.version = apk.get_androidversion_code()
        self.report_saver.app_icon = app_icon_b64

    def get_fallback_app_info(self):
        db = SamplesDatabase.get()
        if self.package_name is None:
            raise RuntimeError("no package name!")
        self.app_name = db.read(self.package_name, 'title')
        self.report_saver.package_name = self.package_name
        self.report_saver.app_name = self.app_name
        self.report_saver.version = ""
        self.report_saver.app_icon = None

    def get_fallback_description(self):
        self.text = SamplesDatabase.get().read(self.package_name, 'description_raw')

    def process_a2t(self):
        if self.text:
            self.create_tfidf_from_description()

        input_values, words_pred = {}, {}

        # -- ids
        #tfidf_model, tfidf_vector = self.create_input_tfidf(self.package_name, 'ids')
        #input_values['ids'], words_pred['ids'] = \
        #    self.apply_tfidf_ml_model(config.TrainedModels.app2text_ids, tfidf_model, tfidf_vector)
        input_values['ids'], words_pred['ids'] = {}, {}

        # -- strings
        #tfidf_model, tfidf_vector = self.create_input_tfidf(self.package_name, 'strings')
        #input_values['strings'], words_pred['strings'] = \
        #    self.apply_tfidf_ml_model(config.TrainedModels.app2text_stringres, tfidf_model, tfidf_vector)
        input_values['strings'], words_pred['strings'] = {}, {}

        # -- methods
        tfidf_model, tfidf_vector = self.create_input_tfidf(self.package_name, 'methods')
        input_values['methods'], words_pred['methods'] = \
            self.apply_tfidf_ml_model(config.TrainedModels.app2text_methods, tfidf_model, tfidf_vector)

        pprint.pprint(list(sorted(words_pred['methods'].items(), key=lambda x: x[1], reverse=True))[:5])

        self.report_saver.a2t['input_values'] = input_values
        self.report_saver.a2t['words_pred'] = words_pred
        self.report_saver.a2t['text_actual'] = self.text

    def create_input_tfidf(self, package_name, input_mode):
        if input_mode in self._preloaded_input_tfidf_model:
            tfidf_model = self._preloaded_input_tfidf_model[input_mode]
        else:
            if input_mode == 'methods':
                model.model_config_class = config.TFIDFClassifier.ModelMethods
                tfidf_model = TfIdfModelCreator(config.TFIDFModels.code_methods_model,
                                                config.TFIDFModels.code_methods_data)
            elif input_mode == 'ids':
                model.model_config_class = config.TFIDFClassifier.ModelStrings
                tfidf_model = TfIdfModelCreator(config.TFIDFModels.code_ids_model,
                                                config.TFIDFModels.code_ids_data)
            elif input_mode == 'strings':
                model.model_config_class = config.TFIDFClassifier.ModelResourceIds
                tfidf_model = TfIdfModelCreator(config.TFIDFModels.code_stringres_model,
                                                config.TFIDFModels.code_stringres_data)
            else:
                raise RuntimeError("unknown input mode")

            self._preloaded_input_tfidf_model[input_mode] = tfidf_model

        tfidf_vector = None

        tfidf_model.load(data=True, model=True)
        if self.load_from_preprocessed and package_name in tfidf_model.doc_ids:
            print("loaded from preprocessed set! (%s)" % input_mode)
            tfidf_vector = tfidf_model.tfidf_data[tfidf_model.doc_ids.index(package_name)]
        else:
            if input_mode == 'methods':
                api_calls_and_count = apk_find_api_calls(self.apk_file)
                api_calls_tokens = []
                for method, count in api_calls_and_count.items():
                    api_calls_tokens += [method] * count
                tfidf_vector = tfidf_model.transform_sample(api_calls_tokens)
            else:
                extractor = APKStringResourceExtractor(self.apk_file)
                extractor.run()
                string_resources = APKStringResources(ids=extractor.get_id_names(),
                                                      res_str=extractor.get_resource_strings(),
                                                      code_str=extractor.get_code_strings())
                if input_mode == 'ids':
                    id_tokens = string_resources.get_ids_cleaned()
                    id_tokens = [t for sublist in id_tokens for t in sublist]
                    tfidf_vector = tfidf_model.transform_sample(id_tokens)
                elif input_mode == 'strings':
                    str_tokens = string_resources.get_all_strings_cleaned()
                    str_tokens = [t for sublist in str_tokens for t in sublist]
                    tfidf_vector = tfidf_model.transform_sample(str_tokens)

        return tfidf_model, tfidf_vector

    def apply_tfidf_ml_model(self, ml_model_file, input_tfidf_model, input_tfidf_vector):
        if not self._preloaded_descriptions_tfidf_model:
            descriptions_tfidf_model = TfIdfModelCreator(config.TFIDFModels.description_model_2,
                                                         config.TFIDFModels.description_data_2)
            descriptions_tfidf_model.load(data=True, model=True)
            self._preloaded_descriptions_tfidf_model = descriptions_tfidf_model
        else:
            descriptions_tfidf_model = self._preloaded_descriptions_tfidf_model

        print("loading ML model")

        ml_model = model_for_tf_idfs(len(input_tfidf_model.tfidf_model.get_feature_names()),
                                     len(descriptions_tfidf_model.tfidf_model.get_feature_names()))

        ml_model.compile(optimizer='adam',
                         loss='mse',
                         metrics=['mse'])

        ml_model.load_weights(ml_model_file)

        y = ml_model.predict([input_tfidf_vector.todense()])

        output_words = get_top_terms(y[0], descriptions_tfidf_model.tfidf_model, top_k=self.n_top_terms)
        output_words = {word: float(value) for word, value in output_words}
        descriptions_tfidf_indices = np.argsort(-y[0]).flatten().tolist()[:self.n_top_terms]

        influences = {}
        influences = get_input_influences(input_tfidf_model, input_tfidf_vector,
                             descriptions_tfidf_model, descriptions_tfidf_indices,
                             ml_model)

        K.backend.clear_session()

        return influences, output_words

    def read_text(self):
        text = open(self.text_file, "r", encoding="utf-8").read()
        try:
            json_data = json.loads(text)
            text = remove_html(json_data['description_html'])
        except ValueError:
            pass

        self.text = text

    def create_tfidf_from_description(self):
        tokens = tokenize_text(self.text, clean_and_stem=True)
        tfidf_model = TfIdfModelCreator(model_file=config.TFIDFModels.description_model,
                                        data_file=None)
        tfidf_model.load(model=True)
        tfidf_vector = tfidf_model.transform_sample(tokens)

        words = get_top_terms(tfidf_vector, tfidf_model.tfidf_model, top_k=self.n_top_terms)
        words = {word: float(value) for word, value in words}

        self.report_saver.a2t['words_actual'] = words


def is_a_dir(s):
    if os.path.isdir(s):
        return s
    else:
        raise NotADirectoryError(s)


def process_apk(file_txt, file_apk):
    parser = argparse.ArgumentParser(description='Android App Description Inference Tool')

    parser.add_argument('--apk',
                        help='Android APK file')
    parser.add_argument('--txt',
                        help='Description text file')

    parser.add_argument('--out',
                        help='folder for generated report')

    args = parser.parse_args(["--txt", file_txt,
                              "--apk", file_apk,
                              "--out", "/data/reports"])

    if args.txt is None or not os.path.isfile(args.txt):
        print("File does not exist: ", args.txt or "<TXT>")
        parser.print_help()
        return
    if args.apk is None or not os.path.isfile(args.apk):
        print("File does not exist: ", args.apk or "<APK>")
        parser.print_help()
        return

    if not args.txt or not args.apk:
        print("Need at least either APK file or description text file.")
        parser.print_help()
        return

    runner = APKRunner(args.out, apk=args.apk, txt=args.txt)
    runner.run()


def from_command_line():
    files = list(os.listdir("/data/test_apps/"))
    try:
        files_exist = list(os.listdir("/data/reports/apps/"))
    except:
        files_exist = []
    random.shuffle(files)
    # files = ['nu.mine.tmyymmt.aflashlight-97.apk']
    # test_packs = SamplesDatabase.get().filter(('set', '==', 'test'))
    # pprint.pprint(test_packs)
    # pprint.pprint(files)
    # asd
    for filename_apk in files:

        file_apk = os.path.join("/data/test_apps/", filename_apk)
        filename_json = filename_apk.rsplit("-", maxsplit=1)[0] + ".json"
        file_txt = os.path.join("/data/samples/metadata/", filename_json)

        # if filename_json in files_exist:
        #    print("exist: ", filename_json)
        #    continue

        try:
            process_apk(file_txt, file_apk)
        except Exception as e:
            print(e)


if __name__ == "__main__":
    assert CountDictIterator
    assert VectorizerDoc
    assert meta_data_description_tokenize

    from_command_line()

