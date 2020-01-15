import json
import keras as K
import numpy as np
import os
import random

from verifier import config
from verifier.neural.app2text import model
from verifier.neural.app2text.datagen import TFIDFGenerator
from verifier.neural.app2text.model import model_for_tf_idfs
from verifier.neural.app2text.preprocessing.descriptions import meta_data_description_tokenize
from verifier.neural.app2text.preprocessing.strings_and_ids import CountDictIterator
from verifier.neural.app2text.training import get_metrics_for_a2p_model
from verifier.preprocessing.samples_database import SamplesDatabase
from verifier.preprocessing.string_resources import APKStringResources, APKStringResourceExtractor
from verifier.test.report_saver import ReportSaver
from verifier.util.android_slicer.app_api_call_analyzer import apk_find_api_calls
from verifier.util.explain.shap import get_input_influences
from verifier.util.strings import remove_html
from verifier.util.text_processing import tokenize_text
from verifier.util.tfidf_model_creator import TfIdfModelCreator, VectorizerDoc
from verifier.util.tfidf_models import get_top_terms


class TestsetEvaluator:

    def __init__(self, package_names, report_folder):
        self.n_top_terms = 25

        self.package_names = package_names
        self.report_folder = report_folder
        self.result_file_name = None
        self._preloaded_input_tfidf_model = {}
        self._preloaded_descriptions_tfidf_model = None
        self.report_saver = ReportSaver(report_folder=report_folder)

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    def run(self):
        self.process_a2t()

    def get_fallback_description(self):
        self.text = SamplesDatabase.get().read(self.package_name, 'description_raw')

    def process_a2t(self):
        self.process_a2t_input('ids')
        self.process_a2t_input('strings')
        self.process_a2t_input('methods')

    def process_a2t_input(self, input_mode):
        print(input_mode)

        if input_mode == 'methods':
            model.model_config_class = config.TFIDFClassifier.ModelMethods
            tfidf_model_file = config.TFIDFModels.code_methods_model
            tfidf_data_file = config.TFIDFModels.code_methods_data
            ml_model_file = config.TrainedModels.app2text_methods
        elif input_mode == 'ids':
            model.model_config_class = config.TFIDFClassifier.ModelResourceIds
            tfidf_model_file = config.TFIDFModels.code_ids_model
            tfidf_data_file = config.TFIDFModels.code_ids_data
            ml_model_file = config.TrainedModels.app2text_ids
        elif input_mode == 'strings':
            model.model_config_class = config.TFIDFClassifier.ModelStrings
            tfidf_model_file = config.TFIDFModels.code_stringres_model
            tfidf_data_file = config.TFIDFModels.code_stringres_data
            ml_model_file = config.TrainedModels.app2text_stringres
        else:
            raise RuntimeError("unknown input mode")

        test_generator = TFIDFGenerator(tfidf_input_data_file=tfidf_data_file,
                                        tfidf_input_model_file=tfidf_model_file,
                                        package_names=self.package_names,
                                        batch_size=1,
                                        shuffle=False,
                                        verbose=True)

        print("loading ML model")

        K.backend.clear_session()

        ml_model = model_for_tf_idfs(test_generator.get_num_inputs(),
                                     test_generator.get_num_outputs())

        ml_model.compile(optimizer='adam',
                         loss='mse',
                         metrics=['mse'])

        ml_model.load_weights(ml_model_file)

        # tensorflow.get_default_graph().finalize()

        print("predicting")

        y = ml_model.predict_generator(test_generator)

        for i in range(y.shape[0]):
            package_name = test_generator.package_names[i]

            descriptions_tfidf_model = test_generator.descriptions['model']
            input_tfidf_model = test_generator.inputs
            input_tfidf_vector = test_generator.inputs.tfidf_data[test_generator.inputs.doc_ids.index(package_name)]

            output_words = get_top_terms(y[i], descriptions_tfidf_model,
                                         top_k=self.n_top_terms)
            words_pred = {word: float(value) for word, value in output_words}
            descriptions_tfidf_indices = np.argsort(-y[i]).flatten().tolist()[:self.n_top_terms]

            influences = get_input_influences(input_tfidf_model, input_tfidf_vector,
                                              descriptions_tfidf_model, descriptions_tfidf_indices,
                                              ml_model)

            progress_pct = "%3d%%    " % ((100*(i+1))//y.shape[0])
            print("-" * 80)
            print(progress_pct, package_name)
            print("-" * 80)
            if len(influences) > 0:
                for word, score in output_words[:7]:
                    print("[%.3f]    %s" % (score, word))
                    if not word in influences:
                        continue
                    for influence, score2 in influences[word][:7]:
                        print("        [%.5f]    %s" % (score2, influence))
            print()

            report_saver = ReportSaver(report_folder=self.report_folder)
            report_saver.set_app_info(package_name)

            report_saver.a2t['words_pred'] = report_saver.a2t.get('words_pred', {})
            report_saver.a2t['words_pred'][input_mode] = words_pred
            report_saver.a2t['input_values'] = report_saver.a2t.get('input_values', {})
            report_saver.a2t['input_values'][input_mode] = influences
            report_saver.a2t['text_actual'] = SamplesDatabase.get().read(package_name, "description_raw")

            report_saver.save()

        K.backend.clear_session()

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


def from_samples_db():
    test_packages = SamplesDatabase.get().filter(('set', '==', 'test'))

    # for p in test_packages:
    #     print("-" * 50)
    #     print(" - ", SamplesDatabase.get().read(p, 'title'))
    #     print("   ", p)
    #     print()
    #     print(SamplesDatabase.get().read(p, 'description_raw'))
    #     print()


    test_packages_1 = ["com.dl.photo.loveframes",
                     "mp3.tube.pro.free",
                     "de.stohelit.folderplayer",
                     "com.avg.cleaner",
                     "com.gau.go.launcherex.gowidget.gopowermaster",
                     "ru.mail",
                     "de.shapeservices.impluslite",
                     "kik.android",
                     "com.hrs.b2c.android",
                     "com.nqmobile.antivirus20.multilang",
                     "com.yellowbook.android2",
                     "com.antivirus.tablet",
                     "taxi.android.client",
                     "com.qihoo.security",
                     "com.jb.gokeyboard.plugin.emoji",
                     "com.niksoftware.snapseed",
                     "com.forshared.music",
                     "mobi.infolife.eraser",
                     "com.hulu.plus",
                     "com.vevo",
                     "com.mobisystems.office",
                     "com.whatsapp",
                     "com.dropbox.android",
                     "com.yahoo.mobile.client.android.yahoo",
                     "com.jessdev.hdcameras",
                     "com.slacker.radio",
                     "com.jb.mms.theme.springtime",
                     "ru.zdevs.zarchiver",
                     "com.newsoftwares.folderlock_v1"]

    test_packages = list(set(test_packages).difference(test_packages_1))

    random.shuffle(test_packages)

    runner = TestsetEvaluator(report_folder="/data/reports", package_names=test_packages)
    runner.run()


if __name__ == "__main__":
    assert CountDictIterator
    assert VectorizerDoc
    assert meta_data_description_tokenize

    from_samples_db()
