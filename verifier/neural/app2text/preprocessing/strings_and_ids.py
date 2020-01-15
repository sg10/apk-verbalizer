import json
import os
from json import JSONDecodeError
import pprint

from verifier import config
from verifier.preprocessing.samples_database import SamplesDatabase
from verifier.preprocessing.string_resources import APKStringResourceExtractor, APKStringResources
from verifier.util.batch_file_preprocessor import find_files_in_folder_recursive, BatchFilePreprocessor
from verifier.util.tfidf_model_creator import TfIdfModelCreator


def process_apks():
    print("started")
    files = find_files_in_folder_recursive("/nas/public/deepcode_apks_data", ".apk")
    print("# files total = ", len(files))

    def process_function(input_file, output_file):
        extractor = APKStringResourceExtractor(input_file)
        extractor.run()
        extractor.save(output_file)

    processor = BatchFilePreprocessor(files,
                                      config.SourceData.stringres_folder,
                                      process_function,
                                      output_file_template='%s.pk',
                                      lock=True,
                                      lock_seconds=5 * 60)
    processor.run()


def get_relevant_files(ext):
    SamplesDatabase.set_file('samples_database.pk')
    samples_props = SamplesDatabase.get()
    packages = samples_props.filter(('lang', '==', 'en'))
    doc_files = ["%s.%s" % (f, ext) for f in packages]

    return packages, doc_files


def read_ids_and_tokenize(file_path):
    resources = APKStringResources(file=file_path)
    tokens = resources.get_ids_cleaned()
    return [s for sublist in tokens for s in sublist]


def read_strings_and_tokenize(file_path):
    resources = APKStringResources(file=file_path)
    tokens = resources.get_all_strings_cleaned()
    #tokens = [s for sublist in tokens for s in sublist]
    #print("-->", tokens)
    return tokens


class CountDictIterator:
    counter = 0
    failed_json = 0
    failed_exist = 0

    @staticmethod
    def count_to_multi_tokens(doc):
        CountDictIterator.counter += 1
        if CountDictIterator.counter % 100 == 0:
            print("Count dict iterator, file ", CountDictIterator.counter)

        if not os.path.exists(doc.file_path):
            CountDictIterator.failed_exist += 1
            print("file does not exist: ", doc.file_path, CountDictIterator.failed_exist)
            return []

        try:
            data = json.load(open(doc.file_path, "r"))
        except JSONDecodeError:
            CountDictIterator.failed_json += 1
            print("error decoding file: ", doc.file_path, CountDictIterator.failed_json)
            return []

        lst = []
        for key, count in data.items():
            lst += [key] * count

        return lst

    @staticmethod
    def reset():
        CountDictIterator.counter = 0
        CountDictIterator.failed_exist = 0
        CountDictIterator.failed_json = 0


def run():

    if True:
        print("Code and Resources")
        packages, files = get_relevant_files("pk")

        #n = 10000
        #packages = packages[:n]
        #files = files[:n]
        stringres_folder = 'app_properties/raw/run4/stringres/'

        model_creator = TfIdfModelCreator(model_file=config.TFIDFModels.code_stringres_model,
                                          data_file=config.TFIDFModels.code_stringres_data,
                                          files_folder=stringres_folder,
                                          files=files,
                                          file_ids=packages,
                                          ngram_range=(1, 1),
                                          tokenize_function=read_strings_and_tokenize)
        #model_creator.load(model=True, data=True)
        model_creator.create_model()
        model_creator.transform_data()
        model_creator.info()

    if False:
        print("IDs")
        #packages, files = get_relevant_files("pk")

        model_creator = TfIdfModelCreator(model_file=config.TFIDFModels.code_ids_model,
                                          data_file=config.TFIDFModels.code_ids_data,
                                          #files_folder=stringres_folder,
                                          #files=files,
                                          #file_ids=packages,
                                          tokenize_function=read_ids_and_tokenize)
        model_creator.load(model=True, data=True)
        # model_creator.create_model()
        # model_creator.transform_data()
        model_creator.info()


if __name__ == "__main__":
    run()
