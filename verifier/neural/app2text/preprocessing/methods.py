import datetime
import json
import pickle
import numpy as np
import os
import random
import threading
import time
import traceback
import matplotlib.pyplot as plt

from verifier import config
from verifier.neural.app2text.preprocessing.strings_and_ids import CountDictIterator, get_relevant_files
from verifier.preprocessing.string_resources import APKStringResourceExtractor
from verifier.util.android_slicer.app_api_call_analyzer import apk_find_api_calls
from verifier.util.batch_file_preprocessor import find_files_in_folder_recursive, BatchFilePreprocessor
from verifier.util.tfidf_model_creator import TfIdfModelCreator


def store_api_calls(apk_full_path, target_file):
    if not os.path.exists(apk_full_path):
        raise RuntimeError("apk file does not exist")

    try:
        linked_api_calls = apk_find_api_calls(apk_full_path)
        json.dump(linked_api_calls, open(target_file, "w"))
    except Exception as e:
        print(e)
        traceback.print_tb(e.__traceback__)
        return

    print("saved %d api calls" % len(linked_api_calls))


def main():
    print("started")
    files = find_files_in_folder_recursive(config.SourceData.apks_folder, ".apk")
    files_dict = {f.rsplit("/", maxsplit=1)[-1].rsplit("-", maxsplit=1)[0].lower(): f for f in files}
    files_missing = [f.lower() for f in open('missing_packages.txt').read().split("\n")]
    print(len(files_missing))
    files = [f for pkg, f in files_dict.items() if pkg in files_missing]

    print("# files total = ", len(files))

    processor = BatchFilePreprocessor(files,
                                      config.SourceData.methods_deep_folder,
                                      process_function=store_api_calls,
                                      output_file_template='%s.code.json',
                                      lock=True,
                                      lock_seconds=15*60)
    processor.run()


def create_tfidf_model():
        print("Methods")
        packages, files = get_relevant_files("code.json")

        model_creator = TfIdfModelCreator(model_file=config.TFIDFModels.code_methods_model,
                                          data_file=config.TFIDFModels.code_methods_data,
                                          files_folder="app_properties/raw/run3/methods_deep/",
                                          files=files,#[:10000],
                                          file_ids=packages,#[:10000],
                                          input_analyzer=CountDictIterator.count_to_multi_tokens,
                                          min_df_pct=0.001,
                                          max_df_pct=0.8)
        model_creator.load(model=True, data=False)
        #model_creator.create_model()
        CountDictIterator.reset()  # progress info only
        #model_creator.info()
        model_creator.transform_data()
        model_creator.info()

        print(len(model_creator.tfidf_model.get_feature_names()))
        print(len(model_creator.get_document_frequencies()))

        dfs = model_creator.get_document_frequencies()

        ranges = [
            (25, 20000),
            (50, 20000),
            (75, 20000),
            (100, 20000),
            (150, 20000),
            (25, 30000),
            (50, 30000),
            (75, 30000),
            (100, 30000),
            (150, 30000),
            (25, 40000),
            (50, 40000),
            (75, 40000),
            (100, 40000),
            (150, 40000),
            (25, 50000),
            (50, 50000),
            (75, 50000),
            (100, 50000),
            (150, 50000),
            (300, 6000),
        ]

        for df_min, df_max in ranges:
            n = len([d for d in dfs if df_min < d < df_max])
            print(df_min, df_max, "     ", n)

        df = model_creator.get_document_frequencies()
        plt.figure()
        plt.hist(df, bins=50, log=True)
        plt.show()
        plt.show()


if __name__ == "__main__":
    #main()
    create_tfidf_model()

