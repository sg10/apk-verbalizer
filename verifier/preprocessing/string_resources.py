import datetime
import pickle
import pprint
import string

import random

import os
from androguard.misc import AnalyzeAPK
from collections import Counter

from verifier.util.strings import is_camel_case, camel_case_split, remove_non_printable, \
    remove_numbers, is_url, contains_whitespace, is_java_package, is_md5, is_base64
from verifier.util.text_processing import tokenize_text


class APKStringResources:

    MAX_LENGTH_PER_STR_WITHOUT_SPACE = 100
    MAX_LENGTH_PER_STR_WITH_SPACE = 1000
    MAX_LENGTH_PER_ID = 40
    MIN_LENGTH = 1

    def __init__(self, file=None, ids=None, code_str=None, res_str=None):
        self.ids, self.code_str, self.res_str = ids, code_str, res_str
        if file:
            self.load_file(file)

    def load_file(self, file_path):
        file = open(file_path, "rb")
        self.ids, self.code_str, self.res_str = pickle.load(file)

    def get_ids_cleaned(self):
        ids = []

        for id in self.ids:
            _id = id
            if len(id) > APKStringResources.MAX_LENGTH_PER_ID:
                continue
            if id.isnumeric():
                continue

            id_parts = self.split_identifier_into_words(id)

            if len(id_parts) >= 0:
                ids.append(id_parts)

        return ids

    def get_code_strings_cleaned(self):
        return self.get_strings_cleanded(self.code_str)

    def get_res_strings_cleaned(self):
        return self.get_strings_cleanded(self.res_str)

    def get_all_strings_cleaned(self):
        return self.get_code_strings_cleaned() + self.get_res_strings_cleaned()

    def get_strings_cleanded(self, raw_strings):
        strings = []

        for string in raw_strings:

            if len(string) < APKStringResources.MIN_LENGTH:
                continue

            if string.isnumeric():
                continue

            if is_java_package(string):
                # add separately as a whole
                strings.append(string)
                pass

            if not " " in string and is_url(string):
                # add separately as a whole
                strings.append(string)
                pass

            if contains_whitespace(string):
                if len(string) > APKStringResources.MAX_LENGTH_PER_STR_WITH_SPACE:
                    continue

                subtokens = tokenize_text(string, clean_and_stem=False)
                subtokens_split = []
                for s in subtokens:
                    subtokens_split += self.split_identifier_into_words(s)

                subtokens = subtokens_split

            else:
                if len(string) > APKStringResources.MAX_LENGTH_PER_STR_WITHOUT_SPACE:
                    continue

                subtokens = self.split_identifier_into_words(string)

            strings += subtokens

        return strings

    def split_identifier_into_words(self, id):
        id = remove_non_printable(id)
        for c in string.punctuation:
            id = id.replace(c, " ")
        id_parts = [p.strip() for p in id.split()]
        id_parts = [p for p in id_parts if len(p) > APKStringResources.MIN_LENGTH]
        id_parts = [camel_case_split(p) if is_camel_case(p) else [p] for p in id_parts if p]
        id_parts = [p for sublist in id_parts for p in sublist]
        #id_parts = [p.lower() if is_first_upper(p) else p for p in id_parts if p]
        id_parts = [p.lower() for p in id_parts if p]
        id_parts = [remove_numbers(p) if not p.isalpha() else p for p in id_parts]
        id_parts = [p for p in id_parts if len(p) > APKStringResources.MIN_LENGTH]

        return id_parts


class APKStringResourceExtractor:

    def __init__(self, file, print_func=print):
        self.file = file
        self.print = print_func

        self.apk = None
        self.apk_analysis = None
        self.resources = None
        self.package_names = []
        self.locales_priority = []

        self.strings_in_resources = set()
        self.strings_in_code = set()
        self.id_names = set()

    def run(self):
        start_time = datetime.datetime.now()

        self.load_apk()
        self.select_locales()
        self.create_string_resources_list()
        self.create_resource_ids_list()

    def create_resource_ids_list(self):
        for package in self.package_names:
            for locale in self.locales_priority:
                if locale not in self.resources.values[package]:
                    continue
                for res_type, values in self.resources.values[package][locale].items():
                    for value in values:
                        if len(value) == 1 or len(value) == 2:
                            self.id_names.add(value[0])
                        elif len(value) == 3:
                            self.id_names.add(value[1])

    def load_apk(self):
        self.apk, _, self.apk_analysis = AnalyzeAPK(self.file)
        self.resources = self.apk.get_android_resources()
        self.package_names = self.resources.get_packages_names()

    def select_locales(self):
        all_locales = set()
        for apk_package_name in self.package_names:
            all_locales.update(self.resources.get_locales(apk_package_name))

        all_locales = list(all_locales)
        all_locales = [l if l != "\x00\x00" else "DEFAULT" for l in all_locales]

        if len(all_locales) == 1:
            self.locales_priority = all_locales
            return

        # prioritized locales

        # first: pure english
        if "en" in all_locales:
            self.locales_priority.append("en")
        # then: country-specific english
        self.locales_priority += [l for l in all_locales if l.startswith("en-")]

        # then: default language
        if "DEFAULT" in all_locales:
            self.locales_priority.append("DEFAULT")
            self.locales_priority.append("\x00\x00")

        # if still no locale found, raise error
        if len(self.locales_priority) == 0:
            raise Exception("no locale found")

    def create_string_resources_list(self):
        strings = {}

        for pkg, locales_res in self.resources.get_resolved_strings().items():
            for locale in self.locales_priority:
                if locale in locales_res:
                    for id, value in locales_res[locale].items():
                        value = value.strip()
                        if len(value) == 0:
                            continue
                        if id not in strings:
                            strings[id] = value

        self.strings_in_resources = set(strings.values())

        self.strings_in_code = set([s.value.strip() for s in self.apk_analysis.get_strings()])

    def get_all_strings(self):
        return list(self.strings_in_resources) + list(self.strings_in_code)

    def get_code_strings(self):
        return list(self.strings_in_code)

    def get_resource_strings(self):
        return list(self.strings_in_resources)

    def get_id_names(self):
        return list(self.id_names)

    def save(self, out_file):
        with open(out_file, "wb") as fp:
            data = (self.get_id_names(), self.get_code_strings(), self.get_resource_strings())
            pickle.dump(data, fp)


def preprocess():
    stringres_folder = 'app_properties/raw/run4/stringres/'
    #apks_folder = '/nas/public/deepcode_apks_data/COMMUNICATION'
    #files = [f for f in os.listdir(apks_folder) if ".apk" in f and "org.telegram.messenger-" in f]
    #files = [f for f in os.listdir(apks_folder) if ".apk" in f and "org.telegram.messenger-" in f]
    #files = ['kik.android.pk']
    files = [f for f in os.listdir(stringres_folder)]

    random.shuffle(files)
    #files = files[0:10]

    cnt = 0
    counter = Counter()

    for file in files:
        file_full = os.path.join(stringres_folder, file)
        code_strings = APKStringResources(file=file_full).get_code_strings_cleaned()

        for c in code_strings:
            for t in c:
                counter[t] = counter.get(t, 0) + 1

        cnt += 1

        if cnt % 100 == 0:
            print(cnt, " files   ", len([c for c, v in counter.items() if v > 100]))
            print(list(sorted([(c, v) for c, v in counter.items() if v > 100], key=lambda x: x[1], reverse=True))[:100])

        #file = os.path.join(stringres_folder, file)
        #resources = APKStringResources(file)
        #elements = resources.get_elements()
        #pprint.pprint(elements)


def test():
    stringres_folder = 'app_properties/raw/run4/stringres/'
    files = [f for f in os.listdir(stringres_folder)]
    random.shuffle(files)
    #files = files[0:10]

    cnt = Counter()
    i = 0

    for file in files:
        a = APKStringResources(file=os.path.join(stringres_folder, file)) #"app_properties/raw/run4/stringres/org.telegram.messenger.pk")
        s = a.get_all_strings_cleaned()
        i += 1
        cnt += Counter(s)

        if i % 100 == 0:
            cnt1 = len(cnt)
            cnt2 = len([d for d in cnt.values() if 1000 < d < 25000])
            print("%8d              %10d / %10d" % (i, cnt1, cnt2))

        continue

        if i % 100 == 0:
            pprint.pprint(cnt)

    #pprint.pprint(s)


if __name__ == "__main__":
    #preprocess()
    test()
    pass
