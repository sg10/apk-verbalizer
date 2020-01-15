import datetime
import os
import pprint
import random
import time
import traceback


class BatchFilePreprocessor:

    def __init__(self, input_files_list, output_folder,
                 process_function,
                 output_file_template="%s",
                 lock=False,
                 lock_seconds=10*60):
        self.input_files = input_files_list
        self.output_folder = output_folder
        self.output_file_template = output_file_template
        self.process_function = process_function

        self.lock = lock
        self.lock_seconds = lock_seconds

        self.num_processed = 0
        self.num_locked = 0

        self.last_progress_printed = datetime.datetime.now()
        self.progress_every_n_seconds = 10

    def run(self):
        random.shuffle(self.input_files)

        for i, input_file_full in enumerate(self.input_files):
            if not os.path.exists(input_file_full) or not os.path.isfile(input_file_full):
                print("invalid file: ", input_file_full)
                continue

            app_package_name = input_file_full.rsplit("/", maxsplit=1)[-1].rsplit("-", maxsplit=1)[0]

            output_file_name = self.output_file_template % app_package_name
            output_file_full = os.path.join(self.output_folder, output_file_name)
            target_file = os.path.join(self.output_folder, output_file_name)

            if os.path.exists(target_file):
                continue

            lock_file = "%s.lock" % target_file
            if self.lock:
                if os.path.exists(lock_file):
                    modification_time = datetime.datetime.fromtimestamp(os.path.getmtime(lock_file))
                    if (datetime.datetime.now() - modification_time).total_seconds() > self.lock_seconds:
                        os.remove(lock_file)
                    else:
                        self.num_locked += 1
                        continue
                open(lock_file, "w").close()

            try:
                self.process_function(input_file_full, output_file_full)
                self.print_progress(i)
                self.num_processed += 1
            except Exception as e:
                print(e)
                traceback.print_tb(e.__traceback__)

            if self.lock:
                os.remove(lock_file)

    def print_progress(self, i):
        now = datetime.datetime.now()
        if (now-self.last_progress_printed).total_seconds() > self.progress_every_n_seconds:
            pct = 100*i/len(self.input_files)
            print("[PROGRESS]  %3.2f%%     %10d processed  %10d skipped" % (pct, self.num_processed, i-self.num_processed))
            self.last_progress_printed = now


def find_files_in_folder_recursive(start_folder, extension):
    files = []

    folders_to_traverse = [start_folder]

    while len(folders_to_traverse) > 0:
        folder = folders_to_traverse.pop(0)

        for item in os.listdir(folder):
            item_full_path = os.path.join(folder, item)

            if os.path.isdir(item_full_path):
                folders_to_traverse.append(item_full_path)
            elif os.path.isfile(item_full_path) and item_full_path.endswith(extension):
                files.append(item_full_path)

    return files
