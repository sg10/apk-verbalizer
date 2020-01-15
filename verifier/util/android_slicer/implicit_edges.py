import datetime
import os
import pickle
import pprint
import random

import networkx as nx
from androguard.core.analysis.analysis import ExternalMethod

from verifier.util.android_slicer.helper import java_class_to_smali_class


class AndroidSdkImplicitEdges:

    _preloaded_pickle_file = None

    def __init__(self, edgeminer_file_raw, edgeminer_file_pickled, print_func=print):
        self.print = print_func
        self.edgeminer_file_pickled = edgeminer_file_pickled
        if AndroidSdkImplicitEdges._preloaded_pickle_file is not None:
            self.targets, self.nodes_entry = AndroidSdkImplicitEdges._preloaded_pickle_file
            self.print("found preloaded pickle file")
        elif os.path.exists(self.edgeminer_file_pickled):
            self.print("loading pickled edgeminer file from fs")
            self.targets, self.nodes_entry = pickle.load(open(edgeminer_file_pickled, "rb"))
        else:
            self.print("loading raw edgeminer file from fs")
            self.targets = {}
            self.nodes_entry = {}  # key: class_name of entry class,
            # value: dict
            #   key: method_name in entry class
            #   value: tuple of triggered class_name, method_name
            self.load(edgeminer_file_raw)

        if AndroidSdkImplicitEdges._preloaded_pickle_file is None:
            AndroidSdkImplicitEdges._preloaded_pickle_file = (self.targets, self.nodes_entry)

    def load(self, edgeminer_file_raw):
        with open(edgeminer_file_raw, "r") as fp:
            for counter, line in enumerate(fp):
                parts = line.split("#")
                if len(parts) != 3:
                    continue

                if counter % 100000 == 0:
                    self.print("read %7d lines" % counter)

                entry_class_and_method = parts[0].split(":")[0]
                entry_class, entry_method_name = entry_class_and_method.rsplit(".", maxsplit=1)
                triggered_class_and_method = parts[1].split(":")[0]
                triggered_class, triggered_method_name = triggered_class_and_method.rsplit(".", maxsplit=1)

                self.nodes_entry[entry_class] = self.nodes_entry.get(entry_class, {})
                self.nodes_entry[entry_class][entry_method_name] = self.nodes_entry[entry_class].get(entry_method_name, [])
                self.nodes_entry[entry_class][entry_method_name].append((triggered_class, triggered_method_name))

                self.targets[triggered_class] = self.targets.get(triggered_class, set())
                self.targets[triggered_class].add(triggered_method_name)

        self.print("done loading data")
        pickle.dump((self.targets, self.nodes_entry), open(self.edgeminer_file_pickled, "wb"))
        self.print("saved data to pickle file")

    def get_smali_classes_entry_list(self):
        return ["L%s;" % n.replace(".", "/") for n in self.nodes_entry.keys()]

    def get_implicit_target_for_entry(self, class_name, method_name):
        class_name_in_java = class_name[1:].replace("/", ".").rstrip(";").replace("$", ".", 1)
        if class_name_in_java in self.nodes_entry and method_name in self.nodes_entry[class_name_in_java]:
            linked_methods = self.nodes_entry[class_name_in_java][method_name]
            linked_methods_smali = [(java_class_to_smali_class(l[0]), l[1]) for l in linked_methods]
            return linked_methods_smali
        else:
            return []

    def get_implicit_targets_of_class(self, class_name):
        class_name = class_name[1:].replace("/", ".").rstrip(";")
        if class_name in self.targets:
            return list(self.targets[class_name]) or []
        else:
            return []


class CFGImplicitEdgeAdder:

    def __init__(self, sdk_implicit_edges, cfg, apk_analysis, print_func=print):
        self.sdk_implicit_edges = sdk_implicit_edges
        self.cfg = cfg
        self.analysis = apk_analysis
        self.print = print_func

        self.class_and_method_to_node = {}
        self.app_classes = {}
        self.additional_edges = set()
        self.num_edges_added = 0
        self.num_nodes_added = 0

    def run(self):
        start_time = datetime.datetime.now()

        self.prepare()
        self.print("[%4d s] done fetching classes" % (datetime.datetime.now()-start_time).total_seconds())
        self.find_additional_edges()
        self.print("[%4d s] done finding additional edges (found %d)" %
              ((datetime.datetime.now()-start_time).total_seconds(), len(self.additional_edges)))
        self.add_additional_edges()
        self.print("[%4d s] done adding additional edges (%d edges and %d nodes)" %
              ((datetime.datetime.now()-start_time).total_seconds(), self.num_edges_added, self.num_nodes_added))

    def prepare(self):
        self.class_and_method_to_node = {
            "%s#%s" % (node.class_name, node.name): node
            for node in self.cfg.nodes
        }
        self.app_classes = {c.name: c for c in self.analysis.find_classes(".*", no_external=True)}

    def find_additional_edges(self):
        for node in self.cfg.nodes.keys():
            inherited_class_names = self.get_inherited_class_names(node)
            self.additional_edges.update(self.matching_virtual_methods(node.class_name, inherited_class_names))

    def add_additional_edges(self):
        for start_node_method, dest_node_method in self.additional_edges:
            start_node_id = "%s#%s" % start_node_method
            dest_node_id = "%s#%s" % dest_node_method
            if self.class_and_method_to_node.get(start_node_id) is not None:
                if self.class_and_method_to_node.get(dest_node_id) is None:
                    em = ExternalMethod(dest_node_method[0], dest_node_method[1], "")
                    self.cfg.add_node(em, external=True, entrypoint=False)
                    self.class_and_method_to_node[dest_node_id] = em
                    self.num_nodes_added += 1
                if not nx.has_path(self.cfg,
                                   self.class_and_method_to_node[start_node_id],
                                   self.class_and_method_to_node[dest_node_id]):
                    self.cfg.add_edge(self.class_and_method_to_node[start_node_id],
                                      self.class_and_method_to_node[dest_node_id])
                    self.num_edges_added += 1

    def get_inherited_class_names(self, node):
        inherited_class_names = []
        class_names_to_check = [node.class_name]

        while len(inherited_class_names) < 5 and len(class_names_to_check) > 0:
            class_name_to_check = class_names_to_check.pop(0)
            class_to_check = self.app_classes.get(class_name_to_check)

            if class_to_check is None:
                continue

            inherited_class_names += class_to_check.implements + [class_to_check.extends]

        inherited_class_names = [r for r in inherited_class_names if not r.startswith("Ljava/lang")]

        return inherited_class_names

    def matching_virtual_methods(self, node_class_name, inherited_class_names):
        class_obj = self.app_classes.get(node_class_name)
        if class_obj is None or class_obj.orig_class.class_data_item is None:
            return []

        virtual_methods = []
        for encoded_method in class_obj.orig_class.class_data_item.virtual_methods:
            virtual_methods.append((class_obj.name, encoded_method.name))

        api_target_methods = []
        for inherited_class_name in inherited_class_names:
            for _, virtual_method_name in virtual_methods:
                api_target_methods += self.sdk_implicit_edges.get_implicit_target_for_entry(inherited_class_name,
                                                                                            virtual_method_name)

        implicit_edges = set()

        # edge from user-defined method directly to api method
        for api_target_class_name, api_target_method_name in api_target_methods:
            source_method = (node_class_name, api_target_method_name)
            target_method = (api_target_class_name, api_target_method_name)
            implicit_edges.add((source_method, target_method))

        # check if user-defined class overwrites api method and create edge between overwritten method and api method
        for virtual_method_class, virtual_method_name in virtual_methods:
            if len(api_target_methods) > 0:
                for api_target_class_name, api_target_method_name in api_target_methods:
                    if api_target_method_name == virtual_method_name:
                        source_method = (virtual_method_class, api_target_method_name)
                        target_method = (api_target_class_name, api_target_method_name)
                        implicit_edges.add((source_method, target_method))

        implicit_edges = list(implicit_edges)

        return implicit_edges

    def get_cfg(self):
        return self.cfg

