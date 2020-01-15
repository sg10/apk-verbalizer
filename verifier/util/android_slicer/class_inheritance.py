import datetime

import networkx as nx

from verifier.util.android_slicer.helper import LJAVA_LANG_OBJECT


class AndroidClass:

    def __init__(self, name, ref):
        self.name = name
        self.ref = ref
        self.derives_from = set()
        self.inherites_to = set()

    def __repr__(self):
        return "CLASS # " + self.name


class InheritanceMap:

    def __init__(self, analysis, print_func=print):
        self.analysis = analysis
        self.num_edges_added = 0
        self.print = print_func
        self.classes = dict()
        self.classes[LJAVA_LANG_OBJECT] = AndroidClass(LJAVA_LANG_OBJECT, None)

    def run(self, cfg):
        start_time = datetime.datetime.now()

        self.print("creating inheritance map")
        self.create_tree()
        self.print("[%4d s] resolved inheritances" % (datetime.datetime.now()-start_time).total_seconds())
        self.add_ctor_edges(cfg)
        self.print("[%4d s] added c'tor edges: %d" % ((datetime.datetime.now()-start_time).total_seconds(), self.num_edges_added))

    def create_tree(self):
        classes = {c.name: c for c in self.analysis.get_classes()}
        for name, c in classes.items():
            derives_from = set([c.extends] + c.implements)
            android_class = AndroidClass(name, c)
            android_class.derives_from = derives_from
            self.classes[name] = android_class

        for class_name, android_class in self.classes.items():
            android_class.derives_from = [self.classes.get(c, c) for c in android_class.derives_from]
            for class_name_2, android_class_2 in self.classes.items():
                if class_name in android_class_2.derives_from or android_class in android_class_2.derives_from:
                    android_class.inherites_to.add(android_class_2)

        self.print("# classes with inheritances resolved    ", len(self.classes))

    def get_virtual_methods_down(self, class_name):
        if class_name == LJAVA_LANG_OBJECT:
            return []

        target_class = self.classes.get(class_name)
        if target_class is None or target_class.ref is None:
            return []

        target_methods = [m for m in target_class.ref._methods.values()]
        virtual_methods_found = []

        sub_queue = list(target_class.inherites_to)
        inherits_to = []

        while len(sub_queue) > 0:
            if LJAVA_LANG_OBJECT in sub_queue:
                sub_queue.remove(LJAVA_LANG_OBJECT)
            next = sub_queue.pop(0)
            if next:
                inherits_to.append(next)
                sub_queue += list(next.inherites_to)
                sub_queue = list(set(sub_queue))

        for sub_class in inherits_to:
            for method in sub_class.ref._inherits_methods.values():
                for target_method in target_methods:
                    if target_method.name == method.name:
                        virtual_methods_found.append(method)

        return virtual_methods_found

    def get_virtual_methods_up(self, class_name, inherited_class=False):
        target_class = self.classes.get(class_name)
        if target_class is None:
            return []

        target_methods = [m.method for m in target_class.ref._methods.values()]
        virtual_methods_found = []

        parents_queue = list(target_class.derives_from)
        derives_from = []

        while len(parents_queue) > 0:
            next = parents_queue.pop(0)
            if next and type(next) is not str and next.name != LJAVA_LANG_OBJECT:
                derives_from.append(next)
                parents_queue += list(next.derives_from)
                parents_queue = list(set(parents_queue))

        for parent in derives_from:
            if parent.name == LJAVA_LANG_OBJECT:
                continue
            for parent_method in parent.ref._methods.values():
                for target_method in target_methods:
                    if target_method.name == parent_method.name and target_method.name != "<init>":
                        if inherited_class:
                            virtual_methods_found.append(parent_method)
                        else:
                            virtual_methods_found.append(target_method)

        return virtual_methods_found

    def add_ctor_edges(self, cfg):
        constructors = self.analysis.find_methods(classname='.*', methodname='<init>')
        for constructor in constructors:
            for linked_method in self.get_virtual_methods_down(constructor.method.class_name):
                if not nx.has_path(cfg, constructor.method, linked_method):
                    cfg.add_edge(constructor.method, linked_method)
                    self.num_edges_added += 1
            for linked_method in self.get_virtual_methods_up(constructor.method.class_name):
                if not nx.has_path(cfg, constructor.method, linked_method):
                    cfg.add_edge(constructor.method, linked_method)
                    self.num_edges_added += 1
