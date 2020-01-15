import datetime
import pprint

from androguard.core.analysis.analysis import ExternalMethod
from androguard.misc import AnalyzeAPK
import networkx as nx

from verifier.util.android_slicer.class_inheritance import InheritanceMap
from verifier.util.android_slicer.helper import java_class_to_smali_class
from verifier.util.android_slicer.implicit_edges import AndroidSdkImplicitEdges, CFGImplicitEdgeAdder


def get_simple_identifier(method):
    class_name = method.class_name
    return "%s->%s" % (class_name, method.name)


class AppAPICallAnalyzer:

    def __init__(self, apk, apk_analysis, cg, inheritance_map, print_func=print):
        self.entry_points = []
        self.api_calls_nodes = []
        self.api_calls_triggered = {}
        self.apk = apk
        self.analysis = apk_analysis
        self.cg = cg
        self.all_classes = {}
        self.print = print_func
        self.inheritance_map = inheritance_map
        self.ignore_packages = ["Ljava/lang", "Ljava/util"]
        self.max_api_paths = 160  # if we've already found n paths to a certain api call,
                                  # don't look for more (otherwise analysis can take 12+ hours per app)

    def run(self):
        start_time = datetime.datetime.now()

        self.load_apk()
        self.print("[%4d s] done loading APK" % (datetime.datetime.now()-start_time).total_seconds())
        self.create_manifest_entry_points()
        self.print("[%4d s] done creating entry points list (found %d)" %
              ((datetime.datetime.now()-start_time).total_seconds(), len(self.entry_points)))
        self.create_api_calls_list()
        self.print("[%4d s] done creating API calls list (found %d)" %
              ((datetime.datetime.now()-start_time).total_seconds(), len(self.api_calls_nodes)))
        self.trace_entry_to_apis()
        self.print("[%4d s] done tracing!" % (datetime.datetime.now()-start_time).total_seconds())

    def load_apk(self):
        self.all_classes = {c.name: c for c in self.analysis.find_classes(".*", no_external=True)}

        main_package_smali = "L" + self.apk.get_package().replace(".", "/")
        self.ignore_packages.append(main_package_smali)
        self.print("ignoring %s" % main_package_smali)

        if main_package_smali.count("/") >= 3:
            main_package_smali_up = ("/".join(main_package_smali.split("/")[:-1])) + "/"
            if len(main_package_smali_up) > 8:
                self.print("ignoring %s" % main_package_smali_up)
                self.ignore_packages.append(main_package_smali_up)

    def create_manifest_entry_points(self):
        entry_classes = self.apk.get_activities() + self.apk.get_providers() + self.apk.get_services()
        entry_classes = [e.replace("..", ".") for e in entry_classes]
        entry_classes_smali = ["L%s" % e for e in entry_classes]
        entry_methods = []

        method_names = ["onStart", "onResume", "onPause", "onCreate", "onStop", "onDestroy",
                        "<init>",
                        "onBind", "onUnbind"]

        method_name_regex = "|".join(["(%s)" % m for m in method_names])

        for e in entry_classes_smali:
            entry_methods += list(self.analysis.find_methods(classname=e,
                                                             methodname=method_name_regex))
        self.entry_points = [m.method for m in entry_methods]

        self.print("# entry methods: ", len(self.entry_points))

    def create_api_calls_list(self):
        api_calls_nodes = []

        for n, dex in self.cg.nodes(data=True):
            if dex['external'] and not any([n.class_name.startswith(ig) for ig in self.ignore_packages]):
                api_calls_nodes.append(n)

        self.api_calls_nodes = api_calls_nodes

    def trace_entry_to_apis(self):
        api_calls_triggered_directly = {}

        self.cg.remove_edges_from(self.cg.selfloop_edges())

        i_total = len(self.entry_points) * len(self.api_calls_nodes)
        i_count = 0

        for api in self.api_calls_nodes:
            api_method_id = get_simple_identifier(api)
            if len(api_calls_triggered_directly.get(api_method_id, [])) > self.max_api_paths:
                continue

            for entry_point in self.entry_points:
                i_count += 1
                if entry_point not in self.cg.nodes:
                    continue

                if nx.has_path(self.cg, entry_point, api):
                    api_calls_triggered_directly[api_method_id] = api_calls_triggered_directly.get(api_method_id, []) \
                                                                  + [(entry_point, api)]

                if i_count % 150000 == 0:
                    self.print("@ %.2f%%" % ((i_count*100)/i_total))

        self.print("found %d methods directly" % len(api_calls_triggered_directly))

        api_calls_triggered_indirectly = {}

        # store inherited class names
        inherited_class_names = {}

        for key, links in list(api_calls_triggered_directly.items()):
            for entry_point, api in links:

                inherited_class_names_for_api = inherited_class_names.get(api.class_name)
                if inherited_class_names_for_api is None:
                    inherited_class_names_for_api = self.inheritance_map.get_virtual_methods_up(api.class_name, inherited_class=True)
                    inherited_class_names[api.class_name] = inherited_class_names_for_api

                for inherited_class in inherited_class_names_for_api:
                    inherited_class_smali = inherited_class.method.class_name
                    new_method = ExternalMethod(inherited_class_smali, api.name, api.descriptor)
                    api_method_id = get_simple_identifier(new_method)
                    if len(api_calls_triggered_directly.get(api_method_id, [])) > self.max_api_paths:
                        continue
                    api_calls_triggered_indirectly[api_method_id] = api_calls_triggered_indirectly.get(api_method_id, []) + [(entry_point, api)]

        self.print("found %d methods indirectly" % len(api_calls_triggered_indirectly))

        self.api_calls_triggered = api_calls_triggered_directly
        for key, links in api_calls_triggered_indirectly.items():
            self.api_calls_triggered[key] = self.api_calls_triggered.get(key, []) + links

    def get_api_calls_count(self):
        return {a: len(c) for a, c in self.api_calls_triggered.items()}


def apk_find_api_calls(apk_full_path, quick=False):

    log_tag = apk_full_path.split("/")[-1].replace(".apk", "").rsplit("-", 1)[0][:35].ljust(38)
    def print_function(*args):
        x = tuple([log_tag] + list(args))
        print(*x)

    start_timestamp = datetime.datetime.now()

    #sdk_implicit_edges = AndroidSdkImplicitEdges("ImplicitEdges.txt",
    #                                             "ImplicitEdges.pk",
    #                                             print_func=print_function)
    sdk_implicit_edges = AndroidSdkImplicitEdges("/home/me/ImplicitEdges.txt",
                                                 "/home/me/ImplicitEdges.pk",
                                                 print_func=print_function)

    print_function("-> ", apk_full_path)
    apk, _, analysis = AnalyzeAPK(apk_full_path)
    cfg = analysis.get_call_graph()
    print_function("apk loaded")

    num_edges_1 = len(cfg.edges)
    num_nodes_1 = len(cfg.nodes)

    inh = InheritanceMap(analysis, print_func=print_function)

    if not quick:
        inh.run(cfg)

        num_edges_2 = len(cfg.edges)
        num_nodes_2 = len(cfg.nodes)

        print_function("+%d edges, +%d nodes" % (num_edges_2-num_edges_1, num_nodes_2-num_nodes_1))

        adder = CFGImplicitEdgeAdder(sdk_implicit_edges,
                                     cfg, analysis,
                                     print_func=print_function)
        adder.run()

        num_edges_3 = len(cfg.edges)
        num_nodes_3 = len(cfg.nodes)

        print_function("+%d edges, +%d nodes" % (num_edges_3 - num_edges_2, num_nodes_3 - num_nodes_2))

    apk_analyzer = AppAPICallAnalyzer(apk, analysis, cfg, inh, print_function)
    apk_analyzer.run()

    api_calls_and_count = apk_analyzer.get_api_calls_count()

    print_function("time total: %.1f minutes" % ((datetime.datetime.now()-start_timestamp).total_seconds()/60))

    return api_calls_and_count


def main():
    apk_full_path = '/home/me/Downloads/apks/com.yinzcam.nhl.flames-912200.apk'
    #apk_full_path = '/home/me/Downloads/apks/sf_18.8.apk'
    #apk_full_path = '/home/me/Downloads/apks/adblockplusandroid-1.3.apk'
    #apk_full_path = '/home/me/Downloads/apks/org.telegram.messenger-379.apk'
    #apk_full_path = '/home/me/Downloads/apks/com.whatsapp_2.19.5.apk'

    #apk_full_path = '/nas/public/deepcode_apks_data/SPORTS/com.yinzcam.nhl.flames-912200.apk'

    #pprint.pprint(api_calls_and_count)
    #print(len(api_calls_and_count))

    api_calls_and_count = apk_find_api_calls(apk_full_path)
    #api_calls_and_count_without = process_apk(apk_full_path, quick=True)

    for k in api_calls_and_count.keys():
        print("%130s    %4d  %4d" % (k[:130], api_calls_and_count[k], -1)) #api_calls_and_count_without.get(k, 0)))

    #tps = apk_analyzer.api_calls_triggered["Landroid/content/DialogInterface;->cancel"]
    #pprint.pprint([(str(t[0]), str(t[1])) for t in tps])


if __name__ == "__main__":
    main()
