import glob
import os
import javalang

from javalang.tree import CompilationUnit, ClassDeclaration, InterfaceDeclaration


class JavaClass:
    def __init__(self):
        self.ref = None
        self.method_names = []


class AndroidSdkStubs:

    def __init__(self, folder):
        self.folder = os.path.join(folder, "src")
        self.sdk_classes = {}

    def load(self):
        p = os.path.join(self.folder, "android/**/*.java")
        java_files = glob.glob(p, recursive=True)
        for java_file in java_files:
            self.parse_class(java_file)

    def parse_class(self, file):
        str_code = open(file, "r").read()
        unit = javalang.parse.parse(str_code)
        if type(unit) is not CompilationUnit:
            return

        for class_type in unit.types:
            if type(class_type) is InterfaceDeclaration or \
               type(class_type) is ClassDeclaration:
                fqd_class = "%s.%s" % (unit.package.name, class_type.name)
                self.add_class(fqd_class, class_type)
                self.parse_inner_classes(fqd_class, class_type.children)

    def parse_inner_classes(self, fqd_class, class_type_children):
        for ch in class_type_children:
            if type(ch) is list:
                for declaration in ch:
                    if type(declaration) is InterfaceDeclaration or type(declaration) is ClassDeclaration:
                        fqd_inner_class = "%s$%s" % (fqd_class, declaration.name)
                        self.add_class(fqd_inner_class, declaration)

    def add_class(self, fqd_class, class_type):
        java_class = JavaClass()
        java_class.ref = class_type
        java_class.method_names = [m.name for m in class_type.methods]
        self.sdk_classes[fqd_class] = java_class

    def method_exists(self, class_name, method_name):
        fqd_class_java_style = class_name.lstrip("L").rstrip(";").replace("/", ".")
        return method_name in self.sdk_classes.get(fqd_class_java_style, JavaClass()).method_names

