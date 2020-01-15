
LJAVA_LANG_OBJECT = 'Ljava/lang/Object;'


def smali_class_to_java_class(c):
    return c[1:].replace("/", ".").rstrip(";")


def java_class_to_smali_class(c):
    return "L%s;" % c.replace(".", "/")
