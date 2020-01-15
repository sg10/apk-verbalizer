from urllib.parse import urlparse

import re
import string

from bs4 import BeautifulSoup

_printable_characters = set(string.printable)
_alphanum_dot_underscore = set(["_", "."] + [c for c in string.ascii_letters + string.digits])


def is_ascii(s):
    return len(s) == len(s.encode())


def remove_non_printable(s):
    return "".join(filter(lambda c: c in _printable_characters, s))


def remove_numbers(s):
    return ''.join([c for c in s if not c.isdigit()])


def is_camel_case(s):
    return s != s.lower() and s != s.upper() and "_" not in s


def camel_case_split(identifier, delimiter=None):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    if delimiter is None:
        return [m.group(0) for m in matches]
    else:
        return str(delimiter).join([m.group(0) for m in matches])


def is_first_upper(s):
    return s[0].isupper() and s[1:].islower()


def is_url(s):
    try:
        result = urlparse(s)
        return result.scheme and result.netloc
    except:
        return False


def is_java_package(s):
    if "." not in s or " " in s:
        return False

    if s[0] == "." or s[-1] == ".":
        return False

    return all([c in _alphanum_dot_underscore for c in s])


def is_md5(s):
    if type(s) is not string:
        return False

    s = s.lower()

    if not s.isalnum() or len(s) < 8:
        return False

    for c in ['a', 'b', 'c', 'd', 'e', 'f', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
        s = s.replace(c, "")

    if len(s) == 0:
        return True
    else:
        return False


def is_base64(s):
    return False


def contains_whitespace(s):
    return True in [c in s for c in string.whitespace]


def remove_html(s):
    try:
        text = BeautifulSoup(s, features="html.parser").text
    except TypeError as e:
        text = ""

    return text
