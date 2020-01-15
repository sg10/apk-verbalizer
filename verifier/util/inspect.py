import types


def copy_func(fn, name=None):
    return types.FunctionType(fn.__code__, fn.__globals__,
                              name=name,
                              argdefs=fn.__defaults__,
                              closure=fn.__closure__)
