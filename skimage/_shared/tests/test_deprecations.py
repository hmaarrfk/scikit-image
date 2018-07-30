from skimage._shared.deprecations import default_parameter_change


@default_parameter_change('0.16', this='tim')
def foo(this='that'):
    """This function is called foo.

    Parameters
    ----------
    this : string
        That you wish to print.

    """
    print(this)


def foo_orig(this='that'):
    print(this)


def foo_tests():
    foo()
    foo(this='me')
