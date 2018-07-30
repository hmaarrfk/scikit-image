from .._shared.deprecations import default_parameter_change


def foo(this='that'):
    """this is my docstring.

    Parameters
    ----------
    this : str
        Should be a string

    """
    print(this)


@default_parameter_change('0.16', this='tim')
def foo_deprecated(this='that'):
    """this is my docstring too.

    Parameters
    ----------
    this : str
        Should be a string

    """
    print(this)


@default_parameter_change('0.13', this='tim')
def foo_deprecated_13(this='that'):
    """this is my docstring too.

    Parameters
    ----------
    this : str
        Should be a string

    """
    print(this)
