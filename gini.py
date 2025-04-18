import numpy as np


def gini(array, axis=-1):
    """
    Calculate the Gini coefficient of a numpy array.

    forked from: https://github.com/oliviaguest/gini
    based on bottom eq: http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
    from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    """

    # Values must be strictly positive
    array = array - np.amin(array) + 1e-7

    # Values must be sorted
    array = np.sort(array, axis=axis)

    # Index per array element
    shape = [1 for _ in array.shape]
    shape[axis] = -1
    index = np.arange(1, array.shape[axis] + 1).reshape(shape)

    # Number of array elements
    n = array.shape[axis]

    # Gini coefficient
    return (np.sum((2 * index - n - 1) * array, axis=axis)) / (n * np.sum(array, axis=axis))
