def gini_numpy(array, axis=-1):
    """
    Calculate the Gini coefficient of a numpy array along an axis.

    vectorized implementation based on: https://github.com/oliviaguest/gini
    based on bottom equation: http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
    from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    """
    import numpy as np

    # Values must be strictly positive
    array = array - np.min(array) + 1e-7

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


def gini_torch(array, axis=-1):
    """
    Calculate the Gini coefficient of a torch tensor along an axis.

    vectorized implementation based on: https://github.com/oliviaguest/gini
    based on bottom equation: http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
    from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    """
    import torch

    # Values must be strictly positive
    array = array - torch.min(array) + 1e-7

    # Values must be sorted
    array = torch.sort(array, dim=axis)[0]

    # Index per array element
    shape = [1 for _ in array.shape]
    shape[axis] = -1
    index = torch.arange(1, array.shape[axis] + 1, device=array.device).reshape(shape)

    # Number of array elements
    n = array.shape[axis]

    # Gini coefficient
    return (torch.sum((2 * index - n - 1) * array, axis=axis)) / (n * torch.sum(array, axis=axis))
