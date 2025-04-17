import random

import numpy as np

from gini import gini


def test_example_1():
    # For a very unequal sample, 999 zeros and a single one, the Gini coefficient is very close to 1.0:
    array = np.zeros((1000))
    array[0] = 1.0
    score = 0.99890010998900103
    assert abs(gini(array) - score) < 0.01


def test_example_2():
    # For uniformly distributed random numbers, it will be low, around 0.33:
    array = np.random.uniform(-1, 0, 1000)
    score = 0.3295183767105907
    assert abs(gini(array) - score) < 0.01


def test_example_3():
    # For a homogeneous sample, the Gini coefficient is 0.0:
    array = np.ones((1000))
    score = 0.0
    assert abs(gini(array) - score) < 0.01


def test_int():
    score = 0.8641975328641904
    array = [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000]
    array = np.array(array)

    assert abs(gini(array) - score) < 0.01

    # Make some values negative
    array -= 1_000

    assert abs(gini(array) - score) < 0.01


def test_random_and_large():
    array = [random.randint(0, 10) for _ in range(100)] + [100_000_000]
    array = np.array(array)

    assert gini(array) > 0.98


def test_random_vector():
    shape = (2, 100)
    array = [[random.randint(0, 10) for _ in range(shape[1])] for _ in range(shape[0])]
    array = np.array(array)

    assert 0 < gini(array) < 1
