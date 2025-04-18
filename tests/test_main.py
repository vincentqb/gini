import random

import numpy as np

from gini import gini


def test_example_1():
    # For a very unequal sample, 999 zeros and a single one, the Gini coefficient is very close to 1.0:
    array = np.zeros((1000))
    array[0] = 1.0
    score = 0.99890010998900103
    assert np.abs(gini(array) - score).item() < 0.01


def test_example_2():
    # For uniformly distributed random numbers, it will be low, around 0.33:
    array = np.random.uniform(-1, 0, 1000)
    array_ = array[:]

    out = gini(array)
    assert (0 <= out).all() and (out <= 0.5).all()

    # Check array was not changed in place
    assert (array == array_).all()


def test_example_3():
    # For a homogeneous sample, the Gini coefficient is 0.0:
    array = np.ones((1000))
    score = 0.0
    assert np.abs(gini(array) - score).item() < 0.01


def test_int():
    score = 0.8641975328641904
    array = [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000]
    array = np.array(array)

    assert np.abs(gini(array) - score).item() < 0.01

    array -= 1_000

    assert np.abs(gini(array) - score).item() < 0.01


def test_random_and_large():
    array = [random.randint(0, 10) for _ in range(100)] + [100_000_000]
    array = np.array(array)

    assert gini(array).item() > 0.98


def test_random_vector():
    shape = (2, 100)
    array = [[random.randint(0, 10) for _ in range(shape[1])] for _ in range(shape[0])]
    array = np.array(array)

    out = gini(array, axis=0)
    assert out.shape[0] == shape[1]
    assert (0 <= out).all() and (out <= 1).all()

    out = gini(array)
    assert out.shape[0] == shape[0]
    assert (0 <= out).all() and (out <= 1).all()
