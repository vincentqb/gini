# gini

A Gini coefficient calculator in Python.

## Overview

This is a function that calculates the Gini coefficient of a numpy array. Gini coefficients are often used to quantify income inequality, read more [here](http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm).

The function in `gini.py` is based on the third equation from [here](http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm), which defines the Gini coefficient as:

![G = frac{ um_{i=1}^{n} (2i - n - 1) x_i}{n  um_{i=1}^{n} x_i}](https://github.com/oliviaguest/gini/raw/master/gini.png "Gini equation")

## Examples

For a very unequal sample, 999 zeros and a single one,

```
>>> from gini import *
>>> a = np.zeros((1000))
>>> a[0] = 1.0
```

the Gini coefficient is very close to 1.0:

```
>>> gini(a)
0.99890010998900103
```

For uniformly distributed random numbers, it will be low, around 0.33:

```
>>> s = np.random.uniform(-1,0,1000)
>>> gini(s)
0.3295183767105907
```

For a homogeneous sample, the Gini coefficient is 0.0:

```
>>> b = np.ones((1000))
>>> gini(b)
0.0
```

## Notes

- It is significantly faster than (the [current implementation of](https://github.com/pysal/pysal/issues/855)) PySAL's Gini coefficient function (see [pysal.inequality.gini](http://pysal.readthedocs.io/en/latest/_modules/pysal/inequality/gini.html)) and outputs are indistinguishable before approximately 6 decimal places. In other words, the two functions are arithmetically identical.

- It is slightly faster than the [Gini coefficient function by David on Ellipsix](http://www.ellipsix.net/blog/2012/11/the-gini-coefficient-for-distribution-inequality.html).

Many other Gini coefficient functions found online do not produce equivalent results, hence why I wrote this.
