This is a function that calculates the Gini coefficient of a numpy array along a given axis. The function in `gini.py` is a fork of [Olivia Guest's](https://github.com/oliviaguest/gini), which is based on the third equation from [here](http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm), which defines the Gini coefficient as:

$$ G = {\\frac{ \\sum\_{i=1}^{n} (2i - n - 1) x_i}{n \\sum\_{i=1}^{n} x_i}} $$

For a very unequal sample, 999 zeros and a single one, the Gini coefficient is very close to 1.0:

```
>>> from gini import *
>>> a = np.zeros((1000))
>>> a[0] = 1.0
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

Gini coefficients are often used to quantify income inequality, read more [here](http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm).
