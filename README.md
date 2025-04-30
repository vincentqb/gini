This is a function that calculates the Gini coefficient of a (numpy or pytorch) array along a given axis. This is a vectorized implementation derived from [Olivia Guest's](https://github.com/oliviaguest/gini), which is based on the third equation from [here](http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm), which defines the Gini coefficient as:

$$ G = {\\frac{ \\sum\_{i=1}^{n} (2i - n - 1) x_i}{n \\sum\_{i=1}^{n} x_i}} $$

For a very unequal sample, 999 zeros and a single one, the Gini coefficient is very close to 1.0:

```
import numpy as np
from gini import gini_numpy as gini

array = np.zeros((1000))
array[0] = 1.0
gini(array)

# np.float64(0.998900109989001)
```

For uniformly distributed random numbers, it will be low, around 0.33:

```
array = np.random.uniform(-1,0,1000)
gini(array)

# np.float64(0.33020664112202275)
```

For a homogeneous sample, the Gini coefficient is 0.0:

```
array = np.ones((1000))
gini(array)

# np.float64(-6.938893903907231e-17)
```

Gini coefficients are often used to quantify income inequality, read more [here](http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm).
