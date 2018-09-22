from collections import OrderedDict
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller
from tests import project_test, assert_output

import numpy as np
import pandas as pd


@project_test
def test_is_spread_stationary(fn):

    # fix random generator so it's easier to reproduce results
    np.random.seed(2018)
    # use returns to create a price series
    drift = 100
    r1 = np.random.normal(0, 1, 1000) 
    s1 = pd.Series(np.cumsum(r1), name='s1') + drift

    #make second series
    offset = 10
    noise = np.random.normal(0, 1, 1000)
    s2 = s1 + offset + noise
    s2.name = 's2'

    ## hedge ratio
    lr = LinearRegression()
    lr.fit(s1.values.reshape(-1,1),s2.values.reshape(-1,1))
    hedge_ratio = lr.coef_[0][0]

    #spread
    spread = s2 - s1 * hedge_ratio
    p_level = 0.05
    adf_result = adfuller(spread)
    pvalue = adf_result[1]
    retval = pvalue >= 0.05

    fn_inputs = {
        'spread': spread,
        'p_level': p_level
        }

    fn_correct_outputs = OrderedDict([
        ('retval',True)
        ])

    assert_output(fn, fn_inputs, fn_correct_outputs)
