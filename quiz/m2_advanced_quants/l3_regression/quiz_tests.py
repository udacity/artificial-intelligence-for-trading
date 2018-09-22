from collections import OrderedDict
import numpy as np
import pandas as pd
import scipy.stats as stats
from tests import project_test, generate_random_tickers, generate_random_dates, assert_output


import numpy as np
import pandas as pd
import string

@project_test
def test_is_normal_ks(fn):

    sample_normal = stats.norm.rvs(loc=0.0, scale=1.0, size=(1000,))
    fn_inputs = {
        'sample': sample_normal}

    fn_correct_outputs = OrderedDict([
        ('normal',np.True_)
        ])

    assert_output(fn, fn_inputs, fn_correct_outputs)


    sample_not_normal = stats.lognorm.rvs(s=0.5, loc=0.0, scale=1.0, size=(1000,))
    fn_inputs = {
        'sample': sample_not_normal}

    fn_correct_outputs = OrderedDict([
        ('not_normal',np.False_)
        ])

    assert_output(fn, fn_inputs, fn_correct_outputs)
