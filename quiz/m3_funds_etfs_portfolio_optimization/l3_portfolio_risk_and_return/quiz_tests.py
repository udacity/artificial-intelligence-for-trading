from collections import OrderedDict
import numpy as np
from tests import project_test, generate_random_tickers, generate_random_dates, assert_output
import string

@project_test
def test_covariance_matrix(fn):
    """Test with a 3 simulated stock return series"""
    days_per_year = 252
    years = 3
    total_days = days_per_year * years

    return_market = np.random.normal(loc=0.05, scale=0.3, size=days_per_year)
    return_1 = np.random.uniform(low=-0.000001, high=.000001, size=days_per_year) + return_market
    return_2 = np.random.uniform(low=-0.000001, high=.000001, size=days_per_year) + return_market
    return_3 = np.random.uniform(low=-0.000001, high=.000001, size=days_per_year) + return_market
    returns = np.array([return_1, return_2, return_3])

    cov = np.cov(returns)

    fn_inputs = {
        'returns': returns
        }

    fn_correct_outputs = OrderedDict([
        ('cov',cov)
        ])

    assert_output(fn, fn_inputs, fn_correct_outputs)
