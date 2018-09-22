from collections import OrderedDict
import numpy as np
import cvxpy as cvx
from tests import project_test, generate_random_tickers, generate_random_dates, assert_output
import string

@project_test
def test_optimize_portfolio(fn):
    """Test with a 3 simulated stock return series"""
    days_per_year = 252
    years = 3
    total_days = days_per_year * years

    return_market = np.random.normal(loc=0.05, scale=0.3, size=days_per_year)
    return_1 = np.random.uniform(low=-0.000001, high=.000001, size=days_per_year) + return_market
    return_2 = np.random.uniform(low=-0.000001, high=.000001, size=days_per_year) + return_market
    return_3 = np.random.uniform(low=-0.000001, high=.000001, size=days_per_year) + return_market
    returns = np.array([return_1, return_2, return_3])

    """simulate index weights"""
    index_weights = np.array([0.9,0.15,0.05])
    
    scale = .00001

    m = returns.shape[0]
    cov = np.cov(returns)
    x = cvx.Variable(m)
    portfolio_variance = cvx.quad_form(x, cov)
    distance_to_index = cvx.norm(x - index_weights)
    objective = cvx.Minimize(portfolio_variance + scale* distance_to_index)
    constraints = [x >= 0, sum(x) == 1]
    problem = cvx.Problem(objective, constraints).solve()
    x_values = x.value

    fn_inputs = {
        'returns': returns,
        'index_weights': index_weights,
        'scale': scale
        }

    fn_correct_outputs = OrderedDict([
        ('x_values',x_values)
        ])

    assert_output(fn, fn_inputs, fn_correct_outputs)
