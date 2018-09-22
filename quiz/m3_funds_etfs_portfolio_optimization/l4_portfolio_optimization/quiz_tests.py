from collections import OrderedDict
import numpy as np
import cvxpy as cvx
from tests import project_test, generate_random_tickers, generate_random_dates, assert_output
import string

@project_test
def test_optimize_twoasset_portfolio(fn):

    varA,varB,rAB = 0.1, 0.05, 0.25
    cov = np.sqrt(varA)*np.sqrt(varB)*rAB
    x = cvx.Variable(2)
    P = np.array([[varA, cov],[cov, varB]])
    quad_form = cvx.quad_form(x,P)
    objective = cvx.Minimize(quad_form)
    constraints = [sum(x)==1]
    problem = cvx.Problem(objective, constraints)
    min_value = problem.solve()
    xA,xB = x.value

    fn_inputs = {
        'varA': varA,
        'varB': varB,
        'rAB': rAB
        }

    fn_correct_outputs = OrderedDict([
        ('xA',xA),
        ('xB',xB)
        ])

    assert_output(fn, fn_inputs, fn_correct_outputs)
