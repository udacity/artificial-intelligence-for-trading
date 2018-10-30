from collections import OrderedDict
import pandas as pd
import numpy as np
from scipy.stats import norm
from tests import project_test, assert_output

@project_test
def test_direction_of_first_PC(fn):
    atol = 10 # set tolerance of numpy.isclose value checking in units of answer (degrees)
    
    # regenerate data
    num_data = 5
    X = norm.rvs(size=(num_data,2), random_state=4)*2
    X = np.dot(X, np.linalg.cholesky([[1, 0.8], [0.8, 0.8]]))
    m = X.mean(axis=0)
    X = X - m

    # calc direction of first eigenvalue
    _, b = np.linalg.eig(np.cov(X.T));
    answer = np.arctan(b[1,0]/b[0,0])*(180/np.pi)
    
    fn_inputs = {}
    fn_correct_outputs = OrderedDict([
            ('first_pc_angle', answer)])
    
    # wrapper for student code output to mod by 180 degrees
    def apply_to_output(fn):
        def wrapper_function():
            return np.mod(fn(), 180)
        wrapper_function.__name__ = fn.__name__
        return wrapper_function
    
    fn = apply_to_output(fn)

    assert_output(fn, fn_inputs, fn_correct_outputs, atol=atol)