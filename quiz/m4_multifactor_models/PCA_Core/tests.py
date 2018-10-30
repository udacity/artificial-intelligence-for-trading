import collections
from collections import OrderedDict
import copy
import pandas as pd
import numpy as np
from datetime import date, timedelta

def _generate_output_error_msg(fn_name, fn_inputs, fn_outputs, fn_expected_outputs):
    formatted_inputs = []
    formatted_outputs = []
    formatted_expected_outputs = []

    for input_name, input_value in fn_inputs.items():
        formatted_outputs.append('INPUT {}:\n{}\n'.format(
            input_name, str(input_value)))
    for output_name, output_value in fn_outputs.items():
        formatted_outputs.append('OUTPUT {}:\n{}\n'.format(
            output_name, str(output_value)))
    for expected_output_name, expected_output_value in fn_expected_outputs.items():
        formatted_expected_outputs.append('EXPECTED OUTPUT FOR {}:\n{}\n'.format(
            expected_output_name, str(expected_output_value)))

    return 'Wrong value for {}.\n' \
           '{}\n' \
           '{}\n' \
           '{}' \
        .format(
            fn_name,
            '\n'.join(formatted_inputs),
            '\n'.join(formatted_outputs),
            '\n'.join(formatted_expected_outputs))

def project_test(func):
    def func_wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        print('Tests Passed')
        return result
    return func_wrapper
    
    
def assert_output(fn, fn_inputs, fn_expected_outputs, **kwargs):
    if kwargs is not None:
        if 'atol' in kwargs:
            atol = kwargs['atol']
    
    assert type(fn_expected_outputs) == OrderedDict

    fn_outputs = OrderedDict()
    fn_inputs_passed_in = copy.deepcopy(fn_inputs)
    fn_raw_out = fn(**fn_inputs_passed_in)

    # Check if inputs have changed
    for input_name, input_value in fn_inputs.items():
        passed_in_unchanged = _is_equal(input_value, fn_inputs_passed_in[input_name])

        assert passed_in_unchanged, 'Input parameter "{}" has been modified inside the function. ' \
                                    'The function shouldn\'t modify the function parameters.'.format(input_name)

    if len(fn_expected_outputs) == 1:
        fn_outputs[list(fn_expected_outputs)[0]] = fn_raw_out
    elif len(fn_expected_outputs) > 1:
        assert type(fn_raw_out) == tuple,\
            'Expecting function to return tuple, got type {}'.format(type(fn_raw_out))
        assert len(fn_raw_out) == len(fn_expected_outputs),\
            'Expected {} outputs in tuple, only found {} outputs'.format(len(fn_expected_outputs), len(fn_raw_out))
        for key_i, output_key in enumerate(fn_expected_outputs.keys()):
            fn_outputs[output_key] = fn_raw_out[key_i]

    err_message = _generate_output_error_msg(
        fn.__name__,
        fn_inputs,
        fn_outputs,
        fn_expected_outputs)

    for fn_out, (out_name, expected_out) in zip(fn_outputs.values(), fn_expected_outputs.items()):
        assert isinstance(fn_out, type(expected_out)),\
            'Wrong type for output {}. Got {}, expected {}'.format(out_name, type(fn_out), type(expected_out))

        if hasattr(expected_out, 'shape'):
            assert fn_out.shape == expected_out.shape, \
                'Wrong shape for output {}. Got {}, expected {}'.format(out_name, fn_out.shape, expected_out.shape)
        elif hasattr(expected_out, '__len__'):
            assert len(fn_out) == len(expected_out), \
                'Wrong len for output {}. Got {}, expected {}'.format(out_name, len(fn_out), len(expected_out))

        if type(expected_out) == pd.DataFrame:
            assert set(fn_out.columns) == set(expected_out.columns), \
                'Incorrect columns for output {}\n' \
                'COLUMNS:          {}\n' \
                'EXPECTED COLUMNS: {}'.format(out_name, sorted(fn_out.columns), sorted(expected_out.columns))

            for column in expected_out.columns:
                assert fn_out[column].dtype == expected_out[column].dtype, \
                    'Incorrect type for output {}, column {}\n' \
                    'Type:          {}\n' \
                    'EXPECTED Type: {}'.format(out_name, column, fn_out[column].dtype, expected_out[column].dtype)

            # Sort Columns
            fn_out = fn_out.sort_index(1)
            expected_out = expected_out.sort_index(1)

        if type(expected_out) in {pd.DataFrame, pd.Series}:
            assert set(fn_out.index) == set(expected_out.index), \
                'Incorrect indices for output {}\n' \
                'INDICES:          {}\n' \
                'EXPECTED INDICES: {}'.format(out_name, sorted(fn_out.index), sorted(expected_out.index))

            # Sort Indices
            fn_out = fn_out.sort_index()
            expected_out = expected_out.sort_index()

        try:
            out_is_close = np.isclose(fn_out, expected_out, equal_nan=True, atol=atol)
        except TypeError:
            out_is_close = fn_out == expected_out
        else:
            if isinstance(expected_out, collections.Iterable):
                out_is_close = out_is_close.all()

        assert out_is_close, err_message

    