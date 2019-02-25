from collections import OrderedDict
import numpy as np
import pandas as pd

from sklearn.ensemble import BaggingClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from unittest.mock import patch
from zipline.data import bundles

from tests import assert_output, project_test, generate_random_dates, assert_structure


def get_assets(ticker_count):
    bundle = bundles.load('eod-quotemedia')
    return bundle.asset_finder.retrieve_all(bundle.asset_finder.sids[:ticker_count])


@project_test
def test_train_valid_test_split(fn):
    columns = ['test column 1', 'test column 2', 'test column 3']
    dates = generate_random_dates(10)
    assets = get_assets(3)
    index = pd.MultiIndex.from_product([dates, assets])
    values = np.arange(len(index) * len(columns)).reshape([len(columns), len(index)]).T
    targets = np.arange(len(index))

    fn_inputs = {
        'all_x': pd.DataFrame(values, index, columns),
        'all_y': pd.Series(targets, index, name='target'),
        'train_size': 0.6,
        'valid_size': 0.2,
        'test_size': 0.2}
    fn_correct_outputs = OrderedDict([
        ('X_train', pd.DataFrame(values[:18], index[:18], columns=columns)),
        ('X_valid', pd.DataFrame(values[18:24], index[18:24], columns=columns)),
        ('X_test', pd.DataFrame(values[24:], index[24:], columns=columns)),
        ('y_train', pd.Series(targets[:18], index[:18])),
        ('y_valid', pd.Series(targets[18:24], index[18:24])),
        ('y_test', pd.Series(targets[24:], index[24:]))])

    assert_output(fn, fn_inputs, fn_correct_outputs, check_parameter_changes=False)


@project_test
def test_non_overlapping_samples(fn):
    columns = ['test column 1', 'test column 2']
    dates = generate_random_dates(8)
    assets = get_assets(3)
    index = pd.MultiIndex.from_product([dates, assets])
    values = np.arange(len(index) * len(columns)).reshape([len(columns), len(index)]).T
    targets = np.arange(len(index))

    fn_inputs = {
        'x': pd.DataFrame(values, index, columns),
        'y': pd.Series(targets, index),
        'n_skip_samples': 2,
        'start_i': 1}

    new_index = pd.MultiIndex.from_product([dates[fn_inputs['start_i']::fn_inputs['n_skip_samples'] + 1], assets])
    fn_correct_outputs = OrderedDict([
        (
            'non_overlapping_x',
            pd.DataFrame(
                [
                    [3, 27],
                    [4, 28],
                    [5, 29],
                    [12, 36],
                    [13, 37],
                    [14, 38],
                    [21, 45],
                    [22, 46],
                    [23, 47]],
                new_index, columns)),
        (
            'non_overlapping_y',
            pd.Series([3, 4, 5, 12, 13, 14, 21, 22, 23], new_index))])

    assert_output(fn, fn_inputs, fn_correct_outputs, check_parameter_changes=False)


@project_test
def test_bagging_classifier(fn):
    n_estimators = 200
    parameters = {
            'criterion': 'entropy',
            'min_samples_leaf': 2500,
            'oob_score': True,
            'n_jobs': -1,
            'random_state': 0}

    fn_inputs = {
        'n_estimators': n_estimators,
        'max_samples': 0.2,
        'max_features': 1.0,
        'parameters': parameters}

    return_value = fn(**fn_inputs)

    assert isinstance(return_value, BaggingClassifier),\
        'Returned object is wrong. It should be a BaggingClassifier.'
    assert return_value.max_samples == fn_inputs['max_samples'],\
        'BaggingClassifier\'s max_samples is the wrong value.'
    assert return_value.max_features == fn_inputs['max_features'],\
        'BaggingClassifier\'s max_features is the wrong value.'
    assert return_value.oob_score == parameters['oob_score'],\
        'BaggingClassifier\'s oob_score is the wrong value.'
    assert return_value.n_jobs == parameters['n_jobs'],\
        'BaggingClassifier\'s n_jobs is the wrong value.'
    assert return_value.random_state == parameters['random_state'],\
        'BaggingClassifier\'s random_state is the wrong value.'

    assert isinstance(return_value.base_estimator, DecisionTreeClassifier),\
        'BaggingClassifier\'s base estimator is the wrong value type. It should be a DecisionTreeClassifier.'
    assert return_value.base_estimator.criterion == parameters['criterion'],\
        'The base estimator\'s criterion is the wrong value.'
    assert return_value.base_estimator.min_samples_leaf == parameters['min_samples_leaf'],\
        'The base estimator\'s min_samples_leaf is the wrong value.'


@project_test
def test_calculate_oob_score(fn):
    n_estimators = 3
    n_features = 2
    n_samples = 1000

    noise = np.random.RandomState(0).random_sample([3, n_samples]) * n_samples
    x = np.arange(n_estimators * n_samples * n_features).reshape([n_estimators, n_samples, n_features])
    y = np.sum(x, axis=-1) + noise
    estimators = [
        RandomForestRegressor(300, oob_score=True, n_jobs=-1, random_state=101).fit(x[estimator_i], y[estimator_i])
        for estimator_i in range(n_estimators)]

    fn_inputs = {
        'classifiers': estimators}
    fn_correct_outputs = OrderedDict([('oob_score', 0.911755651666)])

    assert_output(fn, fn_inputs, fn_correct_outputs, check_parameter_changes=False)


@project_test
def test_non_overlapping_estimators(fn):
    n_estimators = 3
    columns = ['test column 1', 'test column 2']
    dates = generate_random_dates(8)
    assets = get_assets(3)
    index = pd.MultiIndex.from_product([dates, assets])
    noise = np.random.RandomState(0).random_sample([len(index)]) * len(index)
    values = np.arange(len(index) * len(columns)).reshape([len(columns), len(index)]).T
    targets = np.sum(values, axis=-1) + noise

    classifiers = [
        RandomForestRegressor(300, oob_score=True, n_jobs=-1, random_state=101)
        for _ in range(n_estimators)]

    fn_inputs = {
        'x': pd.DataFrame(values, index, columns),
        'y': pd.Series(targets, index),
        'classifiers': classifiers,
        'n_skip_samples': 3}

    random_forest_regressor_fit = RandomForestRegressor.fit
    with patch.object(RandomForestRegressor, 'fit', autospec=True) as mock_fit:
        mock_fit.side_effect = random_forest_regressor_fit
        fn_return_value = fn(**fn_inputs)

        assert_structure(fn_return_value, [RandomForestRegressor for _ in range(n_estimators)], 'PCA')

        for classifier in fn_return_value:
            try:
                classifier.fit.assert_called()
            except AssertionError:
                raise Exception('Test Failure: RandomForestRegressor.fit not called on all classifiers')
