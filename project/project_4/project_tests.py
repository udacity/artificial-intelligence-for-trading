from collections import OrderedDict
import cvxpy as cvx
import numpy as np
import pandas as pd

from unittest.mock import patch
from sklearn.decomposition import PCA
from zipline.data import bundles
from zipline.pipeline import Pipeline
from zipline.pipeline.factors import AverageDollarVolume
from zipline.utils.calendars import get_calendar

import project_helper
from tests import assert_output, project_test, generate_random_dates, assert_structure, does_data_match


def get_assets(ticker_count):
    bundle = bundles.load('eod-quotemedia')
    return bundle.asset_finder.retrieve_all(bundle.asset_finder.sids[:ticker_count])


@project_test
def test_fit_pca(fn):
    dates = generate_random_dates(4)
    assets = get_assets(3)

    fn_inputs = {
        'returns': pd.DataFrame(
            [
                [0.02769242, 1.34872387, 0.23460972],
                [-0.94728692, 0.68386883, -1.23987235],
                [1.93769376, -0.48275934, 0.34957348],
                [0.23985234, 0.35897345, 0.34598734]],
            dates, assets),
        'num_factor_exposures': 2,
        'svd_solver': 'full'}
    fn_correct_values = {
        'PCA': PCA(),
        'PCA.components_': np.array([
            [0.81925896, -0.40427891, 0.40666118],
            [-0.02011128, 0.68848693, 0.72496985]])}

    pca_fit = PCA.fit
    with patch.object(PCA, 'fit', autospec=True) as mock_fit:
        mock_fit.side_effect = pca_fit

        fn_return_value = fn(**fn_inputs)

        assert_structure(fn_return_value, fn_correct_values['PCA'], 'PCA')

        try:
            fn_return_value.fit.assert_called()
        except AssertionError:
            raise Exception('Test Failure: PCA.fit not called')

        try:
            fn_return_value.fit.assert_called_with(self=fn_return_value, X=fn_inputs['returns'])
        except Exception:
            raise Exception('Test Failure: PCA.fit called with the wrong arguments')

        assert_structure(fn_return_value.components_, fn_correct_values['PCA.components_'], 'PCA.components_')

        if not does_data_match(fn_return_value.components_, fn_correct_values['PCA.components_']):
            raise Exception('Test Failure: PCA not fitted correctly\n\n'
                            'PCA.components_:\n'
                            '{}\n\n'
                            'Expected PCA.components_:\n'
                            '{}'.format(fn_return_value.components_, fn_correct_values['PCA.components_']))


@project_test
def test_factor_betas(fn):
    n_components = 3
    dates = generate_random_dates(4)
    assets = get_assets(3)

    pca = PCA(n_components)
    pca.fit(pd.DataFrame(
        [
            [0.21487253,  0.12342312, -0.13245215],
            [0.23423439, -0.23434532,  1.67834324],
            [0.23432445, -0.23563226,  0.23423523],
            [0.24824535, -0.23523435,  0.36235236]],
        dates, assets))

    fn_inputs = {
        'pca': pca,
        'factor_beta_indices': np.array(assets),
        'factor_beta_columns': np.arange(n_components)}
    fn_correct_outputs = OrderedDict([
        (
            'factor_betas', pd.DataFrame(
                [
                    [ 0.00590170, -0.07759542, 0.99696746],
                    [-0.13077609,  0.98836246, 0.07769983],
                    [ 0.99139436,  0.13083807, 0.00431461]],
                fn_inputs['factor_beta_indices'],
                fn_inputs['factor_beta_columns']))])

    assert_output(fn, fn_inputs, fn_correct_outputs, check_parameter_changes=False)


@project_test
def test_factor_returns(fn):
    n_components = 3
    dates = generate_random_dates(4)
    assets = get_assets(3)

    pca = PCA(n_components)
    pca.fit(pd.DataFrame(
        [
            [0.21487253,  0.12342312, -0.13245215],
            [0.23423439, -0.23434532,  1.67834324],
            [0.23432445, -0.23563226,  0.23423523],
            [0.24824535, -0.23523435,  0.36235236]],
        dates, assets))

    fn_inputs = {
        'pca': pca,
        'returns': pd.DataFrame(
            [
                [0.02769242,  1.34872387,  0.23460972],
                [-0.94728692, 0.68386883, -1.23987235],
                [1.93769376, -0.48275934,  0.34957348],
                [0.23985234,  0.35897345,  0.34598734]],
            dates, assets),
        'factor_return_indices': np.array(dates),
        'factor_return_columns': np.arange(n_components)}
    fn_correct_outputs = OrderedDict([
        (
            'factor_returns', pd.DataFrame(
                [
                    [-0.49503261,  1.45332369, -0.08980631],
                    [-1.87563271,  0.67894147, -1.11984992],
                    [-0.13027172, -0.49001128,  1.67259298],
                    [-0.25392567,  0.47320133,  0.04528734]],
                fn_inputs['factor_return_indices'],
                fn_inputs['factor_return_columns']))])

    assert_output(fn, fn_inputs, fn_correct_outputs, check_parameter_changes=False)


@project_test
def test_factor_cov_matrix(fn):
    dates = generate_random_dates(4)

    fn_inputs = {
        'factor_returns': pd.DataFrame([
                [-0.49503261,  1.45332369, -0.08980631],
                [-1.87563271,  0.67894147, -1.11984992],
                [-0.13027172, -0.49001128,  1.67259298],
                [-0.25392567,  0.47320133,  0.04528734]],
                dates),
        'ann_factor': 250}
    fn_correct_outputs = OrderedDict([
        (
            'factor_cov_matrix', np.array([
                [162.26559808, 0.0, 0.0],
                [0.0, 159.86284454, 0.0],
                [0.0, 0.0, 333.09785876]]))])

    assert_output(fn, fn_inputs, fn_correct_outputs)


@project_test
def test_idiosyncratic_var_matrix(fn):
    dates = generate_random_dates(4)
    assets = get_assets(3)

    fn_inputs = {
        'returns': pd.DataFrame(
            [
                [ 0.02769242,  1.34872387,  0.23460972],
                [-0.94728692,  0.68386883, -1.23987235],
                [ 1.93769376, -0.48275934,  0.34957348],
                [ 0.23985234,  0.35897345,  0.34598734]],
            dates, assets),
        'factor_returns': pd.DataFrame([
                [-0.49503261,  1.45332369, -0.08980631],
                [-1.87563271,  0.67894147, -1.11984992],
                [-0.13027172, -0.49001128,  1.67259298],
                [-0.25392567,  0.47320133,  0.04528734]],
                dates),
        'factor_betas': pd.DataFrame([
            [ 0.00590170, -0.07759542, 0.99696746],
            [-0.13077609,  0.98836246, 0.07769983],
            [ 0.99139436,  0.13083807, 0.00431461]]),
        'ann_factor': 250}
    fn_correct_outputs = OrderedDict([
        (
            'idiosyncratic_var_matrix', pd.DataFrame(np.full([3,3], 0.0), assets, assets))])

    assert_output(fn, fn_inputs, fn_correct_outputs)


@project_test
def test_idiosyncratic_var_vector(fn):
    dates = generate_random_dates(4)
    assets = get_assets(3)

    fn_inputs = {
        'returns': pd.DataFrame(
            [
                [ 0.02769242,  1.34872387,  0.23460972],
                [-0.94728692,  0.68386883, -1.23987235],
                [ 1.93769376, -0.48275934,  0.34957348],
                [ 0.23985234,  0.35897345,  0.34598734]],
            dates, assets),
        'idiosyncratic_var_matrix': pd.DataFrame([
                [0.02272535,  0.0, 0.0],
                [0.0,  0.05190083, 0.0],
                [0.0, -0.49001128,  0.05431181]],
            assets, assets),}
    fn_correct_outputs = OrderedDict([
        (
            'idiosyncratic_var_vector', pd.DataFrame([0.02272535, 0.05190083, 0.05431181], assets))])

    assert_output(fn, fn_inputs, fn_correct_outputs)


@project_test
def test_predict_portfolio_risk(fn):
    assets = get_assets(3)

    fn_inputs = {
        'factor_betas': pd.DataFrame([
            [-0.04316847, 0.01955111, -0.00993375,  0.01054038],
            [-0.05874471, 0.19637679,  0.07868756,  0.08209582],
            [-0.03433256, 0.03451503,  0.01133839, -0.02543666]],
            assets),
        'factor_cov_matrix': np.diag([14.01830425, 1.10591127, 0.77099145, 0.18725609]),
        'idiosyncratic_var_matrix': pd.DataFrame(np.diag([0.02272535, 0.05190083, 0.03040361]), assets, assets),
        'weights': pd.DataFrame([0.0, 0.0, 0.25], assets)}
    fn_correct_outputs = OrderedDict([
        (
            'portfolio_risk_prediction', 0.0550369570517)])

    assert_output(fn, fn_inputs, fn_correct_outputs)


@project_test
def test_mean_reversion_5day_sector_neutral(fn):
    column_name = 'Mean_Reversion_5Day_Sector_Neutral'
    start_date_str = '2015-01-05'
    end_date_str = '2015-01-07'

    # Build engine
    trading_calendar = get_calendar('NYSE')
    bundle_data = bundles.load(project_helper.EOD_BUNDLE_NAME)
    engine = project_helper.build_pipeline_engine(bundle_data, trading_calendar)

    # Build pipeline
    universe_window_length = 2
    universe_asset_count = 4
    universe = AverageDollarVolume(window_length=universe_window_length).top(universe_asset_count)
    pipeline = Pipeline(screen=universe)

    run_pipeline_args = {
        'pipeline': pipeline,
        'start_date': pd.Timestamp(start_date_str, tz='utc'),
        'end_date': pd.Timestamp(end_date_str, tz='utc')}
    fn_inputs = {
        'window_length': 3,
        'universe': universe,
        'sector': project_helper.Sector()}
    fn_correct_outputs = OrderedDict([
        (
            'pipline_out', pd.DataFrame(
                [1.34164079, 0.44721360, -1.34164079, -0.44721360,
                 1.34164079, 0.44721360, -1.34164079, -0.44721360,
                 -1.34164079, 0.44721360, 1.34164079, -0.44721360],
                engine.run_pipeline(**run_pipeline_args).index,
                [column_name]))])

    print('Running Integration Test on pipeline:')
    print('> start_dat = pd.Timestamp(\'{}\', tz=\'utc\')'.format(start_date_str))
    print('> end_date = pd.Timestamp(\'{}\', tz=\'utc\')'.format(end_date_str))
    print('> universe = AverageDollarVolume(window_length={}).top({})'.format(
        universe_window_length, universe_asset_count))
    print('> factor = {}('.format(fn.__name__))
    print('    window_length={},'.format(fn_inputs['window_length']))
    print('    universe=universe,')
    print('    sector=project_helper.Sector())')
    print('> pipeline.add(factor, \'{}\')'.format(column_name))
    print('> engine.run_pipeline(pipeline, start_dat, end_date)')
    print('')

    pipeline.add(fn(**fn_inputs), column_name)
    assert_output(engine.run_pipeline, run_pipeline_args, fn_correct_outputs, check_parameter_changes=False)


@project_test
def test_mean_reversion_5day_sector_neutral_smoothed(fn):
    column_name = 'Mean_Reversion_5Day_Sector_Neutral_Smoothed'
    start_date_str = '2015-01-05'
    end_date_str = '2015-01-07'

    # Build engine
    trading_calendar = get_calendar('NYSE')
    bundle_data = bundles.load(project_helper.EOD_BUNDLE_NAME)
    engine = project_helper.build_pipeline_engine(bundle_data, trading_calendar)

    # Build pipeline
    universe_window_length = 2
    universe_asset_count = 4
    universe = AverageDollarVolume(window_length=universe_window_length).top(universe_asset_count)
    pipeline = Pipeline(screen=universe)

    run_pipeline_args = {
        'pipeline': pipeline,
        'start_date': pd.Timestamp(start_date_str, tz='utc'),
        'end_date': pd.Timestamp(end_date_str, tz='utc')}
    fn_inputs = {
        'window_length': 3,
        'universe': universe,
        'sector': project_helper.Sector()}
    fn_correct_outputs = OrderedDict([
        (
            'pipline_out', pd.DataFrame(
                [0.44721360, 1.34164079, -1.34164079, -0.44721360,
                 1.34164079, 0.44721360, -1.34164079, -0.44721360,
                 0.44721360, 1.34164079, -1.34164079, -0.44721360],
                engine.run_pipeline(**run_pipeline_args).index,
                [column_name]))])

    print('Running Integration Test on pipeline:')
    print('> start_dat = pd.Timestamp(\'{}\', tz=\'utc\')'.format(start_date_str))
    print('> end_date = pd.Timestamp(\'{}\', tz=\'utc\')'.format(end_date_str))
    print('> universe = AverageDollarVolume(window_length={}).top({})'.format(
        universe_window_length, universe_asset_count))
    print('> factor = {}('.format(fn.__name__))
    print('    window_length={},'.format(fn_inputs['window_length']))
    print('    universe=universe,')
    print('    sector=project_helper.Sector())')
    print('> pipeline.add(factor, \'{}\')'.format(column_name))
    print('> engine.run_pipeline(pipeline, start_dat, end_date)')
    print('')

    pipeline.add(fn(**fn_inputs), column_name)
    assert_output(engine.run_pipeline, run_pipeline_args, fn_correct_outputs, check_parameter_changes=False)


@project_test
def test_sharpe_ratio(fn):
    dates = generate_random_dates(4)
    factor_names = ['Factor {}'.format(i) for i in range(3)]

    fn_inputs = {
        'factor_returns': pd.DataFrame(
            [
                [ 0.00069242,  0.00072387,  0.00002972],
                [-0.00028692,  0.00086883, -0.00007235],
                [-0.00066376, -0.00045934,  0.00007348],
                [ 0.00085234,  0.00093345,  0.00008734]],
            dates, factor_names),
        'annualization_factor': 16.0}
    fn_correct_outputs = OrderedDict([
        (
            'sharpe_ratio', pd.Series([3.21339895, 12.59157330, 6.54485802], factor_names))])

    assert_output(fn, fn_inputs, fn_correct_outputs)


@project_test
def test_optimal_holdings_get_obj(cl):
    optimal_holdings = cl()
    alpha_vector = pd.DataFrame(
        [-0.58642457, -0.45333845, -0.69993898, -0.06790952],
        get_assets(4),
        ['alpha_vector'])

    fn_inputs = {
        'weights': cvx.Variable(len(alpha_vector)),
        'alpha_vector': alpha_vector}
    fn_correct_outputs = OrderedDict([
        (
            'solution', np.array([-3.33960455e-10, -2.75871416e-11, -5.00000000e-01, 5.00000000e-01]))])

    def solve_problem(weights, alpha_vector):
        constaints = [sum(weights) == 0.0, sum(cvx.abs(weights)) <= 1.0]
        obj = optimal_holdings._get_obj(weights, alpha_vector)
        prob = cvx.Problem(obj, constaints)
        prob.solve(max_iters=500)

        return np.asarray(weights.value).flatten()

    print('Running Integration Test on Problem.solve:')
    print('> constaints = [sum(weights) == 0.0, sum(cvx.abs(weights)) <= 1.0]')
    print('> obj = optimal_holdings._get_obj(weights, alpha_vector)')
    print('> prob = cvx.Problem(obj, constaints)')
    print('> prob.solve(max_iters=500)')
    print('> solution = np.asarray(weights.value).flatten()')
    print('')

    assert_output(solve_problem, fn_inputs, fn_correct_outputs, check_parameter_changes=False)


@project_test
def test_optimal_holdings_get_constraints(cl):
    optimal_holdings = cl()
    x_size = 3
    weights_size = 4

    fn_inputs = {
        'weights': cvx.Variable(weights_size)}
    fn_correct_outputs = OrderedDict([
        (
            'solution', np.array([-0.01095207, 0.0027576, 0.02684978, -0.01865519]))])

    def solve_problem(weights):
        x = np.diag(np.arange(x_size))
        s = np.diag(np.arange(weights_size))
        factor_betas = np.arange(weights_size * x_size).reshape([weights_size, x_size])
        risk = cvx.quad_form(weights * factor_betas, x) + cvx.quad_form(weights, s)
        constaints = optimal_holdings._get_constraints(weights, factor_betas, risk)
        obj = cvx.Maximize([0, 1, 5, -1] * weights)
        prob = cvx.Problem(obj, constaints)
        prob.solve(max_iters=500)

        return np.asarray(weights.value).flatten()

    print('\nRunning Integration Test on Problem.solve:')
    print('> x = np.diag(np.arange({}))'.format(x_size))
    print('> s = np.diag(np.arange({}))'.format(weights_size))
    print('> factor_betas = np.arange({} * {}).reshape([{}, {}])'.format(weights_size, x_size, weights_size, x_size))
    print('> risk = cvx.quad_form(weights * factor_betas, x) + cvx.quad_form(weights, s)')
    print('> constaints = optimal_holdings._get_constraints(weights, factor_betas, risk)')
    print('> obj = cvx.Maximize([0, 1, 5, -1] * weights)')
    print('> prob = cvx.Problem(obj, constaints)')
    print('> prob.solve(max_iters=500)')
    print('> solution = np.asarray(weights.value).flatten()')
    print('')

    assert_output(solve_problem, fn_inputs, fn_correct_outputs, check_parameter_changes=False)


@project_test
def test_optimal_holdings_regualization_get_obj(cl):
    optimal_holdings_regualization = cl()
    alpha_vector = pd.DataFrame(
        [-0.58642457, -0.45333845, -0.69993898, -0.06790952],
        get_assets(4),
        ['alpha_vector'])

    fn_inputs = {
        'weights': cvx.Variable(len(alpha_vector)),
        'alpha_vector': alpha_vector}
    fn_correct_outputs = OrderedDict([
        (
            'solution', np.array([-2.80288449e-10, -4.73562710e-12, -5.12563104e-10, 7.97632862e-10]))])

    def solve_problem(weights, alpha_vector):
        constaints = [sum(weights) == 0.0, sum(cvx.abs(weights)) <= 1.0]
        obj = optimal_holdings_regualization._get_obj(weights, alpha_vector)
        prob = cvx.Problem(obj, constaints)
        prob.solve(max_iters=500)

        return np.asarray(weights.value).flatten()

    print('Running Integration Test on Problem.solve:')
    print('> constaints = [sum(weights) == 0.0, sum(cvx.abs(weights)) <= 1.0]')
    print('> obj = optimal_holdings_regualization._get_obj(weights, alpha_vector)')
    print('> prob = cvx.Problem(obj, constaints)')
    print('> prob.solve(max_iters=500)')
    print('> solution = np.asarray(weights.value).flatten()')
    print('')

    assert_output(solve_problem, fn_inputs, fn_correct_outputs, check_parameter_changes=False)


@project_test
def test_optimal_holdings_strict_factor_get_obj(cl):
    optimal_holdings_strict_factor = cl()
    alpha_vector = pd.DataFrame(
        [-0.58642457, -0.45333845, -0.69993898, -0.06790952],
        get_assets(4),
        ['alpha_vector'])

    fn_inputs = {
        'weights': cvx.Variable(len(alpha_vector)),
        'alpha_vector': alpha_vector}
    fn_correct_outputs = OrderedDict([
        (
            'solution', np.array([-0.07441958, -0.00079418, -0.13721759, 0.21243135]))])

    def solve_problem(weights, alpha_vector):
        constaints = [sum(weights) == 0.0, sum(cvx.abs(weights)) <= 1.0]
        obj = optimal_holdings_strict_factor._get_obj(weights, alpha_vector)
        prob = cvx.Problem(obj, constaints)
        prob.solve(max_iters=500)

        return np.asarray(weights.value).flatten()

    print('Running Integration Test on Problem.solve:')
    print('> constaints = [sum(weights) == 0.0, sum(cvx.abs(weights)) <= 1.0]')
    print('> obj = optimal_holdings_strict_factor._get_obj(weights, alpha_vector)')
    print('> prob = cvx.Problem(obj, constaints)')
    print('> prob.solve(max_iters=500)')
    print('> solution = np.asarray(weights.value).flatten()')
    print('')

    assert_output(solve_problem, fn_inputs, fn_correct_outputs, check_parameter_changes=False)

