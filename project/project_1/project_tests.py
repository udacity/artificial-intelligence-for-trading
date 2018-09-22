from collections import OrderedDict
import pandas as pd
import numpy as np

from tests import generate_random_tickers, assert_output, project_test


@project_test
def test_resample_prices(fn):
    tickers = generate_random_tickers(5)
    dates = pd.DatetimeIndex(['2008-08-19', '2008-09-08', '2008-09-28', '2008-10-18', '2008-11-07', '2008-11-27'])
    resampled_dates = pd.DatetimeIndex(['2008-08-31', '2008-09-30', '2008-10-31', '2008-11-30'])

    fn_inputs = {
        'close_prices': pd.DataFrame(
            [
                [21.050810483942833, 17.013843810658827, 10.984503755486879, 11.248093428369392, 12.961712733997235],
                [15.63570258751384, 14.69054309070934, 11.353027688995159, 475.74195118202061, 11.959640427803022],
                [482.34539247360806, 35.202580592515041, 3516.5416782257166, 66.405314327318209, 13.503960481087077],
                [10.918933017418304, 17.9086438675435, 24.801265417692324, 12.488954191854916, 10.52435923388642],
                [10.675971965144655, 12.749401436636365, 11.805257579935713, 21.539039489843024, 19.99766036804861],
                [11.545495378369814, 23.981468434099405, 24.974763062186504, 36.031962102997689, 14.304332320024963]],
            dates, tickers),
        'freq': 'M'}
    fn_correct_outputs = OrderedDict([
        (
            'prices_resampled',
            pd.DataFrame(
                [
                        [21.05081048, 17.01384381, 10.98450376, 11.24809343, 12.96171273],
                        [482.34539247, 35.20258059, 3516.54167823, 66.40531433, 13.50396048],
                        [10.91893302, 17.90864387, 24.80126542, 12.48895419, 10.52435923],
                        [11.54549538, 23.98146843, 24.97476306, 36.03196210, 14.30433232]],
                resampled_dates, tickers))])

    assert_output(fn, fn_inputs, fn_correct_outputs)


@project_test
def test_compute_log_returns(fn):
    tickers = generate_random_tickers(5)
    dates = pd.DatetimeIndex(['2008-08-31', '2008-09-30', '2008-10-31', '2008-11-30'])

    fn_inputs = {
        'prices': pd.DataFrame(
            [
                    [21.05081048, 17.01384381, 10.98450376, 11.24809343, 12.96171273],
                    [482.34539247, 35.20258059, 3516.54167823, 66.40531433, 13.50396048],
                    [10.91893302, 17.90864387, 24.80126542, 12.48895419, 10.52435923],
                    [11.54549538, 23.98146843, 24.97476306, 36.03196210, 14.30433232]],
            dates, tickers)}
    fn_correct_outputs = OrderedDict([
        (
            'log_returns',
            pd.DataFrame(
                [
                    [np.nan, np.nan, np.nan, np.nan, np.nan],
                    [3.13172138, 0.72709204, 5.76874778, 1.77557845, 0.04098317],
                    [-3.78816218, -0.67583590, -4.95433863, -1.67093250, -0.24929051],
                    [0.05579709, 0.29199789, 0.00697116, 1.05956179, 0.30686995]],
                dates, tickers))])

    assert_output(fn, fn_inputs, fn_correct_outputs)


@project_test
def test_shift_returns(fn):
    tickers = generate_random_tickers(5)
    dates = pd.DatetimeIndex(['2008-08-31', '2008-09-30', '2008-10-31', '2008-11-30'])

    fn_inputs = {
        'returns': pd.DataFrame(
            [
                [np.nan, np.nan, np.nan, np.nan, np.nan],
                [3.13172138, 0.72709204, 5.76874778, 1.77557845, 0.04098317],
                [-3.78816218, -0.67583590, -4.95433863, -1.67093250, -0.24929051],
                [0.05579709, 0.29199789, 0.00697116, 1.05956179, 0.30686995]],
            dates, tickers),
        'shift_n': 1}
    fn_correct_outputs = OrderedDict([
        (
            'shifted_returns',
            pd.DataFrame(
                [
                    [np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan],
                    [3.13172138, 0.72709204, 5.76874778, 1.77557845, 0.04098317],
                    [-3.78816218, -0.67583590, -4.95433863, -1.67093250, -0.24929051]],
                dates, tickers))])

    assert_output(fn, fn_inputs, fn_correct_outputs)


@project_test
def test_get_top_n(fn):
    tickers = generate_random_tickers(5)
    dates = pd.DatetimeIndex(['2008-08-31', '2008-09-30', '2008-10-31', '2008-11-30'])

    fn_inputs = {
        'prev_returns': pd.DataFrame(
            [
                [np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan],
                [3.13172138, 0.72709204, 5.76874778, 1.77557845, 0.04098317],
                [-3.78816218, -0.67583590, -4.95433863, -1.67093250, -0.24929051]],
            dates, tickers),
        'top_n': 3}
    fn_correct_outputs = OrderedDict([
        (
            'top_stocks',
            pd.DataFrame(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [1, 0, 1, 1, 0],
                    [0, 1, 0, 1, 1]],
                dates, tickers))])

    assert_output(fn, fn_inputs, fn_correct_outputs)


@project_test
def test_portfolio_returns(fn):
    tickers = generate_random_tickers(5)
    dates = pd.DatetimeIndex(['2008-08-31', '2008-09-30', '2008-10-31', '2008-11-30'])

    fn_inputs = {
        'df_long': pd.DataFrame(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [1, 0, 1, 1, 0],
                [0, 1, 0, 1, 1]],
            dates, tickers),
        'df_short': pd.DataFrame(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 1, 0, 1, 1],
                [1, 1, 1, 0, 0]],
            dates, tickers),
        'lookahead_returns': pd.DataFrame(
            [
                [3.13172138, 0.72709204, 5.76874778, 1.77557845, 0.04098317],
                [-3.78816218, -0.67583590, -4.95433863, -1.67093250, -0.24929051],
                [0.05579709, 0.29199789, 0.00697116, 1.05956179, 0.30686995],
                [1.25459098, 6.87369275, 2.58265839, 6.92676837, 0.84632677]],
            dates, tickers),
        'n_stocks': 3}
    fn_correct_outputs = OrderedDict([
        (
            'portfolio_returns',
            pd.DataFrame(
                [
                    [0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000],
                    [-0.00000000, -0.00000000, -0.00000000, -0.00000000, -0.00000000],
                    [0.01859903, -0.09733263, 0.00232372, 0.00000000, -0.10228998],
                    [-0.41819699, 0.00000000, -0.86088613, 2.30892279, 0.28210892]],
                dates, tickers))])

    assert_output(fn, fn_inputs, fn_correct_outputs)


@project_test
def test_analyze_alpha(fn):
    dates = pd.DatetimeIndex(['2008-08-31', '2008-09-30', '2008-10-31', '2008-11-30'])

    fn_inputs = {
        'expected_portfolio_returns_by_date': pd.Series(
            [0.00000000, 0.00000000, 0.01859903, -0.41819699],
            dates)}
    fn_correct_outputs = OrderedDict([
        (
            't_value',
            -0.940764456618),
        (
            'p_value',
            0.208114098207)])

    assert_output(fn, fn_inputs, fn_correct_outputs)
