from collections import OrderedDict
import numpy as np
import pandas as pd

from tests import generate_random_tickers, generate_random_dates, assert_output, project_test


@project_test
def test_get_high_lows_lookback(fn):
    tickers = generate_random_tickers(3)
    dates = generate_random_dates(4)

    fn_inputs = {
        'high': pd.DataFrame(
            [
                [35.4411, 34.1799, 34.0223],
                [92.1131, 91.0543, 90.9572],
                [57.9708, 57.7814, 58.1982],
                [34.1705, 92.453, 58.5107]],
            dates, tickers),
        'low': pd.DataFrame(
            [
                [15.6718, 75.1392, 34.0527],
                [27.1834, 12.3453, 95.9373],
                [28.2503, 24.2854, 23.2932],
                [86.3725, 32.223, 38.4107]],
            dates, tickers),
        'lookback_days': 2}
    fn_correct_outputs = OrderedDict([
        (
            'lookback_high',
            pd.DataFrame(
                [
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [92.11310000, 91.05430000, 90.95720000],
                    [92.11310000, 91.05430000, 90.95720000]],
                dates, tickers)),
        (
            'lookback_low',
            pd.DataFrame(
                [
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [15.67180000, 12.34530000, 34.05270000],
                    [27.18340000, 12.34530000, 23.29320000]],
                dates, tickers))
    ])

    assert_output(fn, fn_inputs, fn_correct_outputs)


@project_test
def test_get_long_short(fn):
    tickers = generate_random_tickers(3)
    dates = generate_random_dates(4)

    fn_inputs = {
        'close': pd.DataFrame(
            [
                [25.6788, 35.1392, 34.0527],
                [25.1884, 14.3453, 39.9373],
                [78.2803, 34.3854, 23.2932],
                [88.8725, 52.223, 34.4107]],
            dates, tickers),
        'lookback_high': pd.DataFrame(
            [
                [np.nan, np.nan, np.nan],
                [92.11310000, 91.05430000, 90.95720000],
                [35.4411, 34.1799, 34.0223],
                [92.11310000, 91.05430000, 90.95720000]],
            dates, tickers),
        'lookback_low': pd.DataFrame(
            [
                [np.nan, np.nan, np.nan],
                [34.1705, 92.453, 58.5107],
                [15.67180000, 12.34530000, 34.05270000],
                [27.18340000, 12.34530000, 23.29320000]],
            dates, tickers)}
    fn_correct_outputs = OrderedDict([
        (
            'long_short',
            pd.DataFrame(
                [
                    [0, 0, 0],
                    [-1, -1, -1],
                    [1, 1, -1],
                    [0, 0, 0]],
                dates, tickers))])

    assert_output(fn, fn_inputs, fn_correct_outputs)


@project_test
def test_filter_signals(fn):
    tickers = generate_random_tickers(3)
    dates = generate_random_dates(10)

    fn_inputs = {
        'signal': pd.DataFrame(
            [
                [0, 0, 0],
                [-1, -1, -1],
                [1, 0, -1],
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [0, -1, 1],
                [-1, 0, 0],
                [0, 0, 0]],
            dates, tickers),
        'lookahead_days': 3}
    fn_correct_outputs = OrderedDict([
        (
            'filtered_signal',
            pd.DataFrame(
                [
                    [0, 0, 0],
                    [-1, -1, -1],
                    [1, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [0, -1, 0],
                    [-1, 0, 0],
                    [0, 0, 0]],
                dates, tickers))])

    assert_output(fn, fn_inputs, fn_correct_outputs)


@project_test
def test_get_lookahead_prices(fn):
    tickers = generate_random_tickers(3)
    dates = generate_random_dates(5)

    fn_inputs = {
        'close': pd.DataFrame(
            [
                [25.6788, 35.1392, 34.0527],
                [25.1884, 14.3453, 39.9373],
                [62.3457, 92.2524, 65.7893],
                [78.2803, 34.3854, 23.2932],
                [88.8725, 52.223, 34.4107]],
            dates, tickers),
        'lookahead_days': 2}
    fn_correct_outputs = OrderedDict([
        (
            'lookahead_prices',
            pd.DataFrame(
                [
                    [62.34570000, 92.25240000, 65.78930000],
                    [78.28030000, 34.38540000, 23.29320000],
                    [88.87250000, 52.22300000, 34.41070000],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan]],
                dates, tickers))])

    assert_output(fn, fn_inputs, fn_correct_outputs)


@project_test
def test_get_return_lookahead(fn):
    tickers = generate_random_tickers(3)
    dates = generate_random_dates(5)

    fn_inputs = {
        'close': pd.DataFrame(
            [
                [25.6788, 35.1392, 34.0527],
                [25.1884, 14.3453, 39.9373],
                [62.3457, 92.2524, 65.7893],
                [78.2803, 34.3854, 23.2932],
                [88.8725, 52.223, 34.4107]],
            dates, tickers),
        'lookahead_prices': pd.DataFrame(
            [
                [62.34570000, 92.25240000, 65.78930000],
                [78.28030000, 34.38540000, 23.29320000],
                [88.87250000, 52.22300000, 34.41070000],
                [np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan]],
            dates, tickers)}
    fn_correct_outputs = OrderedDict([
        (
            'lookahead_returns',
            pd.DataFrame(
                [
                    [0.88702896,  0.96521098,  0.65854789],
                    [1.13391240,  0.87420969, -0.53914925],
                    [0.35450805, -0.56900529, -0.64808965],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan]],
                dates, tickers))])

    assert_output(fn, fn_inputs, fn_correct_outputs)


@project_test
def test_get_signal_return(fn):
    tickers = generate_random_tickers(3)
    dates = generate_random_dates(5)

    fn_inputs = {
        'signal': pd.DataFrame(
            [
                [0, 0, 0],
                [-1, -1, -1],
                [1, 0, 0],
                [0, 0, 0],
                [0, 1, 0]],
            dates, tickers),
        'lookahead_returns': pd.DataFrame(
            [
                [0.88702896, 0.96521098, 0.65854789],
                [1.13391240, 0.87420969, -0.53914925],
                [0.35450805, -0.56900529, -0.64808965],
                [0.38572896, -0.94655617, 0.123564379],
                [np.nan, np.nan, np.nan]],
            dates, tickers)}
    fn_correct_outputs = OrderedDict([
        (
            'signal_return',
            pd.DataFrame(
                [
                    [0, 0, 0],
                    [-1.13391240, -0.87420969,  0.53914925],
                    [0.35450805, 0, 0],
                    [0, 0, 0],
                    [np.nan, np.nan, np.nan]],
                dates, tickers))])

    assert_output(fn, fn_inputs, fn_correct_outputs)


@project_test
def test_calculate_kstest(fn):
    tickers = generate_random_tickers(3)

    fn_inputs = {
        'long_short_signal_returns': pd.DataFrame(
            {
                'ticker': tickers * 5,
                'signal_return': [0.12, -0.83, 0.37, 0.83, -0.34, 0.27, -0.68, 0.29, 0.69,
                                  0.57, 0.39, 0.56, -0.97, -0.72, 0.26]})}
    fn_correct_outputs = OrderedDict([
        (
            'ks_values',
            pd.Series(
                [0.29787827, 0.35221525, 0.63919407],
                tickers)),
        (
            'p_values',
            pd.Series(
                [0.69536353, 0.46493498, 0.01650327],
                tickers))])

    assert_output(fn, fn_inputs, fn_correct_outputs)


@project_test
def test_find_outliers(fn):
    tickers = generate_random_tickers(3)

    fn_inputs = {
        'ks_values': pd.Series(
            [0.20326939, 0.34826827, 0.60256811],
            tickers),
        'p_values': pd.Series(
            [0.98593727, 0.48009144, 0.02898631],
            tickers),
        'ks_threshold': 0.5,
        'pvalue_threshold': 0.05}
    fn_correct_outputs = OrderedDict([
        (
            'outliers',
            set([tickers[2]]))])

    assert_output(fn, fn_inputs, fn_correct_outputs)
