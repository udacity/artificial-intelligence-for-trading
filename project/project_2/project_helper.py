import helper
import scipy.stats
from colour import Color
import numpy as np
import pandas as pd
import plotly as py
import plotly.graph_objs as go
import plotly.offline as offline_py
offline_py.init_notebook_mode(connected=True)


def _generate_stock_trace(prices):
    return go.Scatter(
        name='Index',
        x=prices.index,
        y=prices,
        line={'color': helper.color_scheme['main_line']})


def _generate_buy_annotations(prices, signal):
    return [{
        'x': index, 'y': price, 'text': 'Long', 'bgcolor': helper.color_scheme['background_label'],
        'ayref': 'y', 'ax': 0, 'ay': 20}
        for index, price in prices[signal == 1].iteritems()]


def _generate_sell_annotations(prices, signal):
    return [{
        'x': index, 'y': price, 'text': 'Short', 'bgcolor': helper.color_scheme['background_label'],
        'ayref': 'y', 'ax': 0, 'ay': 160}
        for index, price in prices[signal == -1].iteritems()]


def _generate_second_tetration_stock(stock_symbol, dates):
    """
    Generate stock that follows the second tetration curve
    :param stock_symbol: Stock Symbol
    :param dates: Dates for ticker
    :return: Stock data
    """
    n_stock_columns = 5
    linear_line = np.linspace(1, 5, len(dates))
    all_noise = ((np.random.rand(n_stock_columns, len(dates)) - 0.5) * 0.01)
    sector_stock = pd.DataFrame({
        'ticker': stock_symbol,
        'date': dates,
        'base_line': np.power(linear_line, linear_line)})

    sector_stock['base_line'] = sector_stock['base_line'] + all_noise[0]*sector_stock['base_line']
    sector_stock['adj_open'] = sector_stock['base_line'] + all_noise[1]*sector_stock['base_line']
    sector_stock['adj_close'] = sector_stock['base_line'] + all_noise[2]*sector_stock['base_line']
    sector_stock['adj_high'] = sector_stock['base_line'] + all_noise[3]*sector_stock['base_line']
    sector_stock['adj_low'] = sector_stock['base_line'] + all_noise[4]*sector_stock['base_line']

    sector_stock['adj_high'] = sector_stock[['adj_high', 'adj_open', 'adj_close']].max(axis=1)
    sector_stock['adj_low'] = sector_stock[['adj_low', 'adj_open', 'adj_close']].min(axis=1)

    return sector_stock.drop(columns='base_line')


def generate_tb_sector(dates):
    """
    Generate TB sector of stocks
    :param dates: Dates that stocks should have market data on
    :return: TB sector stocks
    """
    symbol_length = 6
    stock_names = [
        'kaufmanniana', 'clusiana', 'greigii', 'sylvestris', 'turkestanica', 'linifolia', 'gesneriana',
        'humilis', 'tarda', 'saxatilis', 'dasystemon', 'orphanidea', 'kolpakowskiana', 'praestans',
        'sprengeri', 'bakeri', 'pulchella', 'biflora', 'schrenkii', 'armena', 'vvedenskyi', 'agenensis',
        'altaica', 'urumiensis']

    return [
        _generate_second_tetration_stock(stock_name[:symbol_length].upper(), dates)
        for stock_name in stock_names]


def plot_stock(prices, title):
    config = helper.generate_config()
    layout = go.Layout(title=title)

    stock_trace = _generate_stock_trace(prices)

    offline_py.iplot({'data': [stock_trace], 'layout': layout}, config=config)


def plot_high_low(prices, lookback_high, lookback_low, title):
    config = helper.generate_config()
    layout = go.Layout(title=title)

    stock_trace = _generate_stock_trace(prices)
    high_trace = go.Scatter(
        x=lookback_high.index,
        y=lookback_high,
        name='Column lookback_high',
        line={'color': helper.color_scheme['major_line']})
    low_trace = go.Scatter(
        x=lookback_low.index,
        y=lookback_low,
        name='Column lookback_low',
        line={'color': helper.color_scheme['minor_line']})

    offline_py.iplot({'data': [stock_trace, high_trace, low_trace], 'layout': layout}, config=config)


def plot_signal(price, signal, title):
    config = helper.generate_config()
    buy_annotations = _generate_buy_annotations(price, signal)
    sell_annotations = _generate_sell_annotations(price, signal)
    layout = go.Layout(
        title=title,
        annotations=buy_annotations + sell_annotations)

    stock_trace = _generate_stock_trace(price)

    offline_py.iplot({'data': [stock_trace], 'layout': layout}, config=config)


def plot_lookahead_prices(prices, lookahead_price_list, title):
    config = helper.generate_config()
    layout = go.Layout(title=title)
    colors = Color(helper.color_scheme['low_value'])\
        .range_to(Color(helper.color_scheme['high_value']), len(lookahead_price_list))

    traces = [_generate_stock_trace(prices)]
    for (lookahead_prices, lookahead_days), color in zip(lookahead_price_list, colors):
        traces.append(
            go.Scatter(
                x=lookahead_prices.index,
                y=lookahead_prices,
                name='{} Day Lookahead'.format(lookahead_days),
                line={'color': str(color)}))

    offline_py.iplot({'data': traces, 'layout': layout}, config=config)


def plot_price_returns(prices, lookahead_returns_list, title):
    config = helper.generate_config()
    layout = go.Layout(
        title=title,
        yaxis2={
            'title': 'Returns',
            'titlefont': {'color': helper.color_scheme['y_axis_2_text_color']},
            'tickfont': {'color': helper.color_scheme['y_axis_2_text_color']},
            'overlaying': 'y',
            'side': 'right'})
    colors = Color(helper.color_scheme['low_value'])\
        .range_to(Color(helper.color_scheme['high_value']), len(lookahead_returns_list))

    traces = [_generate_stock_trace(prices)]
    for (lookahead_returns, lookahead_days), color in zip(lookahead_returns_list, colors):
        traces.append(
            go.Scatter(
                x=lookahead_returns.index,
                y=lookahead_returns,
                name='{} Day Lookahead'.format(lookahead_days),
                line={'color': str(color)},
                yaxis='y2'))

    offline_py.iplot({'data': traces, 'layout': layout}, config=config)


def plot_signal_returns(prices, signal_return_list, titles):
    config = helper.generate_config()
    layout = go.Layout(
        yaxis2={
            'title': 'Signal Returns',
            'titlefont': {'color': helper.color_scheme['y_axis_2_text_color']},
            'tickfont': {'color': helper.color_scheme['y_axis_2_text_color']},
            'overlaying': 'y',
            'side': 'right'})
    colors = Color(helper.color_scheme['low_value'])\
        .range_to(Color(helper.color_scheme['high_value']), len(signal_return_list))

    stock_trace = _generate_stock_trace(prices)
    for (signal_return, signal, lookahead_days), color, title in zip(signal_return_list, colors, titles):
        non_zero_signals = signal_return[signal_return != 0]
        signal_return_trace = go.Scatter(
                x=non_zero_signals.index,
                y=non_zero_signals,
                name='{} Day Lookahead'.format(lookahead_days),
                line={'color': str(color)},
                yaxis='y2')

        buy_annotations = _generate_buy_annotations(prices, signal)
        sell_annotations = _generate_sell_annotations(prices, signal)
        layout['title'] = title
        layout['annotations'] = buy_annotations + sell_annotations

        offline_py.iplot({'data': [stock_trace, signal_return_trace], 'layout': layout}, config=config)


def plot_signal_histograms(signal_list, title, subplot_titles):
    assert len(signal_list) == len(subplot_titles)

    signal_series_list = [signal.stack() for signal in signal_list]
    all_values = pd.concat(signal_series_list)
    x_range = [all_values.min(), all_values.max()]
    y_range = [0, 1500]
    config = helper.generate_config()
    colors = Color(helper.color_scheme['low_value']).range_to(Color(helper.color_scheme['high_value']), len(signal_series_list))

    fig = py.tools.make_subplots(rows=1, cols=len(signal_series_list), subplot_titles=subplot_titles, print_grid=False)
    fig['layout'].update(title=title, showlegend=False)

    for series_i, (signal_series, color) in enumerate(zip(signal_series_list, colors), 1):
        filtered_series = signal_series[signal_series != 0].dropna()
        trace = go.Histogram(x=filtered_series, marker={'color': str(color)})
        fig.append_trace(trace, 1, series_i)
        fig['layout']['xaxis{}'.format(series_i)].update(range=x_range)
        fig['layout']['yaxis{}'.format(series_i)].update(range=y_range)

    offline_py.iplot(fig, config=config)


def plot_signal_to_normal_histograms(signal_list, title, subplot_titles):
    assert len(signal_list) == len(subplot_titles)

    signal_series_list = [signal.stack() for signal in signal_list]
    all_values = pd.concat(signal_series_list)
    x_range = [all_values.min(), all_values.max()]
    y_range = [0, 1500]
    config = helper.generate_config()

    fig = py.tools.make_subplots(rows=1, cols=len(signal_series_list), subplot_titles=subplot_titles, print_grid=False)
    fig['layout'].update(title=title)

    for series_i, signal_series in enumerate(signal_series_list, 1):
        filtered_series = signal_series[signal_series != 0].dropna()
        filtered_series_trace = go.Histogram(
            x=filtered_series,
            marker={'color': helper.color_scheme['low_value']},
            name='Signal Return Distribution',
            showlegend=False)
        normal_trace = go.Histogram(
            x=np.random.normal(np.mean(filtered_series), np.std(filtered_series), len(filtered_series)),
            marker={'color': helper.color_scheme['shadow']},
            name='Normal Distribution',
            showlegend=False)
        fig.append_trace(filtered_series_trace, 1, series_i)
        fig.append_trace(normal_trace, 1, series_i)
        fig['layout']['xaxis{}'.format(series_i)].update(range=x_range)
        fig['layout']['yaxis{}'.format(series_i)].update(range=y_range)

    # Show legened
    fig['data'][0]['showlegend'] = True
    fig['data'][1]['showlegend'] = True

    offline_py.iplot(fig, config=config)
