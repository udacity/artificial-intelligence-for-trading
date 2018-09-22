import helper
import numpy as np
from IPython.core.display import display, HTML
import plotly.graph_objs as go
import plotly.figure_factory as ff

import plotly.offline as offline_py
offline_py.init_notebook_mode(connected=True)


def _generate_hover_text(x_text, y_text, z_values, x_label, y_label, z_label):
    float_to_str = np.vectorize('{:.7f}'.format)

    x_hover_text_values = np.tile(x_text, (len(y_text), 1))
    y_hover_text_values = np.tile(y_text, (len(x_text), 1))

    padding_len = np.full(3, max(len(x_label), len(y_label), len(z_label))) - \
                  [len(x_label), len(y_label), len(z_label)]

    # Additional padding added to ticker and date to align
    hover_text = x_label + ':  ' + padding_len[0] * ' ' + x_hover_text_values + '<br>' + \
                 y_label + ':  ' + padding_len[1] * ' ' + y_hover_text_values.T + '<br>' + \
                 z_label + ': ' + padding_len[2] * ' ' + float_to_str(z_values)

    return hover_text


def _generate_heatmap_trace(df, x_label, y_label, z_label, scale_min, scale_max):
    hover_text = _generate_hover_text(df.index, df.columns, df.values.T, x_label, y_label, z_label)

    return go.Heatmap(
        x=df.index,
        y=df.columns,
        z=df.values.T,
        zauto=False,
        zmax=scale_max,
        zmin=scale_min,
        colorscale=helper.color_scheme['heatmap_colorscale'],
        text=hover_text,
        hoverinfo='text')


def _sanatize_string(string):
    return ''.join([i for i in string if i.isalpha()])


def large_dollar_volume_stocks(df, price_column, volume_column, top_percent):
    """
    Get the stocks with the largest dollar volume stocks.

    Parameters
    ----------
    df : DataFrame
        Stock prices with dates and ticker symbols
    price_column : str
        The column with the price data in `df`
    volume_column : str
        The column with the volume in `df`
    top_percent : float
        The top x percent to consider largest in the stock universe

    Returns
    -------
    large_dollar_volume_stocks_symbols : List of str
        List of of large dollar volume stock symbols
    """
    dollar_traded = df.groupby('ticker').apply(lambda row: sum(row[volume_column] * row[price_column]))

    return dollar_traded.sort_values().tail(int(len(dollar_traded) * top_percent)).index.values.tolist()


def plot_benchmark_returns(benchmark_data, etf_data, title):
    config = helper.generate_config()
    index_trace = go.Scatter(
        name='Index',
        x=benchmark_data.index,
        y=benchmark_data,
        line={'color': helper.color_scheme['index']})
    etf_trace = go.Scatter(
        name='ETF',
        x=etf_data.index,
        y=etf_data,
        line={'color': helper.color_scheme['etf']})

    layout = go.Layout(
        title=title,
        xaxis={'title': 'Date'},
        yaxis={'title': 'Cumulative Returns', 'range': [0, 3]})

    fig = go.Figure(data=[index_trace, etf_trace], layout=layout)
    offline_py.iplot(fig, config=config)


def print_dataframe(df, n_rows=10, n_columns=3):
    missing_val_str = '...'
    config = helper.generate_config()

    formatted_df = df.iloc[:n_rows, :n_columns]
    formatted_df = formatted_df.applymap('{:.3f}'.format)

    if len(df.columns) > n_columns:
        formatted_df[missing_val_str] = [missing_val_str]*len(formatted_df.index)
    if len(df.index) > n_rows:
        formatted_df.loc[missing_val_str] = [missing_val_str]*len(formatted_df.columns)

    trace = go.Table(
        type='table',
        columnwidth=[1, 3],
        header={
            'values': [''] + list(formatted_df.columns.values),
            'line': {'color': helper.color_scheme['df_line']},
            'fill': {'color': helper.color_scheme['df_header']},
            'font': {'size': 13}},
        cells={
            'values': formatted_df.reset_index().values.T,
            'line': {'color': helper.color_scheme['df_line']},
            'fill': {'color': [helper.color_scheme['df_header'], helper.color_scheme['df_value']]},
            'font': {'size': 13}})

    offline_py.iplot([trace], config=config)


def plot_weights(weights, title):
    config = helper.generate_config()
    graph_path = 'graphs/{}.html'.format(_sanatize_string(title))
    trace = _generate_heatmap_trace(weights.sort_index(axis=1, ascending=False), 'Date', 'Ticker', 'Weight', 0.0, 0.2)
    layout = go.Layout(
        title=title,
        xaxis={'title': 'Dates'},
        yaxis={'title': 'Tickers'})

    fig = go.Figure(data=[trace], layout=layout)
    offline_py.plot(fig, config=config, filename=graph_path, auto_open=False)
    display(HTML('The graph for {} is too large. You can view it <a href="{}" target="_blank">here</a>.'
                 .format(title, graph_path)))


def plot_returns(returns, title):
    config = helper.generate_config()
    graph_path = 'graphs/{}.html'.format(_sanatize_string(title))
    trace = _generate_heatmap_trace(returns.sort_index(axis=1, ascending=False), 'Date', 'Ticker', 'Weight', -0.3, 0.3)
    layout = go.Layout(
        title=title,
        xaxis={'title': 'Dates'},
        yaxis={'title': 'Tickers'})

    fig = go.Figure(data=[trace], layout=layout)
    offline_py.plot(fig, config=config, filename=graph_path, auto_open=False)
    display(HTML('The graph for {} is too large. You can view it <a href="{}" target="_blank">here</a>.'
                 .format(title, graph_path)))


def plot_covariance_returns_correlation(correlation, title):
    config = helper.generate_config()
    graph_path = 'graphs/{}.html'.format(_sanatize_string(title))
    data = []

    dendro_top = ff.create_dendrogram(correlation, orientation='bottom')
    for i in range(len(dendro_top['data'])):
        dendro_top['data'][i]['yaxis'] = 'y2'
    data.extend(dendro_top['data'])

    dendro_left = ff.create_dendrogram(correlation, orientation='right')
    for i in range(len(dendro_left['data'])):
        dendro_left['data'][i]['xaxis'] = 'x2'
    data.extend(dendro_left['data'])

    heatmap_hover_text = _generate_hover_text(
        correlation.index,
        correlation.columns,
        correlation.values,
        'Ticker 2',
        'Ticker 1',
        'Correlation')
    heatmap_trace = go.Heatmap(
        x=dendro_top['layout']['xaxis']['tickvals'],
        y=dendro_left['layout']['yaxis']['tickvals'],
        z=correlation.values,
        zauto=False,
        zmax=1.0,
        zmin=-1.0,
        text=heatmap_hover_text,
        hoverinfo='text')
    data.append(heatmap_trace)

    xaxis1_layout = {
        'showgrid': False,
        'showline': False,
        'zeroline': False,
        'showticklabels': False,
        'ticks': ""}
    xaxis2_layout = {
        'showgrid': False,
        'zeroline': False,
        'showticklabels': False}

    layout = go.Layout(
        title=title,
        showlegend=False,
        width=800,
        height=800)

    figure = go.Figure(data=data, layout=layout)
    figure['layout']['xaxis'].update({'domain': [.15, 1]})
    figure['layout']['xaxis'].update(xaxis1_layout)
    figure['layout']['yaxis'].update({'domain': [0, .85]})
    figure['layout']['yaxis'].update(xaxis1_layout)

    figure['layout']['xaxis2'].update({'domain': [0, .15]})
    figure['layout']['xaxis2'].update(xaxis2_layout)
    figure['layout']['yaxis2'].update({'domain': [.825, .975]})
    figure['layout']['yaxis2'].update(xaxis2_layout)

    offline_py.plot(figure, config=config, filename=graph_path, auto_open=False)
    display(HTML('The graph for {} is too large. You can view it <a href="{}" target="_blank">here</a>.'
                 .format(title, graph_path)))


def plot_xty(xty, title):
    config = helper.generate_config()
    trace = go.Bar(
        x=xty.index,
        y=xty.values)

    layout = go.Layout(
        title=title,
        xaxis={'title': 'Tickers'},
        yaxis={'title': 'Covariance'})

    fig = go.Figure(data=[trace], layout=layout)
    offline_py.iplot(fig, config=config)
