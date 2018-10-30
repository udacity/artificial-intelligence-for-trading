from collections import OrderedDict
from tests import project_test, generate_random_tickers, generate_random_dates, assert_output

import numpy as np
import pandas as pd


@project_test
def test_calculate_simple_moving_average(fn):
    tickers = generate_random_tickers(5)
    dates = generate_random_dates(6)

    fn_inputs = {
        'rolling_window': 3,
        'close': pd.DataFrame(
            [
                [21.050810483942833, 17.013843810658827, 10.984503755486879, 11.248093428369392, 12.961712733997235],
                [15.63570258751384, 14.69054309070934, 11.353027688995159, 475.74195118202061, 11.959640427803022],
                [482.34539247360806, 35.202580592515041, 3516.5416782257166, 66.405314327318209, 13.503960481087077],
                [10.918933017418304, 17.9086438675435, 24.801265417692324, 12.488954191854916, 10.52435923388642],
                [10.675971965144655, 12.749401436636365, 11.805257579935713, 21.539039489843024, 19.99766036804861],
                [11.545495378369814, 23.981468434099405, 24.974763062186504, 36.031962102997689, 14.304332320024963]],
            dates, tickers)}
    fn_correct_outputs = OrderedDict([
        (
            'simple_moving_average',
            pd.DataFrame(
                [
                    [np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan],
                    [173.01063518,22.30232250,1179.62640322,184.46511965,12.80843788],
                    [169.63334269,22.60058918,1184.23199044,184.87873990,11.99598671],
                    [167.98009915,21.95354197,1184.38273374,33.47776934,14.67532669],
                    [11.04680012,18.21317125,20.52709535,23.35331859,14.94211731]],
                dates, tickers))])

    assert_output(fn, fn_inputs, fn_correct_outputs)
