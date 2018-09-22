from collections import OrderedDict
import pandas as pd
from tests import project_test, assert_output


@project_test
def test_csv_to_close(fn):
    tickers = ['A', 'B', 'C']
    dates = ['2017-09-22', '2017-09-25', '2017-09-26', '2017-09-27', '2017-09-28']

    fn_inputs = {
        'csv_filepath': 'prices_2017_09_22_2017-09-28.csv',
        'field_names': ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume', 'adj_close', 'adj_volume']}
    fn_correct_outputs = OrderedDict([
        (
            'close',
            pd.DataFrame(
                [
                    [152.48000000, 149.19000000, 59.35000000],
                    [151.11000000, 145.06000000, 60.29000000],
                    [152.42000000, 145.21000000, 57.74000000],
                    [154.34000000, 147.02000000, 58.41000000],
                    [153.68000000, 147.19000000, 56.76000000]],
                dates, tickers))])

    assert_output(fn, fn_inputs, fn_correct_outputs)
