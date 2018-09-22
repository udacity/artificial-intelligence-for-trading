from collections import OrderedDict
import pandas as pd
from tests import project_test, generate_random_tickers, generate_random_dates, assert_output


@project_test
def test_date_top_industries(fn):
    tickers = generate_random_tickers(10)
    dates = generate_random_dates(2)

    fn_inputs = {
        'prices': pd.DataFrame(
            [
                [21.050810483942833, 17.013843810658827, 10.984503755486879, 11.248093428369392, 12.961712733997235,
                 482.34539247360806, 35.202580592515041, 3516.5416782257166, 66.405314327318209, 13.503960481087077],
                [15.63570258751384, 14.69054309070934, 11.353027688995159, 475.74195118202061, 11.959640427803022,
                 10.918933017418304, 17.9086438675435, 24.801265417692324, 12.488954191854916, 15.63570258751384]],
            dates, tickers),
        'sector': pd.Series(
            ['ENERGY', 'MATERIALS', 'ENERGY', 'ENERGY', 'TELECOM', 'FINANCIALS',
             'TECHNOLOGY', 'HEALTH', 'MATERIALS', 'REAL ESTATE'],
            tickers),
        'date': dates[-1],
        'top_n': 4}
    fn_correct_outputs = OrderedDict([
        (
            'top_industries',
            {'ENERGY', 'HEALTH', 'TECHNOLOGY'})])

    assert_output(fn, fn_inputs, fn_correct_outputs)


@project_test
def test_generate_positions(fn):
    tickers = generate_random_tickers(5)
    dates = generate_random_dates(6)

    fn_inputs = {
        'prices': pd.DataFrame(
            [
                [65.40757705426432, 27.556319958924323, 50.59935209411175, 56.274712269629134, 99.9873070881051],
                [47.82126720752715, 56.812865745668375, 40.75685814634723, 27.469680989736023, 41.449858088448735],
                [88.20038097315815, 45.573972499280494, 36.592711369868724, 21.36570423559795, 0.698919959739297],
                [14.670236824202721, 49.557949251949054, 18.935364730808935, 23.163368660093298, 8.075599541367884],
                [41.499140208637705, 9.75987296846733, 66.08677766062186, 37.927861417544385, 10.792730405945827],
                [86.26923464863536, 32.12679487375028, 15.621592524570282, 77.1908860965619, 52.733950486350444]],
            dates, tickers)}
    fn_correct_outputs = OrderedDict([
        (
            'final_positions',
            pd.DataFrame(
                [
                    [30, 0, 30, 30, 30],
                    [0, 30, 0, 0, 0],
                    [30, 0, 0, 0, -10],
                    [-10, 0, -10, 0, -10],
                    [0, -10, 30, 0, -10],
                    [30, 0, -10, 30, 30]],
                dates, tickers))])

    assert_output(fn, fn_inputs, fn_correct_outputs)
