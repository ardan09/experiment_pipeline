import pandas as pd
import numpy as np
import abc

from scipy.stats import ttest_ind_from_stats, ttest_ind, mannwhitneyu
from statsmodels.stats.proportion import proportions_ztest

import config as cfg


class EstimatorCriteriaValues:
    def __init__(self, pvalue: float, statistic: float):
        self.pvalue = pvalue
        self.statistic = statistic


class Statistics:
    def __init__(self, mean_0: float, mean_1: float, var_0: float, var_1: float, n_0: int, n_1: int, numerator_metric_0: np.ndarray, numerator_metric_1: np.ndarray):
        self.mean_0 = mean_0
        self.mean_1 = mean_1
        self.var_0 = var_0
        self.var_1 = var_1
        self.n_0 = n_0
        self.n_1 = n_1
        self.numerator_metric_0 = numerator_metric_0
        self.numerator_metric_1 = numerator_metric_1


class MetricStats(abc.ABC):
    @abc.abstractmethod
    def __call__(self, df) -> Statistics:
        pass


class Estimator(abc.ABC):
    @abc.abstractmethod
    def __call__(self, Statistics) -> EstimatorCriteriaValues:
        pass


class BaseStatsRatio(MetricStats):
    def __call__(self, df) -> Statistics:

        _unique_variants = df[cfg.VARIANT_COL].unique()
        n_0 = sum(df['n'][df[cfg.VARIANT_COL] == _unique_variants[0]])
        n_1 = sum(df['n'][df[cfg.VARIANT_COL] == _unique_variants[1]])
        mean_0 = sum(df['num'][df[cfg.VARIANT_COL] == _unique_variants[0]]) / sum(df['den'][df[cfg.VARIANT_COL] == _unique_variants[0]])
        mean_1 = sum(df['num'][df[cfg.VARIANT_COL] == _unique_variants[1]]) / sum(df['den'][df[cfg.VARIANT_COL] == _unique_variants[1]])

        var_0 = df['l_ratio'][df[cfg.VARIANT_COL] == _unique_variants[0]].var()
        var_1 = df['l_ratio'][df[cfg.VARIANT_COL] == _unique_variants[1]].var()

        # для манна уитни просто выборки (числитель)
        numerator_metric_0 = df['num'][df[cfg.VARIANT_COL] == _unique_variants[0]]
        numerator_metric_1 = df['num'][df[cfg.VARIANT_COL] == _unique_variants[1]]

        return Statistics(mean_0, mean_1, var_0, var_1, n_0, n_1, numerator_metric_0, numerator_metric_1)


class Linearization():

    def __call__(self, num_0, den_0, num_1, den_1):
        k = np.sum(num_0) / np.sum(den_0)
        l_0 = num_0 - k * den_0
        l_1 = num_1 - k * den_1
        return l_0, l_1


class TTestFromStats(Estimator):

    def __call__(self, stat: Statistics) -> EstimatorCriteriaValues:
        try:
            statistic, pvalue = ttest_ind_from_stats(
                mean1=stat.mean_0,
                std1=np.sqrt(stat.var_0),
                nobs1=stat.n_0,
                mean2=stat.mean_1,
                std2=np.sqrt(stat.var_1),
                nobs2=stat.n_1
            )
        except Exception as e:
            cfg.logger.error(e)
            statistic, pvalue = None, None

        return EstimatorCriteriaValues(pvalue, statistic)

class MWTestFromStats(Estimator): ## ARDAN TASK SOLUTION Mann-Whitney test for ratio ?

    def __call__(self, stat: Statistics) -> EstimatorCriteriaValues:
        try:
            statistic, pvalue = mannwhitneyu(stat.numerator_metric_0, stat.numerator_metric_1)
        except Exception as e:
            cfg.logger.error(e)
            statistic, pvalue = None, None

        return EstimatorCriteriaValues(pvalue, statistic)

class PropTestFromStats(Estimator): ## ARDAN TASK SOLUTION 2 sample proportion test for ratio ?

    def __call__(self, stat: Statistics) -> EstimatorCriteriaValues:
        try:
            # Число успехов в каждой группе
            count = [sum(stat.numerator_metric_0), sum(stat.numerator_metric_1)]

            # Общее количество наблюдений в каждой группе
            nobs = [stat.n_0, stat.n_0]

            statistic, pvalue = proportions_ztest(count, nobs)
        except Exception as e:
            cfg.logger.error(e)
            statistic, pvalue = None, None

        return EstimatorCriteriaValues(pvalue, statistic)

def calculate_statistics(df, type):
    mappings = {
        "ratio": BaseStatsRatio()
        # TODO расчет статистик не для ratio
    }

    calculate_method = mappings[type]

    return calculate_method(df)


def calculate_linearization(df):
    _variants = df[cfg.VARIANT_COL].unique()
    linearization = Linearization()

    df['l_ratio'] = 0
    if (df['den'] == df['n']).all():
        df.loc[df[cfg.VARIANT_COL] == _variants[0], 'l_ratio'] = df.loc[df[cfg.VARIANT_COL] == _variants[0], 'num']
        df.loc[df[cfg.VARIANT_COL] == _variants[1], 'l_ratio'] = df.loc[df[cfg.VARIANT_COL] == _variants[1], 'num']
    else:
        l_0, l_1 = linearization(
            df['num'][df[cfg.VARIANT_COL] == _variants[0]],
            df['den'][df[cfg.VARIANT_COL] == _variants[0]],
            df['num'][df[cfg.VARIANT_COL] == _variants[1]],
            df['den'][df[cfg.VARIANT_COL] == _variants[1]]
        )
        df.loc[df[cfg.VARIANT_COL] == _variants[0], 'l_ratio'] = l_0
        df.loc[df[cfg.VARIANT_COL] == _variants[1], 'l_ratio'] = l_1

    return df

