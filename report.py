import pandas as pd
import numpy as np
import abc
import utils
import config as cfg
from itertools import product
from metric_builder import Metric, CalculateMetric
from stattests import TTestFromStats, MWTestFromStats, PropTestFromStats, calculate_statistics, calculate_linearization


class Report:
    def __init__(self, report):
        self.report = report


class BuildMetricReport:
    def __call__(self, calculated_metric, metric_items) -> Report:
        ## ARDAN SOLUTION
        if metric_items.estimator in ['t_test', 't_test_linearization']:
            test_criteria = TTestFromStats() # ttest
        elif metric_items.estimator == 'mann_whitney':
            test_criteria = MWTestFromStats() # mw test
        elif metric_items.estimator == 'prop_test':
            test_criteria = PropTestFromStats() # proportion test
        else:
            raise ValueError("stat criteria is not defined. Choose ttest, mann-whitney or proportion test")

        cfg.logger.info(f"{metric_items.name}")

        df_ = calculate_linearization(calculated_metric)
        stats = calculate_statistics(df_, metric_items.type)
        criteria_res = test_criteria(stats)

        report_items = pd.DataFrame({
            "metric_name": metric_items.name,
            "mean_0": stats.mean_0,
            "mean_1": stats.mean_1,
            "var_0": stats.var_0,
            "var_1": stats.var_1,
            "delta": stats.mean_1 - stats.mean_0,
            "lift":  (stats.mean_1 - stats.mean_0) / stats.mean_0,
            "criteria": metric_items.estimator, ## ARDAN SOLUTION
            "pvalue": criteria_res.pvalue,
            "statistic": criteria_res.statistic
        }, index=[0])

        return Report(report_items)


def build_experiment_report(df, metric_config):
    build_metric_report = BuildMetricReport()
    reports = []

    for metric_params in metric_config:
        metric_parsed = Metric(metric_params)
        calculated_metric = CalculateMetric(metric_parsed)(df)
        metric_report = build_metric_report(calculated_metric, metric_parsed)
        reports.append(metric_report.report)

    return pd.concat(reports)

