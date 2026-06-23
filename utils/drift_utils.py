from scipy.stats import ks_2samp

def calculate_ks_test(reference_series, current_series):
    """
    Performs Kolmogorov-Smirnov 2-sample test on two arrays/series of continuous data.
    Returns (ks_statistic, p_value).
    """
    res = ks_2samp(reference_series, current_series)
    return float(res.statistic), float(res.pvalue)
