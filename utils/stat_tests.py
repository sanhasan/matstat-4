from __future__ import annotations

import numpy as np
from scipy import stats


def _finite_sample(values, name):
    sample = np.asarray(values, dtype=float)
    sample = sample[np.isfinite(sample)]
    if sample.size < 3:
        raise ValueError(f"{name}: требуется хотя бы три конечных наблюдения")
    return sample


def jarque_bera_manual(sample, alpha=0.05):
    """Критерий Жарка—Бера, вычисленный напрямую по выборочным моментам."""
    x = _finite_sample(sample, "sample")
    n = x.size
    centered = x - x.mean()
    m2 = np.mean(centered**2)
    if m2 == 0:
        raise ValueError("Критерий не определён для постоянной выборки")

    skewness = np.mean(centered**3) / m2**1.5
    kurtosis = np.mean(centered**4) / m2**2
    statistic = n / 6 * (skewness**2 + (kurtosis - 3) ** 2 / 4)
    critical = stats.chi2.ppf(1 - alpha, df=2)
    p_value = stats.chi2.sf(statistic, df=2)
    return {
        "test": "Жарка—Бера (собственная реализация)",
        "statistic": float(statistic),
        "critical": float(critical),
        "p_value": float(p_value),
        "reject": bool(statistic > critical),
        "n": int(n),
        "skewness": float(skewness),
        "kurtosis": float(kurtosis),
    }


def ks_2samp_manual(first, second, alpha=0.05):
    """Двухвыборочный KS-критерий с асимптотическим законом Колмогорова."""
    x = np.sort(_finite_sample(first, "first"))
    y = np.sort(_finite_sample(second, "second"))
    points = np.unique(np.concatenate([x, y]))
    cdf_x = np.searchsorted(x, points, side="right") / x.size
    cdf_y = np.searchsorted(y, points, side="right") / y.size
    statistic = np.max(np.abs(cdf_x - cdf_y))

    effective_n = x.size * y.size / (x.size + y.size)
    scaled_statistic = np.sqrt(effective_n) * statistic
    critical = stats.kstwobign.ppf(1 - alpha) / np.sqrt(effective_n)
    p_value = stats.kstwobign.sf(scaled_statistic)
    return {
        "test": "Колмогорова—Смирнова (собственная реализация)",
        "statistic": float(statistic),
        "critical": float(critical),
        "p_value": float(p_value),
        "reject": bool(statistic > critical),
        "n_first": int(x.size),
        "n_second": int(y.size),
    }


def pearson_permutation_test(x, y, permutations=5000, alpha=0.05, seed=24422442):
    """Двусторонний перестановочный тест для коэффициента Пирсона."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        raise ValueError("Нужны парные непостоянные выборки длины не менее трёх")
    if permutations < 1:
        raise ValueError("Число перестановок должно быть положительным")

    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = np.linalg.norm(x_centered) * np.linalg.norm(y_centered)
    statistic = np.dot(x_centered, y_centered) / denominator

    rng = np.random.default_rng(seed)
    absolute_null = np.empty(permutations)
    for index in range(permutations):
        absolute_null[index] = abs(
            np.dot(x_centered, rng.permutation(y_centered)) / denominator
        )

    absolute_statistic = abs(statistic)
    p_value = (1 + np.count_nonzero(absolute_null >= absolute_statistic)) / (
        permutations + 1
    )
    critical = np.quantile(absolute_null, 1 - alpha, method="higher")
    return {
        "test": "Пирсон, перестановочный (собственная реализация)",
        "statistic": float(statistic),
        "critical": float(critical),
        "p_value": float(p_value),
        "reject": bool(absolute_statistic > critical),
        "n": int(x.size),
        "permutations": int(permutations),
    }

