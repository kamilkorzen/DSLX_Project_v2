#!/usr/bin/env python3
"""
Sample Quality & Cohort Drift Module for DSLX_Project.

Performs:
1. Row-Level Completeness Check: Flags subjects with >20% missing subtests.
2. Outlier Detection: Flags raw subtest scores exceeding +/- 3 SD from the historical baseline.
3. Feature Drift Report: Performs two-sample Kolmogorov-Smirnov (KS) tests comparing the input cohort
   against the baseline training distribution.
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, Tuple


def check_sample_quality(
    input_df: pd.DataFrame,
    feature_metadata_path: str = 'artifacts/feature_metadata.json',
    missing_threshold: float = 0.20,
    outlier_sd: float = 3.0,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Runs comprehensive data quality checks on the input cohort.
    """
    if not os.path.exists(feature_metadata_path):
        alt_path = os.path.join(os.path.dirname(__file__), 'artifacts', 'feature_metadata.json')
        if os.path.exists(alt_path):
            feature_metadata_path = alt_path
        else:
            raise FileNotFoundError(f'Feature metadata not found at {feature_metadata_path}')

    with open(feature_metadata_path, 'r') as f:
        meta = json.load(f)

    feature_names = meta['feature_names']
    means = meta['means']
    scales = meta['scales']

    df_feats = input_df[feature_names].copy()
    n_samples, n_features = df_feats.shape

    # 1. Row-level completeness check
    missing_counts = df_feats.isna().sum(axis=1)
    missing_pcts = missing_counts / n_features
    high_missing_mask = missing_pcts > missing_threshold
    high_missing_indices = df_feats.index[high_missing_mask].tolist()

    # 2. Outlier detection (+/- outlier_sd)
    outliers_report = {}
    total_outliers = 0
    for col in feature_names:
        m = means[col]
        s = scales[col]
        lower_bound = m - outlier_sd * s
        upper_bound = m + outlier_sd * s
        col_vals = df_feats[col].dropna()
        col_outliers = col_vals[(col_vals < lower_bound) | (col_vals > upper_bound)]
        if len(col_outliers) > 0:
            outliers_report[col] = {
                'count': int(len(col_outliers)),
                'lower_bound': round(float(lower_bound), 2),
                'upper_bound': round(float(upper_bound), 2),
                'outlier_values': [round(float(v), 2) for v in col_outliers.tolist()[:10]],
                'sample_indices': col_outliers.index.tolist()[:10]
            }
            total_outliers += len(col_outliers)

    # 3. Two-sample Kolmogorov-Smirnov test for distribution drift
    # Simulates baseline distribution from moments or checks if baseline sample is available
    drift_report = {}
    drift_detected_cols = []
    rng = np.random.RandomState(42)

    for col in feature_names:
        obs_vals = df_feats[col].dropna().values
        if len(obs_vals) >= 5:
            # Generate reference Gaussian distribution matching training baseline moments
            ref_vals = rng.normal(loc=means[col], scale=scales[col], size=max(len(obs_vals), 500))
            ks_stat, p_val = stats.ks_2samp(obs_vals, ref_vals)
            has_drift = bool(p_val < alpha)
            if has_drift:
                drift_detected_cols.append(col)
            drift_report[col] = {
                'ks_stat': round(float(ks_stat), 4),
                'p_value': round(float(p_val), 4),
                'drift_detected': has_drift
            }

    summary = {
        'total_subjects': n_samples,
        'high_missingness_subjects_count': len(high_missing_indices),
        'high_missingness_indices': high_missing_indices,
        'total_outlier_values': total_outliers,
        'features_with_outliers': list(outliers_report.keys()),
        'outlier_details': outliers_report,
        'features_with_drift': drift_detected_cols,
        'drift_details': drift_report
    }

    return summary


def print_quality_report(summary: Dict[str, Any]):
    """Prints a structured summary of quality audit findings to stdout."""
    print('\n' + '='*60)
    print('          DSLX COHORT QUALITY & DRIFT AUDIT REPORT')
    print('='*60)

    print(f"\n1. Completeness Check (Threshold: >20% missing subtests):")
    n_high = summary['high_missingness_subjects_count']
    if n_high > 0:
        print(f"   [!] WARNING: {n_high} subjects have >20% missing features.")
        print(f"   Row indices: {summary['high_missingness_indices']}")
    else:
        print(f"   [OK] All {summary['total_subjects']} subjects meet completeness requirements.")

    print(f"\n2. Outlier Detection (+/- 3 SD from training baseline):")
    if summary['total_outlier_values'] > 0:
        print(f"   [!] Found {summary['total_outlier_values']} outlier value(s) across {len(summary['features_with_outliers'])} feature(s):")
        for feat in summary['features_with_outliers']:
            d = summary['outlier_details'][feat]
            print(f"       - {feat}: {d['count']} outlier(s) (expected bounds: [{d['lower_bound']}, {d['upper_bound']}]). Values: {d['outlier_values']}")
    else:
        print(f"   [OK] Zero extreme outliers (+/- 3 SD) detected.")

    print(f"\n3. Distribution Drift (Kolmogorov-Smirnov Test vs Baseline):")
    drift_feats = summary['features_with_drift']
    if drift_feats:
        print(f"   [!] Potential distribution drift detected in {len(drift_feats)} feature(s) (p < 0.05):")
        for feat in drift_feats:
            d = summary['drift_details'][feat]
            print(f"       - {feat:30s} KS-stat={d['ks_stat']:.3f}, p={d['p_value']:.4f}")
    else:
        print(f"   [OK] No significant distribution drift detected across tested features.")

    print('='*60 + '\n')


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python dslx_quality.py <input_csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    df = pd.read_csv(csv_path)
    res = check_sample_quality(df)
    print_quality_report(res)
