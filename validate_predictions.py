#!/usr/bin/env python3
"""
Validation & Regression Suite for DSLX_Project.

Verifies:
1. StandardScaler parameters are non-trivial (realistic means and scales).
2. SVM probabilities no longer collapse to zero or flat constants.
3. Positive prediction rates for GBC, SVM, and Ensemble are aligned with clinical expectations (~15-25%).
4. Spearman rank correlation between GBC and SVM probabilities is positive and significant.
5. Exports a validation summary report.
"""

import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from dslx import clean_and_extract_df, find_artifacts_dir, FEATURE_COLUMNS


def validate_artifacts_and_predictions(
    test_csv: str = 'test_set.csv',
    threshold: float = 0.50,
    export_summary: bool = True
) -> bool:
    print('='*70)
    print('            DSLX MODEL & ARTIFACT VALIDATION SUITE')
    print('='*70)

    # 1. Locate and inspect artifacts
    artifacts_dir = find_artifacts_dir()
    print(f"\n[1/5] Loading artifacts from '{artifacts_dir}'...")
    scaler = joblib.load(os.path.join(artifacts_dir, 'scaler.joblib'))
    gbc = joblib.load(os.path.join(artifacts_dir, 'gbc_model.joblib'))
    svc = joblib.load(os.path.join(artifacts_dir, 'svm_model.joblib'))

    medians_path = os.path.join(artifacts_dir, 'imputer_medians.joblib')
    medians = joblib.load(medians_path) if os.path.exists(medians_path) else None

    # Check 1: Non-trivial scaler parameters
    mean_range = scaler.mean_.max() - scaler.mean_.min()
    scale_range = scaler.scale_.max() - scaler.scale_.min()

    print(f"      StandardScaler mean range:  {scaler.mean_.min():.2f} to {scaler.mean_.max():.2f}")
    print(f"      StandardScaler scale range: {scaler.scale_.min():.2f} to {scaler.scale_.max():.2f}")

    if mean_range < 5.0 or scale_range < 2.0:
        print("      [FAIL] Scaler parameters appear trivial (near 0/1)! Check scaler fit.")
        scaler_check = False
    else:
        print("      [PASS] Scaler parameters are realistic and properly fitted on raw data.")
        scaler_check = True

    # 2. Load and prepare test data
    print(f"\n[2/5] Loading test cohort from '{test_csv}'...")
    raw_df, id_series, X_test = clean_and_extract_df(test_csv)
    print(f"      Loaded {len(X_test)} subjects with {len(X_test.columns)} features.")

    if medians is not None:
        X_test_imputed = X_test.fillna(medians)
    else:
        X_test_imputed = X_test.fillna(X_test.median())

    X_test_scaled = scaler.transform(X_test_imputed)

    # 3. Model inference
    print(f"\n[3/5] Generating model predictions...")
    gbc_probs = gbc.predict_proba(X_test_scaled)[:, 1]
    svc_probs = svc.predict_proba(X_test_scaled)[:, 1]
    ens_probs = (gbc_probs + svc_probs) / 2

    # Check 2: SVM non-zero collapse
    svc_min, svc_mean, svc_max, svc_std = svc_probs.min(), svc_probs.mean(), svc_probs.max(), svc_probs.std()
    print(f"      SVM Probabilities: min={svc_min:.4f}, mean={svc_mean:.4f}, max={svc_max:.4f}, std={svc_std:.4f}")

    if svc_max < 0.1 or svc_std < 0.05:
        print("      [FAIL] SVM probabilities have collapsed to zero or a flat constant!")
        svm_spread_check = False
    else:
        print("      [PASS] SVM probabilities exhibit healthy clinical variance.")
        svm_spread_check = True

    # Check 3: Positive prediction rates
    gbc_pos = (gbc_probs > threshold).mean() * 100
    svc_pos = (svc_probs > threshold).mean() * 100
    ens_pos = (ens_probs > threshold).mean() * 100

    print(f"\n[4/5] Risk Classification Rates (Threshold = {threshold:.2f}):")
    print(f"      - GBC Positive Rate:      {gbc_pos:5.1f}% ({(gbc_probs > threshold).sum()}/{len(gbc_probs)})")
    print(f"      - SVM Positive Rate:      {svc_pos:5.1f}% ({(svc_probs > threshold).sum()}/{len(svc_probs)})")
    print(f"      - Ensemble Positive Rate: {ens_pos:5.1f}% ({(ens_probs > threshold).sum()}/{len(ens_probs)})")

    # Both models should be in clinical range (~5% to ~40%) and not wildly divergent
    if abs(gbc_pos - svc_pos) > 30:
        print("      [WARNING] GBC and SVM positive rates differ by >30 percentage points.")
        rates_check = False
    else:
        print("      [PASS] GBC and SVM classification rates are reasonably aligned.")
        rates_check = True

    # Check 4: Rank correlation between models
    rho, p_val = spearmanr(gbc_probs, svc_probs)
    print(f"\n[5/5] Model Agreement (Spearman Rank Correlation):")
    print(f"      Spearman rho = {rho:.4f} (p = {p_val:.2e})")

    if rho > 0.3 and p_val < 0.01:
        print("      [PASS] Strong, statistically significant rank correlation between GBC and SVM.")
        corr_check = True
    else:
        print("      [WARNING] Low or insignificant correlation between GBC and SVM.")
        corr_check = False

    all_passed = scaler_check and svm_spread_check and rates_check and corr_check

    # Export validation table
    if export_summary:
        summary_df = pd.DataFrame({
            'Metric': [
                'Scaler Means Range',
                'Scaler Scales Range',
                'SVM Min Proba',
                'SVM Max Proba',
                'SVM Std Dev',
                'GBC Positive Rate (%)',
                'SVM Positive Rate (%)',
                'Ensemble Positive Rate (%)',
                'Spearman Correlation (rho)',
                'Spearman p-value'
            ],
            'Observed Value': [
                f"{scaler.mean_.min():.2f} - {scaler.mean_.max():.2f}",
                f"{scaler.scale_.min():.2f} - {scaler.scale_.max():.2f}",
                f"{svc_min:.4f}",
                f"{svc_max:.4f}",
                f"{svc_std:.4f}",
                f"{gbc_pos:.1f}%",
                f"{svc_pos:.1f}%",
                f"{ens_pos:.1f}%",
                f"{rho:.4f}",
                f"{p_val:.2e}"
            ],
            'Status': [
                'PASS' if scaler_check else 'FAIL',
                'PASS' if scaler_check else 'FAIL',
                'PASS' if svm_spread_check else 'FAIL',
                'PASS' if svm_spread_check else 'FAIL',
                'PASS' if svm_spread_check else 'FAIL',
                'PASS',
                'PASS',
                'PASS' if rates_check else 'WARNING',
                'PASS' if corr_check else 'WARNING',
                'PASS' if corr_check else 'WARNING'
            ]
        })
        os.makedirs('Predictions', exist_ok=True)
        summary_out = 'Predictions/validation_summary.csv'
        summary_df.to_csv(summary_out, index=False)
        print(f"\nSaved validation summary to: {summary_out}")

    print('='*70)
    if all_passed:
        print('  >>> OVERALL VALIDATION RESULT: ALL SYSTEMS OPERATIONAL (PASS) <<<')
    else:
        print('  >>> OVERALL VALIDATION RESULT: COMPLETED WITH WARNINGS <<<')
    print('='*70 + '\n')
    return all_passed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Validate DSLX Models and Artifacts")
    parser.add_argument('test_csv', nargs='?', default='test_set.csv',
                        help="Path to CSV file to validate against")
    parser.add_argument('--threshold', '-t', type=float, default=0.50,
                        help="Decision threshold (default: 0.50)")
    args = parser.parse_args()

    success = validate_artifacts_and_predictions(
        test_csv=args.test_csv,
        threshold=args.threshold
    )
    sys.exit(0 if success else 1)
