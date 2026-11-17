#!/usr/bin/env python3
"""
DSLX Prediction Engine (Modernized 2.0).

Performs dyslexia risk prediction using an ensemble of Gradient Boosting Classifier (GBC)
and Support Vector Classifier (SVC). Part of the Dyslexia Project in cooperation with
Nencki Institute of Experimental Biology, Warsaw, Poland.
"""

__author__ = "Aleksander Molak (Modernized for Nencki Lab)"
__copyright__ = "Copyright 2018-2026, DSLX Project"
__version__ = "2.0.0"
__status__ = "Production"

import os
import sys
import glob
import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
import joblib

# Friendly greeting phrases (preserving original UX)
HELLO_PHRASES = ['Jak się masz?', 'Jak Ci mija dzień?', 'Dzień dobry!', 'Cześć!']
GOODBYE_PHRASES = ['Do zobaczenia!', 'Miłego dnia!', 'Wszystkiego dobrego!']

# Canonical 16 features in exact required order
FEATURE_COLUMNS = [
    'TURA1_ZnajomoscLiter',
    'TURA1_AnalizaFonemowa',
    'TURA1_UsuwanieFonemow',
    'TURA1_TSN_PrzedmiotySurowy',
    'TURA1_TSN_KolorySurowy',
    'TURA1_SlownictwoSurowy',
    'TURA1_GrafotaktykaSurowy',
    'TURA1_PowtarzanieCyfr',
    'TURA1_PowtarzanieSylab',
    'SRD_21',
    'ARHQ_M_17',
    'ARHQ_M_21',
    'SRD_29',
    'SRD_1',
    'ARHQ_M_10',
    'SRD_9',
]

QUESTIONNAIRE_MAPPING = {
    'ARHQ_M_10': 'ARHQ_MATKA: arhq-10 ',
    'ARHQ_M_17': 'ARHQ_MATKA: arhq-17 ',
    'ARHQ_M_21': 'ARHQ_MATKA: arhq-21 ',
    'SRD_1': 'SRD: srd-1 ',
    'SRD_9': 'SRD: srd-9 ',
    'SRD_21': 'SRD: srd-21 ',
    'SRD_29': 'SRD: srd-29 ',
}


def find_artifacts_dir() -> str:
    """Locates the directory containing joblib artifacts."""
    candidates = [
        'artifacts',
        os.path.join(os.path.dirname(__file__), 'artifacts'),
        'App/Models',
        os.path.join(os.path.dirname(__file__), 'App', 'Models'),
        'Models',
        os.path.join(os.path.dirname(__file__), 'Models')
    ]
    for path in candidates:
        if os.path.exists(os.path.join(path, 'scaler.joblib')):
            return path
    raise FileNotFoundError(
        "Could not find model artifacts (scaler.joblib). "
        "Please run 'python rebuild_artifacts.py' first."
    )


def clean_and_extract_df(input_file: str) -> tuple[pd.DataFrame, pd.Series | None, pd.DataFrame]:
    """
    Loads raw CSV, identifies ID column, handles comma decimals / placeholders,
    maps questionnaire questions if necessary, and extracts the 16 canonical features.
    """
    # 1. Load CSV with robust delimiter and encoding detection
    df = None
    for enc in ['utf-8', 'cp1250', 'iso-8859-2', 'latin1']:
        for sep in [',', ';', '\t']:
            try:
                temp_df = pd.read_csv(input_file, sep=sep, encoding=enc, low_memory=False)
                if temp_df.shape[1] > 1:
                    df = temp_df
                    break
            except Exception:
                continue
        if df is not None:
            break

    if df is None:
        raise ValueError(f"Could not parse CSV file: {input_file}")

    # Clean string placeholders into NaN across dataframe
    df = df.replace(to_replace=[r'^\s*$', r'^-+$', r'^\?+$', r'^None$', r'^nan$'], value=np.nan, regex=True)

    # 2. Extract Identifier (ID, kod, etc.)
    id_series = None
    for id_col in ['ID', 'kod', 'Kod', 'id', 'subject_id', 'Subject', 'Unnamed: 0']:
        if id_col in df.columns:
            id_series = df[id_col].copy()
            id_series.name = 'ID'
            break

    # 3. Extract canonical 16 features
    has_canonical = all(col in df.columns for col in FEATURE_COLUMNS)
    extracted = pd.DataFrame(index=df.index)

    if has_canonical:
        for col in FEATURE_COLUMNS:
            val = df[col]
            if val.dtype == object:
                val = val.astype(str).str.replace(',', '.').str.strip()
            extracted[col] = pd.to_numeric(val, errors='coerce')
    else:
        # Check mapping from longitudinal export
        missing_unmapped = []
        for col in FEATURE_COLUMNS:
            if col in df.columns:
                val = df[col]
                if val.dtype == object:
                    val = val.astype(str).str.replace(',', '.').str.strip()
                extracted[col] = pd.to_numeric(val, errors='coerce')
            elif col in QUESTIONNAIRE_MAPPING:
                prefix = QUESTIONNAIRE_MAPPING[col]
                mapped_col = None
                for c in df.columns:
                    if prefix in c:
                        mapped_col = c
                        break
                if mapped_col is not None:
                    val = df[mapped_col]
                    if val.dtype == object:
                        val = val.astype(str).str.replace(',', '.').str.strip()
                    extracted[col] = pd.to_numeric(val, errors='coerce')
                else:
                    missing_unmapped.append(col)
            else:
                missing_unmapped.append(col)

        if missing_unmapped:
            raise ValueError(
                f"Column names in your file do not match expected features.\n"
                f"Missing required columns:\n{missing_unmapped}\n"
                f"Your file should contain:\n{FEATURE_COLUMNS}"
            )

        # Standardize Grafotaktyka if out of 30
        if 'TURA1_GrafotaktykaSurowy' in extracted.columns:
            if extracted['TURA1_GrafotaktykaSurowy'].mean(skipna=True) > 15:
                extracted['TURA1_GrafotaktykaSurowy'] = extracted['TURA1_GrafotaktykaSurowy'] / 1.5

    return df, id_series, extracted[FEATURE_COLUMNS]


def run_prediction(
    input_file: str,
    threshold: float = 0.5,
    check_quality: bool = False,
    plot: bool = False,
    output_dir: str = 'Predictions'
):
    """Executes the full prediction workflow."""
    print(f"\nDslxApp v. {__version__} ({__status__}) -- Modernized Nencki Lab Suite\n")
    hello_phrase = HELLO_PHRASES[np.random.randint(len(HELLO_PHRASES))]
    print(f"\n{hello_phrase} :)\n\nWczytuję moduły i artefakty...\n")

    # Locate and load artifacts
    artifacts_dir = find_artifacts_dir()
    scaler = joblib.load(os.path.join(artifacts_dir, 'scaler.joblib'))
    gbc = joblib.load(os.path.join(artifacts_dir, 'gbc_model.joblib'))
    svc = joblib.load(os.path.join(artifacts_dir, 'svm_model.joblib'))

    medians_path = os.path.join(artifacts_dir, 'imputer_medians.joblib')
    if os.path.exists(medians_path):
        medians = joblib.load(medians_path)
    else:
        medians = {col: float(m) for col, m in zip(FEATURE_COLUMNS, scaler.mean_)}

    print(f"Artefakty załadowane z: {artifacts_dir}\n")

    # Clean and extract input
    print("Sprawdzam strukturę danych...")
    raw_df, id_series, X_test = clean_and_extract_df(input_file)
    print(f"Struktura danych OK! Wczytano {len(X_test)} obserwacji.\n")

    # Missingness check and median imputation
    missing_pcts = X_test.isna().mean(axis=1)
    high_missing = X_test.index[missing_pcts > 0.20].tolist()
    if high_missing:
        print(f"[!] OSTRZEŻENIE: {len(high_missing)} obserwacji ma >20% brakujących subtestów.")
        print(f"    Wskaźniki wierszy: {high_missing}")

    n_missing_vals = X_test.isna().sum().sum()
    if n_missing_vals > 0:
        print(f"Uzupełniam {n_missing_vals} brakujących wartości medianami ze zbioru uczącego...")
        X_test_imputed = X_test.fillna(medians)
    else:
        X_test_imputed = X_test.copy()

    # Scale the features
    print("Przekształcam dane (StandardScaler)...")
    X_test_scaled = scaler.transform(X_test_imputed)
    print("Dane przekształcone!\n")

    # Run predictions
    print("Generuję predykcje...")
    gbc_proba = gbc.predict_proba(X_test_scaled)
    svc_proba = svc.predict_proba(X_test_scaled)
    meta_proba = (gbc_proba + svc_proba) / 2

    # Predict positive risk class based on threshold
    meta_pred = (meta_proba[:, 1] > threshold).astype(int)
    print(f"Predykcje wygenerowane! Zastosowany próg decyzyjny: {threshold:.2f}\n")

    # Format result dataframe
    results_df = pd.DataFrame()
    if id_series is not None:
        results_df['ID'] = id_series.values
    else:
        results_df['ID'] = np.arange(len(X_test))

    for col in FEATURE_COLUMNS:
        results_df[col] = X_test_imputed[col].values

    results_df['P_GBC'] = np.round(gbc_proba[:, 1], 4)
    results_df['P_SVM'] = np.round(svc_proba[:, 1], 4)
    results_df['P_Ensemble'] = np.round(meta_proba[:, 1], 4)
    results_df['Prog_Decyzyjny'] = threshold
    results_df['Predykcja'] = meta_pred

    # Optional quality checks
    if check_quality:
        try:
            from dslx_quality import check_sample_quality, print_quality_report
            metadata_file = os.path.join(artifacts_dir, 'feature_metadata.json')
            q_summary = check_sample_quality(X_test, feature_metadata_path=metadata_file)
            print_quality_report(q_summary)
        except Exception as e:
            print(f"[!] Błąd modułu dslx_quality: {e}")

    # Optional plotting
    if plot:
        try:
            from dslx_viz import plot_diagnostics
            z_df = pd.DataFrame(X_test_scaled, columns=FEATURE_COLUMNS)
            plots_dir = os.path.join(output_dir, 'Plots')
            plot_diagnostics(
                gbc_proba=gbc_proba[:, 1],
                svc_proba=svc_proba[:, 1],
                ensemble_proba=meta_proba[:, 1],
                z_scored_features=z_df,
                threshold=threshold,
                output_dir=plots_dir
            )
        except Exception as e:
            print(f"[!] Błąd modułu dslx_viz: {e}")

    # Save output
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_DslxApp_Prediction.csv"
    output_path = os.path.join(output_dir, filename)

    print(f"Zapisuję Twoje predykcje...")
    results_df.to_csv(output_path, index=False)
    print(f"Plik {output_path} został pomyślnie zapisany!")

    pos_count = int(meta_pred.sum())
    pos_pct = (pos_count / len(meta_pred)) * 100
    print(f"Podsumowanie: {pos_count}/{len(meta_pred)} ({pos_pct:.1f}%) zaklasyfikowano jako podwyższone ryzyko dysleksji.")

    goodbye_phrase = GOODBYE_PHRASES[np.random.randint(len(GOODBYE_PHRASES))]
    print(f"\n{goodbye_phrase}\n")
    return output_path


def parse_arguments():
    """Parses command-line arguments supporting both positional and flagged syntax."""
    # Custom parser allowing legacy syntax: python dslx.py <file> [threshold]
    parser = argparse.ArgumentParser(
        description="DSLX Dyslexia Risk Prediction Engine",
        usage="%(prog)s [input_csv] [threshold] [options]"
    )
    parser.add_argument('input_csv', nargs='?', default=None,
                        help="Path to input test CSV file")
    parser.add_argument('threshold_pos', nargs='?', type=float, default=None,
                        help="Optional decision threshold (default: 0.5)")
    parser.add_argument('--threshold', '-t', type=float, default=None,
                        help="Decision threshold between 0.0 and 1.0 (default: 0.5)")
    parser.add_argument('--check-quality', action='store_true',
                        help="Run sample completeness, outlier, and drift tests")
    parser.add_argument('--plot', action='store_true',
                        help="Generate diagnostic visualizations and threshold simulation")
    parser.add_argument('--output-dir', type=str, default='Predictions',
                        help="Directory to save prediction CSV and plots")

    args = parser.parse_args()

    if args.input_csv is None:
        parser.print_help()
        sys.exit(1)

    threshold = 0.5
    if args.threshold is not None:
        threshold = args.threshold
    elif args.threshold_pos is not None:
        threshold = args.threshold_pos

    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"Threshold must be between 0.0 and 1.0, got: {threshold}")

    return args.input_csv, threshold, args.check_quality, args.plot, args.output_dir


if __name__ == '__main__':
    csv_file, thresh, q_flag, p_flag, out_dir = parse_arguments()
    run_prediction(
        input_file=csv_file,
        threshold=thresh,
        check_quality=q_flag,
        plot=p_flag,
        output_dir=out_dir
    )
