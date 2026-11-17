#!/usr/bin/env python3
"""
Rebuild Artifacts Module for DSLX_Project.

Reconstructs modern, clean joblib artifacts (StandardScaler, Imputer Medians,
GBC and SVM models, and feature metadata) from raw unnormalized training data
and original model hyperparameters. Eliminates legacy pickling errors and fixes
the SVM scaling issue.
"""

import os
import sys
import json
import argparse
import pickle
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

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


def extract_features_from_raw(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts the 16 canonical features from a raw dataframe, whether it is
    already pre-formatted or a full study export (like Juventus_TP123_predykcje.csv).
    """
    # Check if all 16 columns already exist
    if all(col in df.columns for col in FEATURE_COLUMNS):
        extracted = df[FEATURE_COLUMNS].copy()
        for col in FEATURE_COLUMNS:
            if extracted[col].dtype == object:
                extracted[col] = pd.to_numeric(
                    extracted[col].astype(str).str.replace(',', '.').str.strip(),
                    errors='coerce'
                )
        return extracted

    # Otherwise, map from longitudinal study export
    extracted = pd.DataFrame()
    for col in FEATURE_COLUMNS:
        if col in df.columns:
            extracted[col] = df[col]
        else:
            mapped_src = None
            if col in QUESTIONNAIRE_MAPPING:
                prefix = QUESTIONNAIRE_MAPPING[col]
                for c in df.columns:
                    if c.startswith(prefix) or prefix in c:
                        mapped_src = c
                        break
            if mapped_src is not None:
                extracted[col] = df[mapped_src]
            else:
                raise KeyError(f'Could not map required feature: {col}')

    # Clean numeric format (Polish decimal commas etc.)
    for col in FEATURE_COLUMNS:
        if extracted[col].dtype == object:
            extracted[col] = pd.to_numeric(
                extracted[col].astype(str).str.replace(',', '.').str.strip(),
                errors='coerce'
            )

    # In longitudinal study exports, Grafotaktyka is raw out of 30, whereas DSLX uses scaled out of 20 (ratio 1.5)
    if 'TURA1_GrafotaktykaSurowy' in extracted.columns:
        if extracted['TURA1_GrafotaktykaSurowy'].mean() > 15:
            extracted['TURA1_GrafotaktykaSurowy'] = extracted['TURA1_GrafotaktykaSurowy'] / 1.5

    return extracted[FEATURE_COLUMNS]


def rebuild_artifacts(
    raw_train_path: str = 'test_set.csv',
    legacy_models_dir: str = 'App/Models',
    output_artifacts_dir: str = 'artifacts',
    sync_to_app_models: bool = True
):
    """
    Rebuilds modern joblib artifacts from raw training data and legacy parameters.
    """
    print(f'Starting artifact rebuild...')
    print(f'Raw training data source: {raw_train_path}')
    print(f'Legacy models directory:  {legacy_models_dir}')
    print(f'Output artifacts dir:     {output_artifacts_dir}')

    os.makedirs(output_artifacts_dir, exist_ok=True)

    # 1. Load raw training data
    if raw_train_path.endswith('.csv'):
        try:
            raw_df = pd.read_csv(raw_train_path, low_memory=False)
        except UnicodeDecodeError:
            raw_df = pd.read_csv(raw_train_path, sep=';', encoding='cp1250', low_memory=False)
    else:
        raise ValueError(f'Unsupported file format: {raw_train_path}')

    if 'Unnamed: 0' in raw_df.columns:
        raw_df = raw_df.drop(columns=['Unnamed: 0'])

    raw_features = extract_features_from_raw(raw_df)
    print(f'Extracted raw features shape: {raw_features.shape}')

    # Drop NaNs for scaler fitting to prevent biased moments
    clean_raw_features = raw_features.dropna()
    print(f'Clean complete cases for scaler fitting: {len(clean_raw_features)}')

    # 2. Fit fresh StandardScaler on raw unnormalized training data
    scaler = StandardScaler()
    scaler.fit(clean_raw_features)
    print(f'Fitted StandardScaler successfully!')
    print(f'  Means min={scaler.mean_.min():.2f}, mean={scaler.mean_.mean():.2f}, max={scaler.mean_.max():.2f}')
    print(f'  Scales min={scaler.scale_.min():.2f}, mean={scaler.scale_.mean():.2f}, max={scaler.scale_.max():.2f}')

    # 3. Compute deterministic training medians for imputation
    imputer_medians = clean_raw_features.median().to_dict()

    # 4. Load legacy training data & hyperparameters
    x_gbc_path = os.path.join(legacy_models_dir, 'X_gbc.csv')
    y_gbc_path = os.path.join(legacy_models_dir, 'y_gbc.csv')
    x_svc_path = os.path.join(legacy_models_dir, 'X_svc.csv')
    y_svc_path = os.path.join(legacy_models_dir, 'y_svc.csv')
    gbc_pkl_path = os.path.join(legacy_models_dir, 'gbc_params.pkl')
    svc_pkl_path = os.path.join(legacy_models_dir, 'svc_params.pkl')

    X_gbc = pd.read_csv(x_gbc_path)
    y_gbc = pd.read_csv(y_gbc_path).values.ravel()
    X_svc = pd.read_csv(x_svc_path)
    y_svc = pd.read_csv(y_svc_path).values.ravel()

    with open(gbc_pkl_path, 'rb') as f:
        gbc_params = pickle.load(f)
    with open(svc_pkl_path, 'rb') as f:
        svc_params = pickle.load(f)

    print('Loaded original hyperparameters and training matrices.')

    # 5. Train modern estimators with original hyperparameters
    print('Training GradientBoostingClassifier...')
    gbc = GradientBoostingClassifier(**gbc_params, random_state=4)
    gbc.fit(X_gbc, y_gbc)

    print('Training Support Vector Classifier (SVC)...')
    svc = SVC(**svc_params, random_state=12345)
    svc.fit(X_svc, y_svc)

    # 6. Save joblib artifacts
    scaler_out = os.path.join(output_artifacts_dir, 'scaler.joblib')
    medians_out = os.path.join(output_artifacts_dir, 'imputer_medians.joblib')
    gbc_out = os.path.join(output_artifacts_dir, 'gbc_model.joblib')
    svc_out = os.path.join(output_artifacts_dir, 'svm_model.joblib')
    metadata_out = os.path.join(output_artifacts_dir, 'feature_metadata.json')

    joblib.dump(scaler, scaler_out)
    joblib.dump(imputer_medians, medians_out)
    joblib.dump(gbc, gbc_out)
    joblib.dump(svc, svc_out)

    feature_metadata = {
        'feature_names': FEATURE_COLUMNS,
        'means': {col: float(m) for col, m in zip(FEATURE_COLUMNS, scaler.mean_)},
        'scales': {col: float(s) for col, s in zip(FEATURE_COLUMNS, scaler.scale_)},
        'medians': {col: float(v) for col, v in imputer_medians.items()},
        'min': {col: float(clean_raw_features[col].min()) for col in FEATURE_COLUMNS},
        'max': {col: float(clean_raw_features[col].max()) for col in FEATURE_COLUMNS},
        'n_samples_scaler_fit': len(clean_raw_features)
    }

    with open(metadata_out, 'w') as f:
        json.dump(feature_metadata, f, indent=2)

    print(f'Exported modern artifacts to {output_artifacts_dir}/:')
    print(f'  - scaler.joblib')
    print(f'  - imputer_medians.joblib')
    print(f'  - gbc_model.joblib')
    print(f'  - svm_model.joblib')
    print(f'  - feature_metadata.json')

    if sync_to_app_models:
        os.makedirs(legacy_models_dir, exist_ok=True)
        # Also persist modern joblib versions in App/Models
        joblib.dump(scaler, os.path.join(legacy_models_dir, 'scaler.joblib'))
        joblib.dump(imputer_medians, os.path.join(legacy_models_dir, 'imputer_medians.joblib'))
        joblib.dump(gbc, os.path.join(legacy_models_dir, 'gbc_model.joblib'))
        joblib.dump(svc, os.path.join(legacy_models_dir, 'svm_model.joblib'))
        # Overwrite legacy scaler1.pkl with properly fitted scaler for backward compatibility
        with open(os.path.join(legacy_models_dir, 'scaler1.pkl'), 'wb') as f:
            pickle.dump(scaler, f)
        print(f'Synced modern artifacts to {legacy_models_dir}/ (including updated scaler1.pkl).')

    print('\nArtifact rebuild complete!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rebuild DSLX Model Artifacts')
    parser.add_argument('--raw-train', type=str, default='test_set.csv',
                        help='Path to raw unnormalized training CSV (e.g. test_set.csv or Juventus_TP123_predykcje.csv)')
    parser.add_argument('--legacy-models-dir', type=str, default='App/Models',
                        help='Directory with legacy training data and params')
    parser.add_argument('--output-artifacts-dir', type=str, default='artifacts',
                        help='Directory to output modern joblib artifacts')
    args = parser.parse_args()

    rebuild_artifacts(
        raw_train_path=args.raw_train,
        legacy_models_dir=args.legacy_models_dir,
        output_artifacts_dir=args.output_artifacts_dir
    )
