# DSLX_Project (Modernized 2.0)

A machine learning framework for predicting dyslexia risk in children, developed in cooperation with the **Nencki Institute of Experimental Biology**, Warsaw, Poland.

---

## 1. Quick Start

### Installation
Ensure Python 3.10+ is installed, then install dependencies:
```bash
pip install -r requirements.txt
```

### Generating Predictions
You can run the prediction engine directly from the repository root:
```bash
# Standard run (default threshold = 0.50)
python dslx.py path/to/your_file.csv

# With a custom threshold (e.g., 0.45)
python dslx.py path/to/your_file.csv 0.45

# Full clinical run with Quality Audit and Diagnostic Visualizations
python dslx.py path/to/your_file.csv --check-quality --plot
```

*Note: Backward compatibility is fully preserved. The team can still run `cd App && python dslx.py <path_to_your_csv_file>`.*

Results will be automatically saved in the `Predictions/` folder with timestamped filenames:
- `Predictions/YYYY-MM-DD_HH-MM-SS_DslxApp_Prediction.csv`
- Diagnostic plots (when `--plot` is used) in `Predictions/Plots/`:
  - `probability_distributions.png` (Probability KDEs & Threshold Simulation Curve)
  - `model_agreement_scatter.png` (GBC vs. SVM agreement with Spearman correlation)
  - `feature_zscore_boxplots.png` (Z-score distributions vs. baseline cohort)

---

## 2. Input Data Format

The engine accepts either:
1. **Pre-extracted 16-feature CSVs** (such as `test_set.csv`) containing the canonical DSLX variables:
   - 9 cognitive tests: `TURA1_ZnajomoscLiter`, `TURA1_AnalizaFonemowa`, `TURA1_UsuwanieFonemow`, `TURA1_TSN_PrzedmiotySurowy`, `TURA1_TSN_KolorySurowy`, `TURA1_SlownictwoSurowy`, `TURA1_GrafotaktykaSurowy`, `TURA1_PowtarzanieCyfr`, `TURA1_PowtarzanieSylab`.
   - 7 parent questionnaire subtests: `SRD_21`, `ARHQ_M_17`, `ARHQ_M_21`, `SRD_29`, `SRD_1`, `ARHQ_M_10`, `SRD_9`.
2. **Full longitudinal study exports** (such as `Juventus_TP123_predykcje.csv`):
   - The engine automatically detects Polish text encoding (`cp1250`), comma decimal numbers (`24,00`), and maps questionnaire questions from full column titles.
   - Missing subtests are imputed using saved training cohort medians.

---

## 3. Validation & Quality Tools

### Automated Regression & Validation Suite
To verify that model artifacts are sound, `StandardScaler` parameters are realistic, and SVM probabilities exhibit proper clinical spread:
```bash
python validate_predictions.py test_set.csv
```
This generates a pass/fail audit report and exports `Predictions/validation_summary.csv`.

### Cohort Quality & Drift Audit
To test a new patient cohort for missingness (>20%), extreme outliers (±3 SD), or distribution drift (two-sample KS test):
```bash
python dslx_quality.py path/to/cohort.csv
```

---

## 4. Rebuilding Model Artifacts (One-Time Setup)

To re-fit the scaler on raw unnormalized training data and serialize modern `joblib` artifacts:
```bash
python rebuild_artifacts.py --raw-train test_set.csv
```
Artifacts are saved to `artifacts/` and synced with `App/Models/`.
