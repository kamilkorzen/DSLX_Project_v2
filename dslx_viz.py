#!/usr/bin/env python3
"""
Diagnostics & Visualization Module for DSLX_Project.

Generates:
1. Probability Distributions & Threshold Simulation (`probability_distributions.png`):
   - Side-by-side histograms/KDEs for GBC, SVM, and Ensemble.
   - Threshold simulation curve showing predicted positive rate across thresholds (0.0 to 1.0)
     with the selected decision threshold highlighted.
2. Model Agreement Scatter Plot (`model_agreement_scatter.png`):
   - GBC vs. SVM probabilities with decision boundaries and Spearman correlation.
3. Standardized Feature Boxplots (`feature_zscore_boxplots.png`):
   - Z-scored distributions of all input features against baseline (mean=0, SD=1).
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr


def plot_diagnostics(
    gbc_proba: np.ndarray,
    svc_proba: np.ndarray,
    ensemble_proba: np.ndarray,
    z_scored_features: pd.DataFrame,
    threshold: float = 0.5,
    output_dir: str = 'Plots'
):
    """
    Generates and saves the diagnostic visualization figures.
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style='whitegrid', font_scale=1.0)

    # -------------------------------------------------------------
    # 1. Probability Distributions & Threshold Simulation
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('DSLX Risk Probability Distributions & Threshold Simulation', fontsize=15, fontweight='bold')

    models = [
        ('Gradient Boosting (GBC)', gbc_proba, '#1f77b4', axes[0, 0]),
        ('Support Vector Machine (SVM)', svc_proba, '#2ca02c', axes[0, 1]),
        ('Ensemble Average (GBC + SVM) / 2', ensemble_proba, '#d62728', axes[1, 0])
    ]

    for name, probs, color, ax in models:
        sns.histplot(probs, bins=20, kde=True, color=color, ax=ax, stat='density', alpha=0.6)
        ax.axvline(threshold, color='black', linestyle='--', linewidth=1.5,
                   label=f'Threshold = {threshold:.2f}')
        pos_rate = (probs > threshold).mean() * 100
        ax.set_title(f'{name}\nPositive Rate: {pos_rate:.1f}% (N={len(probs)})', fontsize=11)
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('Density')
        ax.set_xlim(0, 1)
        ax.legend(loc='upper right')

    # Threshold Simulation Subplot (axes[1, 1])
    ax_sim = axes[1, 1]
    t_range = np.linspace(0.01, 0.99, 100)
    gbc_sim = [(gbc_proba > t).mean() * 100 for t in t_range]
    svc_sim = [(svc_proba > t).mean() * 100 for t in t_range]
    ens_sim = [(ensemble_proba > t).mean() * 100 for t in t_range]

    ax_sim.plot(t_range, gbc_sim, label='GBC', color='#1f77b4', linewidth=2)
    ax_sim.plot(t_range, svc_sim, label='SVM', color='#2ca02c', linewidth=2)
    ax_sim.plot(t_range, ens_sim, label='Ensemble', color='#d62728', linewidth=2.5)

    ax_sim.axvline(threshold, color='black', linestyle='--', linewidth=1.5,
                   label=f'Selected Threshold: {threshold:.2f}')
    ens_pos_at_t = (ensemble_proba > threshold).mean() * 100
    ax_sim.scatter([threshold], [ens_pos_at_t], color='#d62728', s=80, zorder=5)
    ax_sim.annotate(f'{ens_pos_at_t:.1f}% Positive',
                    xy=(threshold, ens_pos_at_t),
                    xytext=(threshold + 0.05, ens_pos_at_t + 5),
                    arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.2),
                    fontweight='bold', fontsize=9)

    ax_sim.set_title('Threshold Simulation: Positive Rate vs. Cutoff', fontsize=11, fontweight='bold')
    ax_sim.set_xlabel('Classification Threshold')
    ax_sim.set_ylabel('% Classified as High Risk')
    ax_sim.set_xlim(0, 1)
    ax_sim.set_ylim(-2, 102)
    ax_sim.legend(loc='upper right')

    plt.tight_layout()
    dist_path = os.path.join(output_dir, 'probability_distributions.png')
    fig.savefig(dist_path, dpi=200)
    plt.close(fig)
    print(f'Saved probability distribution plot: {dist_path}')

    # -------------------------------------------------------------
    # 2. Model Agreement Scatter Plot
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 8))
    rho, p_val = spearmanr(gbc_proba, svc_proba)

    # Classify ensemble prediction
    ens_class = (ensemble_proba > threshold).astype(int)
    palette = {0: '#1f77b4', 1: '#d62728'}
    labels = {0: f'Low Risk (≤{threshold:.2f})', 1: f'High Risk (>{threshold:.2f})'}

    for cls in [0, 1]:
        mask = ens_class == cls
        ax.scatter(gbc_proba[mask], svc_proba[mask], color=palette[cls],
                   label=labels[cls], alpha=0.7, s=50, edgecolors='none')

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Agreement (y = x)')
    ax.axvline(threshold, color='grey', linestyle=':', alpha=0.6)
    ax.axhline(threshold, color='grey', linestyle=':', alpha=0.6)

    ax.set_title(f'GBC vs. SVM Model Agreement\nSpearman Rank Correlation: ρ = {rho:.3f} (p = {p_val:.2e})',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('GBC Predicted Probability', fontsize=11)
    ax.set_ylabel('SVM Predicted Probability', fontsize=11)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc='upper left', frameon=True)

    plt.tight_layout()
    scatter_path = os.path.join(output_dir, 'model_agreement_scatter.png')
    fig.savefig(scatter_path, dpi=200)
    plt.close(fig)
    print(f'Saved model agreement plot: {scatter_path}')

    # -------------------------------------------------------------
    # 3. Standardized Feature Boxplots
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.boxplot(data=z_scored_features, ax=ax, orient='h', palette='Blues_r', fliersize=3)
    ax.axvline(0, color='red', linestyle='-', linewidth=1.5, label='Baseline Training Mean (Z=0)')
    ax.axvline(1, color='orange', linestyle='--', linewidth=1, alpha=0.7, label='Baseline ±1 SD')
    ax.axvline(-1, color='orange', linestyle='--', linewidth=1, alpha=0.7)
    ax.axvline(2, color='grey', linestyle=':', linewidth=1, alpha=0.7, label='Baseline ±2 SD')
    ax.axvline(-2, color='grey', linestyle=':', linewidth=1, alpha=0.7)
    ax.axvline(3, color='purple', linestyle=':', linewidth=1, alpha=0.7, label='Outlier Boundary ±3 SD')
    ax.axvline(-3, color='purple', linestyle=':', linewidth=1, alpha=0.7)

    ax.set_title('Standardized Feature Z-Score Distributions (vs. Baseline Training Cohort)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Standardized Z-Score [(Raw - Mean) / Scale]', fontsize=11)
    ax.set_ylabel('Subtest Feature', fontsize=11)
    ax.legend(loc='lower right', frameon=True)

    plt.tight_layout()
    box_path = os.path.join(output_dir, 'feature_zscore_boxplots.png')
    fig.savefig(box_path, dpi=200)
    plt.close(fig)
    print(f'Saved feature Z-score boxplot: {box_path}')
