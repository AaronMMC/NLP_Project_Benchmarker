"""
plots.py — All visualisations (PNG + SVG in results/figures/).

Generates normalised confusion matrices, ROC/PR curves, metric comparisons,
per-song accuracy heatmaps, learning curves, coverage plots, Zipf plots,
calibration curves, and the core lexicon×genre benchmark heatmap.
"""

import os
import logging
from typing import List, Dict, Optional, Tuple, Any
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.calibration import calibration_curve

logger = logging.getLogger(__name__)

# Consistent colour scheme per lexicon condition
LEXICON_COLOURS = {
    'none': '#999999',
    'house': '#2196F3',
    'existing': '#FF9800',
    'combined': '#4CAF50',
    'wiktionary': '#9C27B0',
    'panlex': '#F44336',
    'user_supplied': '#795548',
}

# Plot style
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.dpi': 150,
})


def _save_fig(fig, name: str, output_dir: str = 'results/figures'):
    """Save figure as both PNG and SVG."""
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, f'{name}.png'), bbox_inches='tight', dpi=150)
    fig.savefig(os.path.join(output_dir, f'{name}.svg'), bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved figure: {name}")


def plot_confusion_matrix(y_true, y_pred, labels, title: str,
                          output_dir: str = 'results/figures',
                          name: str = 'confusion_matrix'):
    """Plot normalised confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Normalised Confusion Matrix: {title}')
    _save_fig(fig, name, output_dir)


def plot_metric_comparison(results_df: pd.DataFrame, metric: str,
                           title: str = '', output_dir: str = 'results/figures',
                           name: str = 'metric_comparison'):
    """
    Bar plot comparing a metric across lexicon conditions and models.

    results_df should have columns: model, condition, <metric>
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    conditions = results_df['condition'].unique()
    models = results_df['model'].unique()
    x = np.arange(len(models))
    width = 0.8 / len(conditions)

    for i, cond in enumerate(conditions):
        subset = results_df[results_df['condition'] == cond]
        values = []
        for model in models:
            row = subset[subset['model'] == model]
            values.append(row[metric].values[0] if len(row) > 0 else 0)
        colour = LEXICON_COLOURS.get(cond, '#666666')
        ax.bar(x + i * width, values, width, label=cond, color=colour, alpha=0.85)

    ax.set_xlabel('Model')
    ax.set_ylabel(metric)
    ax.set_title(title or f'{metric} by Model and Lexicon Condition')
    ax.set_xticks(x + width * len(conditions) / 2)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(title='Lexicon Condition')
    ax.grid(axis='y', alpha=0.3)

    _save_fig(fig, name, output_dir)


def plot_lexicon_genre_heatmap(grid_df: pd.DataFrame, metric: str,
                                title: str = '',
                                output_dir: str = 'results/figures',
                                name: str = 'lexicon_genre_heatmap'):
    """
    Core benchmark figure: heatmap of lexicon condition × genre.

    grid_df should have rows=conditions, columns=genres, values=metric.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Replace NaN with a distinct value for display
    display_df = grid_df.copy()

    sns.heatmap(display_df, annot=True, fmt='.3f', cmap='YlOrRd',
                ax=ax, vmin=0, vmax=1, linewidths=0.5,
                mask=display_df.isna(),
                cbar_kws={'label': metric})

    ax.set_title(title or f'Lexicon Condition × Genre: {metric}')
    ax.set_xlabel('Genre')
    ax.set_ylabel('Lexicon Condition')

    # Mark missing cells
    for i in range(display_df.shape[0]):
        for j in range(display_df.shape[1]):
            if pd.isna(display_df.iloc[i, j]):
                ax.text(j + 0.5, i + 0.5, 'no data',
                       ha='center', va='center', fontsize=8, color='gray')

    _save_fig(fig, name, output_dir)


def plot_coverage_comparison(coverage_data: Dict[str, Dict], title: str = '',
                              output_dir: str = 'results/figures',
                              name: str = 'coverage_comparison'):
    """
    Bar chart of lexicon coverage before and after variant normalisation.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    conditions = list(coverage_data.keys())
    x = np.arange(len(conditions))
    width = 0.35

    before = [coverage_data[c].get('before_coverage_pct', 0) for c in conditions]
    after = [coverage_data[c].get('after_coverage_pct', 0) for c in conditions]

    bars1 = ax.bar(x - width/2, before, width, label='Before normalisation',
                   color='#FF9800', alpha=0.8)
    bars2 = ax.bar(x + width/2, after, width, label='After normalisation',
                   color='#4CAF50', alpha=0.8)

    ax.set_xlabel('Lexicon Condition')
    ax.set_ylabel('Coverage (%)')
    ax.set_title(title or 'Lexicon Coverage: Before vs After Spelling Normalisation')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=8)

    _save_fig(fig, name, output_dir)


def plot_zipf(tokens: List[str], title: str = '',
              output_dir: str = 'results/figures',
              name: str = 'zipf_plot'):
    """Zipf's law plot: log(rank) vs log(frequency)."""
    freq = Counter(tokens)
    ranks = range(1, len(freq) + 1)
    freqs = sorted(freq.values(), reverse=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(ranks, freqs, 'b.', alpha=0.6, markersize=4)
    ax.set_xlabel('Rank (log scale)')
    ax.set_ylabel('Frequency (log scale)')
    ax.set_title(title or "Zipf's Law: Word Frequency Distribution")
    ax.grid(True, alpha=0.3)
    _save_fig(fig, name, output_dir)


def plot_leakage_inflation(regime_results: Dict[str, Dict[str, float]],
                           output_dir: str = 'results/figures',
                           name: str = 'leakage_inflation'):
    """
    Plot showing leakage inflation across split regimes.

    regime_results: {regime_name: {metric: value}}
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    regimes = list(regime_results.keys())
    metrics = ['f1', 'mcc', 'accuracy']
    x = np.arange(len(metrics))
    width = 0.8 / len(regimes)

    colours = ['#2196F3', '#FF9800', '#4CAF50']
    for i, regime in enumerate(regimes):
        values = [regime_results[regime].get(m, 0) for m in metrics]
        ax.bar(x + i * width, values, width, label=regime,
               color=colours[i % len(colours)], alpha=0.85)

    ax.set_xlabel('Metric')
    ax.set_ylabel('Score')
    ax.set_title('Leakage Inflation: Grouped vs Non-Grouped vs Deduplicated')
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    _save_fig(fig, name, output_dir)


def plot_per_song_heatmap(song_metrics: pd.DataFrame,
                           output_dir: str = 'results/figures',
                           name: str = 'per_song_heatmap'):
    """Heatmap of per-song retrieval accuracy."""
    if song_metrics.empty:
        logger.info("No per-song metrics to plot")
        return

    fig, ax = plt.subplots(figsize=(12, max(6, len(song_metrics) * 0.4)))
    sns.heatmap(song_metrics, annot=True, fmt='.2f', cmap='RdYlGn',
                ax=ax, vmin=0, vmax=1)
    ax.set_title('Per-Song Retrieval Accuracy')
    _save_fig(fig, name, output_dir)


def plot_corpus_stats(stats, output_dir: str = 'results/figures'):
    """Plot corpus statistics: unit size distribution."""
    if not stats.lines_per_unit:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Lines per unit
    axes[0].bar(range(len(stats.lines_per_unit)), stats.lines_per_unit,
                color='#2196F3', alpha=0.8)
    axes[0].set_xlabel('Unit Index')
    axes[0].set_ylabel('Number of Lines')
    axes[0].set_title('Lines per Liturgical Unit')

    # Tokens per unit
    axes[1].bar(range(len(stats.tokens_per_unit)), stats.tokens_per_unit,
                color='#4CAF50', alpha=0.8)
    axes[1].set_xlabel('Unit Index')
    axes[1].set_ylabel('Number of Tokens')
    axes[1].set_title('Tokens per Liturgical Unit')

    fig.tight_layout()
    _save_fig(fig, 'corpus_unit_sizes', output_dir)


def plot_retrieval_results(retrieval_metrics: Dict[str, float],
                            output_dir: str = 'results/figures',
                            name: str = 'retrieval_metrics'):
    """Bar chart of retrieval metrics (top-k, MRR)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    metrics = {k: v for k, v in retrieval_metrics.items()}
    names = list(metrics.keys())
    values = list(metrics.values())

    bars = ax.bar(names, values, color='#2196F3', alpha=0.85)
    ax.set_ylabel('Score')
    ax.set_title('Song Retrieval Performance (Task C)')
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    _save_fig(fig, name, output_dir)


def plot_coverage_masking(masking_df: pd.DataFrame,
                          output_dir: str = 'results/figures',
                          name: str = 'coverage_masking_curve'):
    """Plot performance / coverage vs available lexicon coverage."""
    if masking_df.empty:
        return

    fig, ax1 = plt.subplots(figsize=(9, 5))

    grouped = masking_df.groupby('mask_fraction')
    fractions = sorted(masking_df['mask_fraction'].unique())

    cov_means = [grouped.get_group(f)['coverage_pct'].mean() for f in fractions]
    cov_stds = [grouped.get_group(f)['coverage_pct'].std() if len(grouped.get_group(f)) > 1 else 0 for f in fractions]

    color = '#2196F3'
    ax1.set_xlabel('Mask Fraction (fraction of headwords removed)')
    ax1.set_ylabel('Lexicon Coverage (%)', color=color)
    ax1.errorbar(fractions, cov_means, yerr=cov_stds, fmt='-o', color=color, capsize=4, label='Lexicon Coverage')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    if 'f1_macro' in masking_df.columns:
        ax2 = ax1.twinx()
        color2 = '#E91E63'
        f1_means = [grouped.get_group(f)['f1_macro'].mean() for f in fractions]
        f1_stds = [grouped.get_group(f)['f1_macro'].std() if len(grouped.get_group(f)) > 1 else 0 for f in fractions]
        ax2.set_ylabel('Task C Macro F1', color=color2)
        ax2.errorbar(fractions, f1_means, yerr=f1_stds, fmt='--s', color=color2, capsize=4, label='Task C F1')
        ax2.tick_params(axis='y', labelcolor=color2)

    plt.title('Lexicon Coverage Masking: Performance vs Mask Fraction')
    fig.tight_layout()
    _save_fig(fig, name, output_dir)

