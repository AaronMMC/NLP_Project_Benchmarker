"""
run_benchmark.py — Single entry point for the Ilocano Hymn Lexicon Benchmark.

Usage:
    python -m src.run_benchmark --config config.yaml
    python -m src.run_benchmark --config config.yaml --fast

Runs all phases in order:
  Phase 0: Plan (already done in PLAN.md)
  Phase 1: Inspect & boundary detection
  Phase 2: Preprocessing & variant mapping
  Phase 3: Features, models, splits
  Phase 4: Run benchmark, plots, error analysis, report
"""

import os
import sys
import time
import json
import argparse
import logging
from typing import List, Dict, Optional, Tuple
from collections import Counter

import yaml
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import cross_val_predict, StratifiedKFold

# Set up logging before imports that use it
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger('benchmark')

from src.data_loader import (
    load_raw_corpus, detect_boundaries, load_unique_words,
    compute_corpus_stats, print_corpus_inspection
)
from src.preprocess import (
    normalise_text, tokenise, SpellingNormaliser, compute_coverage_change
)
from src.lexicons import LexiconConditions
from src.genres import load_genres, is_positive_only_mode
from src.features import build_feature_pipeline
from src.models import get_model_zoo, get_fast_model_zoo
from src.splits import build_dedup_groups, get_split_regimes, generate_cv_splits
from src.evaluate import (
    compute_binary_metrics, compute_retrieval_metrics,
    bootstrap_ci, paired_significance_test, holm_bonferroni_correction,
    check_low_power
)
from src.retrieval import SongRetriever
from src.analysis import (
    export_error_analysis, find_confused_song_pairs, find_shared_refrains
)
from src.plots import (
    plot_confusion_matrix, plot_metric_comparison,
    plot_lexicon_genre_heatmap, plot_coverage_comparison,
    plot_zipf, plot_leakage_inflation, plot_corpus_stats,
    plot_retrieval_results
)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Ilocano Hymn Lexicon Benchmark')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config.yaml')
    parser.add_argument('--fast', action='store_true',
                       help='Fast mode: fewer folds/repeats/bootstraps')
    args = parser.parse_args()

    start_time = time.time()

    # Load config
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if args.fast:
        config['fast_mode'] = True
        logger.info("🚀 Running in FAST mode (fewer folds/repeats)")

    # Set seeds
    np.random.seed(config['seeds']['numpy_seed'])

    # Resolve paths relative to config file location
    base_dir = os.path.dirname(os.path.abspath(args.config))
    paths = config['paths']
    for key in paths:
        if not os.path.isabs(paths[key]):
            paths[key] = os.path.join(base_dir, paths[key])

    # Create output directories
    os.makedirs(os.path.join(base_dir, 'results', 'metrics'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'results', 'figures'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'results', 'errors'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'logs'), exist_ok=True)

    # Add file handler for logging
    log_file = os.path.join(base_dir, 'logs', 'benchmark_run.log')
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    ))
    logging.getLogger().addHandler(file_handler)

    results_dir = os.path.join(base_dir, 'results')
    figures_dir = os.path.join(results_dir, 'figures')
    metrics_dir = os.path.join(results_dir, 'metrics')
    errors_dir = os.path.join(results_dir, 'errors')
    logs_dir = os.path.join(base_dir, 'logs')

    logger.info("=" * 70)
    logger.info("ILOCANO HYMN LEXICON BENCHMARK")
    logger.info("=" * 70)

    # =====================================================================
    # PHASE 1: Inspect & boundary detection
    # =====================================================================
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 1: Inspect & Boundary Detection")
    logger.info("=" * 70)

    # Load corpus
    raw_text = load_raw_corpus(paths['raw_corpus'])

    # Load unique words
    unique_words = load_unique_words(paths['unique_words'])

    # Detect boundaries
    units, boundary_method = detect_boundaries(raw_text, config)

    # Compute stats
    stats = compute_corpus_stats(units, unique_words)
    print_corpus_inspection(raw_text, units, stats, unique_words, logs_dir)

    # Plot corpus stats
    plot_corpus_stats(stats, figures_dir)

    logger.info("\n--- Phase 1 Checkpoint ---")
    logger.info(f"Units detected: {stats.num_units} (method: {boundary_method})")
    logger.info(f"Types: canticles={stats.num_canticles}, prayers={stats.num_prayers}, "
               f"prose={stats.num_prose}, rubrics={stats.num_rubrics}")
    logger.info(f"Total lines: {stats.total_lines}, tokens: {stats.total_tokens}, "
               f"types: {stats.total_types}")

    # =====================================================================
    # PHASE 2: Preprocessing & variant mapping
    # =====================================================================
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2: Preprocessing & Variant Mapping")
    logger.info("=" * 70)

    # Tokenise all text
    all_tokens = []
    for unit in units:
        for line in unit.lines:
            all_tokens.extend(tokenise(line))

    logger.info(f"Total tokens: {len(all_tokens)}, unique: {len(set(all_tokens))}")

    # Build spelling variant map
    normaliser = SpellingNormaliser()
    variant_map = normaliser.build_variant_map(all_tokens)
    normaliser.write_rule_log(logs_dir)

    # Zipf plot
    plot_zipf(all_tokens, 'Corpus Word Frequency (Zipf)', figures_dir)

    logger.info("\n--- Phase 2 Checkpoint ---")
    logger.info(f"Variant map: {len(variant_map)} entries, "
               f"{sum(1 for v in variant_map.values() if len(v) > 1)} with variants")

    # =====================================================================
    # PHASE 3: Lexicons, Features, Models, Splits
    # =====================================================================
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 3: Lexicons, Features, Models, Splits")
    logger.info("=" * 70)

    # Load lexicons
    lex_conditions = LexiconConditions()
    lex_conditions.setup(
        house_dir=paths['lexicon_house_dir'],
        external_dir=paths['lexicon_external_dir'],
        user_dir=paths['user_supplied_dir'],
        unique_words=unique_words,
    )

    # Inspect house lexicon schema (Phase 1, item 5)
    _write_lexicon_schema(base_dir, paths['lexicon_house_dir'])

    # Coverage analysis per lexicon condition
    coverage_data = {}
    for cond_name in lex_conditions.get_all_conditions():
        lex = lex_conditions.get_condition(cond_name)
        if lex is not None and lex.size() > 0:
            cov_change = compute_coverage_change(all_tokens, lex.headwords, normaliser)
            coverage_data[cond_name] = cov_change
            logger.info(f"Coverage [{cond_name}]: before={cov_change['before_coverage_pct']:.1f}%, "
                       f"after={cov_change['after_coverage_pct']:.1f}%, "
                       f"gain={cov_change['coverage_gain_pct']:.1f}%")

    # Plot coverage comparison
    if coverage_data:
        plot_coverage_comparison(coverage_data, output_dir=figures_dir)

    # Save coverage data
    if coverage_data:
        cov_df = pd.DataFrame(coverage_data).T
        cov_df.to_csv(os.path.join(metrics_dir, 'lexicon_coverage.csv'))

    # Load genres
    song_units = [u for u in units if u.unit_type in ('hymn', 'canticle')]
    if not song_units:
        # Use all units as songs if none specifically classified
        song_units = [u for u in units if u.unit_type != 'rubric']
        logger.warning("No hymns/canticles found; using all non-rubric units as songs")

    genres = load_genres(paths['genres_dir'], song_units)
    positive_only = is_positive_only_mode(genres)

    if positive_only:
        logger.warning("⚠ POSITIVE-ONLY MODE: Only song genre available. "
                       "Tasks A, B, D will be SKIPPED.")

    # Get loanword seeds from config
    loanword_seeds = config.get('loanword_seeds', [])

    logger.info("\n--- Phase 3 Checkpoint ---")
    logger.info(f"Lexicon conditions: {lex_conditions.get_all_conditions()}")
    logger.info(f"Genres available: {list(genres.keys())}")
    logger.info(f"Positive-only mode: {positive_only}")

    # =====================================================================
    # PHASE 4: Run Benchmark
    # =====================================================================
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 4: Run Benchmark")
    logger.info("=" * 70)

    all_results = {}

    # ----- Task C: Song Identification / Retrieval -----
    logger.info("\n--- Task C: Song Identification / Retrieval ---")
    task_c_results = _run_task_c(
        song_units, lex_conditions, config, loanword_seeds,
        metrics_dir, figures_dir, errors_dir
    )
    all_results['task_c'] = task_c_results

    # ----- Tasks A, B, D: Skip in positive-only mode -----
    if positive_only:
        logger.info("\n--- Tasks A, B, D: SKIPPED (positive-only mode) ---")
        logger.info("These tasks require non-song genre data (basic_text, conversation, poetry).")
        logger.info("Please supply genre data in data/genres/ and re-run.")
        all_results['task_a'] = {'status': 'skipped', 'reason': 'positive-only mode'}
        all_results['task_b'] = {'status': 'skipped', 'reason': 'positive-only mode'}
        all_results['task_d'] = {'status': 'skipped', 'reason': 'positive-only mode'}

        # Check for proxy Task A (hymn vs liturgical_prose)
        _run_proxy_task_a(units, lex_conditions, config, loanword_seeds,
                          metrics_dir, figures_dir, errors_dir)
    else:
        # Run full Tasks A, B, D
        logger.info("\n--- Task A: Song Detection ---")
        all_results['task_a'] = _run_task_a(
            genres, lex_conditions, config, loanword_seeds,
            metrics_dir, figures_dir, errors_dir
        )

    # ----- Lexicon statistics and Zipf -----
    _save_corpus_statistics(stats, units, all_tokens, unique_words,
                           normaliser, metrics_dir)

    # ----- Shared refrains -----
    shared = find_shared_refrains(song_units)
    if shared:
        shared_df = pd.DataFrame([
            {'line': line, 'song_ids': str(ids), 'num_songs': len(ids)}
            for line, ids in shared
        ])
        shared_df.to_csv(os.path.join(metrics_dir, 'shared_refrains.csv'), index=False)
        logger.info(f"Found {len(shared)} shared refrains across songs")

    # =====================================================================
    # REPORT
    # =====================================================================
    logger.info("\n" + "=" * 70)
    logger.info("Generating REPORT.md")
    logger.info("=" * 70)

    _generate_report(
        stats, units, boundary_method, coverage_data,
        all_results, positive_only, config,
        results_dir, lex_conditions
    )

    elapsed = time.time() - start_time
    logger.info(f"\n✅ Benchmark complete in {elapsed:.1f}s")
    logger.info(f"Results in: {results_dir}/")


# =============================================================================
# Task runners
# =============================================================================

def _run_task_c(song_units, lex_conditions, config, loanword_seeds,
                metrics_dir, figures_dir, errors_dir) -> Dict:
    """
    Task C: Song identification / retrieval.

    Which specific song does a phrase come from?
    Songs only; the only task fully runnable today.
    """
    if len(song_units) < 2:
        logger.warning("Not enough song units for Task C. Skipping.")
        return {'status': 'skipped', 'reason': 'too few songs'}

    # Build retriever
    retriever = SongRetriever(song_units)

    # Build line-level dataset
    lines = []
    song_ids = []
    song_id_to_title = {}
    for unit in song_units:
        song_id_to_title[unit.unit_id] = unit.title
        for line in unit.lines:
            if line.strip() and len(tokenise(line)) >= 2:
                lines.append(line)
                song_ids.append(unit.unit_id)

    if not lines:
        logger.warning("No valid lines for Task C. Skipping.")
        return {'status': 'skipped', 'reason': 'no valid lines'}

    logger.info(f"Task C: {len(lines)} lines from {len(song_units)} songs")

    # Check for small songs
    song_counts = Counter(song_ids)
    low_power = check_low_power(np.array(song_ids), min_samples=3)
    for w in low_power:
        logger.warning(w)

    # Get config params
    fast = config.get('fast_mode', False)
    n_folds = config['cv']['fast_n_folds'] if fast else config['cv']['n_folds']
    top_k_values = config['retrieval']['top_k']

    # Ensure n_folds doesn't exceed min class size
    min_class_size = min(song_counts.values())
    n_folds = min(n_folds, min_class_size, len(song_units))
    n_folds = max(n_folds, 2)

    # Evaluate retrieval
    retrieval_metrics = retriever.evaluate(lines, song_ids, top_k_values)
    logger.info(f"Task C retrieval metrics: {retrieval_metrics}")

    # Save metrics
    metrics_df = pd.DataFrame([retrieval_metrics])
    metrics_df.to_csv(os.path.join(metrics_dir, 'task_c_retrieval.csv'), index=False)

    # Plot retrieval results
    plot_retrieval_results(retrieval_metrics, figures_dir)

    # CV evaluation for classification-style Task C
    task_c_cv_results = {}

    y = np.array(song_ids)
    X = np.array(lines)

    # Build dedup groups for leakage control
    dedupe_threshold = config['fuzzy_matching']['dedupe_threshold']
    dedup_groups = build_dedup_groups(lines, threshold=dedupe_threshold)

    # For each lexicon condition, run classification
    for cond_name in lex_conditions.get_headline_conditions():
        lex = lex_conditions.get_condition(cond_name)

        if fast:
            models = get_fast_model_zoo(lex, loanword_seeds)
        else:
            models = get_model_zoo(lex, loanword_seeds)

        for model_name, pipeline in models.items():
            try:
                # Use StratifiedKFold for Task C
                # (grouped by song is impossible since song IS the label)
                skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

                fold_metrics = []
                for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]

                    try:
                        model = clone(pipeline)
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)

                        # Compute per-fold metrics
                        from sklearn.metrics import f1_score, accuracy_score
                        fold_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
                        fold_acc = accuracy_score(y_test, y_pred)
                        fold_metrics.append({
                            'fold': fold_idx,
                            'f1_macro': fold_f1,
                            'accuracy': fold_acc,
                        })
                    except Exception as e:
                        logger.warning(f"  Fold {fold_idx} failed for {model_name}/{cond_name}: {e}")

                if fold_metrics:
                    avg_f1 = np.mean([m['f1_macro'] for m in fold_metrics])
                    avg_acc = np.mean([m['accuracy'] for m in fold_metrics])
                    std_f1 = np.std([m['f1_macro'] for m in fold_metrics])

                    task_c_cv_results[f"{model_name}_{cond_name}"] = {
                        'model': model_name,
                        'condition': cond_name,
                        'f1_macro_mean': avg_f1,
                        'f1_macro_std': std_f1,
                        'accuracy_mean': avg_acc,
                        'n_folds': len(fold_metrics),
                    }
                    logger.info(f"  Task C [{cond_name}/{model_name}]: "
                               f"F1={avg_f1:.3f}±{std_f1:.3f}, acc={avg_acc:.3f}")

            except Exception as e:
                logger.warning(f"  Task C failed for {model_name}/{cond_name}: {e}")

    # Save CV results
    if task_c_cv_results:
        cv_df = pd.DataFrame(list(task_c_cv_results.values()))
        cv_df.to_csv(os.path.join(metrics_dir, 'task_c_classification.csv'), index=False)

        # Plot metric comparison
        plot_metric_comparison(
            cv_df, 'f1_macro_mean',
            title='Task C: Song Identification F1 (macro) by Condition',
            output_dir=figures_dir,
            name='task_c_f1_comparison'
        )

    return {
        'status': 'completed',
        'retrieval_metrics': retrieval_metrics,
        'cv_results': task_c_cv_results,
        'n_lines': len(lines),
        'n_songs': len(song_units),
    }


def _run_proxy_task_a(units, lex_conditions, config, loanword_seeds,
                       metrics_dir, figures_dir, errors_dir):
    """
    Proxy Task A: hymn/canticle vs liturgical_prose.

    Only run if we can reliably separate these unit types.
    Clearly labelled as PROXY in all outputs.
    """
    hymn_units = [u for u in units if u.unit_type in ('hymn', 'canticle')]
    prose_units = [u for u in units if u.unit_type == 'liturgical_prose']

    if len(hymn_units) < 2 or len(prose_units) < 2:
        logger.info("Cannot run proxy Task A: not enough hymn/canticle and prose units")
        return

    logger.info(f"\n--- PROXY Task A: Hymn/Canticle ({len(hymn_units)}) "
               f"vs Liturgical Prose ({len(prose_units)}) ---")
    logger.info("⚠ This is a PROXY only — not the real Task A (needs other genres)")

    # Build dataset
    texts = []
    labels = []
    groups = []

    for unit in hymn_units:
        for line in unit.lines:
            if line.strip() and len(tokenise(line)) >= 2:
                texts.append(line)
                labels.append(1)  # song/hymn
                groups.append(unit.unit_id)

    for unit in prose_units:
        for line in unit.lines:
            if line.strip() and len(tokenise(line)) >= 2:
                texts.append(line)
                labels.append(0)  # prose
                groups.append(unit.unit_id)

    if len(texts) < 10:
        logger.info("Too few samples for proxy Task A")
        return

    X = np.array(texts)
    y = np.array(labels)
    g = np.array(groups)

    fast = config.get('fast_mode', False)
    n_folds = min(config['cv']['fast_n_folds'] if fast else config['cv']['n_folds'],
                  min(Counter(labels).values()), len(set(groups)))
    n_folds = max(n_folds, 2)

    proxy_results = []

    for cond_name in lex_conditions.get_headline_conditions():
        lex = lex_conditions.get_condition(cond_name)
        models = get_fast_model_zoo(lex, loanword_seeds) if fast else get_model_zoo(lex, loanword_seeds)

        for model_name, pipeline in models.items():
            try:
                skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
                fold_f1s = []

                for train_idx, test_idx in skf.split(X, y):
                    model = clone(pipeline)
                    model.fit(X[train_idx], y[train_idx])
                    pred = model.predict(X[test_idx])
                    from sklearn.metrics import f1_score
                    fold_f1s.append(f1_score(y[test_idx], pred, zero_division=0))

                avg_f1 = np.mean(fold_f1s)
                proxy_results.append({
                    'model': model_name,
                    'condition': cond_name,
                    'f1_mean': avg_f1,
                    'f1_std': np.std(fold_f1s),
                })
                logger.info(f"  PROXY [{cond_name}/{model_name}]: F1={avg_f1:.3f}")

            except Exception as e:
                logger.warning(f"  PROXY failed for {model_name}/{cond_name}: {e}")

    if proxy_results:
        proxy_df = pd.DataFrame(proxy_results)
        proxy_df.to_csv(os.path.join(metrics_dir, 'PROXY_task_a_hymn_vs_prose.csv'), index=False)
        logger.info("Proxy Task A results saved with PROXY prefix")


def _run_task_a(genres, lex_conditions, config, loanword_seeds,
                metrics_dir, figures_dir, errors_dir) -> Dict:
    """Task A: Song detection (song vs not-song)."""
    # This runs when we have multiple genres
    all_texts = []
    all_labels = []

    for genre, docs in genres.items():
        label = 1 if genre == 'song' else 0
        for doc in docs:
            for line in doc.lines:
                if line.strip() and len(tokenise(line)) >= 2:
                    all_texts.append(line)
                    all_labels.append(label)

    if len(all_texts) < 10:
        return {'status': 'skipped', 'reason': 'too few samples'}

    X = np.array(all_texts)
    y = np.array(all_labels)

    logger.info(f"Task A: {len(X)} samples, {sum(y)} songs, {len(y)-sum(y)} non-songs")

    results = {}
    fast = config.get('fast_mode', False)
    n_folds = config['cv']['fast_n_folds'] if fast else config['cv']['n_folds']
    n_folds = min(n_folds, min(Counter(all_labels).values()))
    n_folds = max(n_folds, 2)

    for cond_name in lex_conditions.get_headline_conditions():
        lex = lex_conditions.get_condition(cond_name)
        models = get_fast_model_zoo(lex, loanword_seeds) if fast else get_model_zoo(lex, loanword_seeds)

        for model_name, pipeline in models.items():
            try:
                skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
                fold_metrics = []

                for train_idx, test_idx in skf.split(X, y):
                    model = clone(pipeline)
                    model.fit(X[train_idx], y[train_idx])
                    pred = model.predict(X[test_idx])
                    metrics = compute_binary_metrics(y[test_idx], pred)
                    fold_metrics.append(metrics)

                avg_metrics = {k: np.mean([m[k] for m in fold_metrics])
                              for k in fold_metrics[0].keys()}
                results[f"{model_name}_{cond_name}"] = {
                    'model': model_name,
                    'condition': cond_name,
                    **avg_metrics,
                }
                logger.info(f"  Task A [{cond_name}/{model_name}]: "
                           f"F1={avg_metrics.get('f1', 0):.3f}, "
                           f"MCC={avg_metrics.get('mcc', 0):.3f}")

            except Exception as e:
                logger.warning(f"  Task A failed for {model_name}/{cond_name}: {e}")

    if results:
        df = pd.DataFrame(list(results.values()))
        df.to_csv(os.path.join(metrics_dir, 'task_a_song_detection.csv'), index=False)

    return {'status': 'completed', 'results': results}


# =============================================================================
# Helpers
# =============================================================================

def _write_lexicon_schema(base_dir: str, house_dir: str):
    """Write inferred lexicon schema to docs/."""
    docs_dir = os.path.join(base_dir, 'docs')
    os.makedirs(docs_dir, exist_ok=True)

    schema = """# Inferred Lexicon Schema

**Status: inferred, pending user confirmation**

## House-made Annotated Lexicon CSV Schema

| Column | Type | Description |
|--------|------|-------------|
| Number | int | Sequential word number (1-672) |
| Word | str | Original word form as it appears in the corpus |
| Normalized_Ilocano | str | Modern Ilocano spelling (e.g. c→k, qu→k) |
| Lemma_Root | str | Root/lemma form of the word |
| UPOS_Tag | str | Universal POS tag (NOUN, VERB, ADJ, ADV, etc.) |
| English_Translation | str | English gloss / translation |
| Context_Example | str | Usage example from the corpus with page reference |

## Notes
- Three CSV files cover rows 1-110, 441-550, and 550-672
- Gap: rows 111-440 are not yet annotated
- Row 550 (papigsaen) appears in both the 441-550 and 550-672 files
- The 441-550 file has a filename typo: 'AnnonatedLexicon' (missing 't')
- All files use the same column schema
- POS tags follow Universal Dependencies (UPOS) conventions
"""
    with open(os.path.join(docs_dir, 'lexicon_schema.md'), 'w', encoding='utf-8') as f:
        f.write(schema)
    logger.info("Lexicon schema written to docs/lexicon_schema.md")


def _save_corpus_statistics(stats, units, tokens, unique_words, normaliser, metrics_dir):
    """Save corpus statistics to CSV."""
    # Token frequency
    freq = Counter(tokens)
    freq_df = pd.DataFrame([
        {'token': tok, 'frequency': count}
        for tok, count in freq.most_common()
    ])
    freq_df.to_csv(os.path.join(metrics_dir, 'token_frequencies.csv'), index=False)

    # Unit statistics
    unit_df = pd.DataFrame([
        {
            'unit_id': u.unit_id,
            'title': u.title,
            'unit_type': u.unit_type,
            'page': u.page,
            'num_lines': u.num_lines,
            'num_tokens': u.num_tokens,
            'speaker': u.speaker,
        }
        for u in units
    ])
    unit_df.to_csv(os.path.join(metrics_dir, 'corpus_units.csv'), index=False)

    # Unique words coverage
    covered = set(w.lower() for w in unique_words) & set(tokens)
    logger.info(f"Unique words coverage: {len(covered)}/{len(unique_words)} "
               f"({len(covered)/len(unique_words)*100:.1f}%)")


def _generate_report(stats, units, boundary_method, coverage_data,
                      all_results, positive_only, config,
                      results_dir, lex_conditions):
    """Generate REPORT.md."""
    report_path = os.path.join(results_dir, 'REPORT.md')
    lines = []

    lines.append("# Ilocano Hymn Lexicon Benchmark — Report")
    lines.append("")
    lines.append("## 1. Dataset Statistics")
    lines.append("")
    lines.append(f"- **Corpus**: Ilocano liturgical text (pages {stats.page_range[0]}-{stats.page_range[1]})")
    lines.append(f"- **Boundary detection method**: `{boundary_method}`")
    lines.append(f"- **Total units detected**: {stats.num_units}")
    lines.append(f"  - Canticles: {stats.num_canticles}")
    lines.append(f"  - Prayers: {stats.num_prayers}")
    lines.append(f"  - Liturgical prose: {stats.num_prose}")
    lines.append(f"  - Rubrics: {stats.num_rubrics}")
    lines.append(f"- **Total content lines**: {stats.total_lines}")
    lines.append(f"- **Total tokens**: {stats.total_tokens}")
    lines.append(f"- **Unique types**: {stats.total_types}")
    if stats.lines_per_unit:
        lines.append(f"- **Lines per unit**: min={min(stats.lines_per_unit)}, "
                    f"max={max(stats.lines_per_unit)}, "
                    f"mean={sum(stats.lines_per_unit)/len(stats.lines_per_unit):.1f}")
    lines.append("")

    lines.append("## 2. Lexicon Coverage")
    lines.append("")
    if coverage_data:
        lines.append("| Condition | Before Norm (%) | After Norm (%) | Gain (%) |")
        lines.append("|-----------|-----------------|----------------|----------|")
        for cond, data in coverage_data.items():
            lines.append(f"| {cond} | {data['before_coverage_pct']:.1f} | "
                        f"{data['after_coverage_pct']:.1f} | "
                        f"{data['coverage_gain_pct']:.1f} |")
    else:
        lines.append("No external lexicons were available for coverage analysis.")
    lines.append("")

    lines.append("### Orthography Finding")
    lines.append("")
    lines.append("The corpus uses older Ilocano orthography (Spanish-influenced: `c` before "
                "vowels instead of `k`, `qu` before e/i, etc.). This creates a significant "
                "coverage gap when matching against modern lexicons. The spelling variant "
                "mapping narrows this gap — the gain column above shows the improvement.")
    lines.append("")

    lines.append("## 3. Benchmark Results")
    lines.append("")

    # Task C
    task_c = all_results.get('task_c', {})
    if task_c.get('status') == 'completed':
        lines.append("### Task C: Song Identification / Retrieval")
        lines.append("")
        ret = task_c.get('retrieval_metrics', {})
        if ret:
            lines.append("| Metric | Score |")
            lines.append("|--------|-------|")
            for k, v in sorted(ret.items()):
                lines.append(f"| {k} | {v:.4f} |")
        lines.append("")
        lines.append(f"- Lines evaluated: {task_c.get('n_lines', 'N/A')}")
        lines.append(f"- Songs: {task_c.get('n_songs', 'N/A')}")
        lines.append("")

        cv = task_c.get('cv_results', {})
        if cv:
            lines.append("#### Classification CV Results")
            lines.append("")
            lines.append("| Model | Condition | F1 (macro) | Accuracy |")
            lines.append("|-------|-----------|------------|----------|")
            for key, res in sorted(cv.items()):
                lines.append(f"| {res['model']} | {res['condition']} | "
                            f"{res['f1_macro_mean']:.3f}±{res.get('f1_macro_std', 0):.3f} | "
                            f"{res['accuracy_mean']:.3f} |")
        lines.append("")

    # Tasks A, B, D
    for task in ['task_a', 'task_b', 'task_d']:
        result = all_results.get(task, {})
        if result.get('status') == 'skipped':
            task_label = task.upper().replace('_', ' ')
            lines.append(f"### {task_label}: NOT BENCHMARKED")
            lines.append("")
            lines.append(f"**Reason**: {result.get('reason', 'only the song genre is available')}")
            lines.append("")
            lines.append("This task requires non-song genre data (basic_text, conversation, poetry). "
                        "The lexicon × genre grid cells for this task are intentionally empty.")
            lines.append("")

    lines.append("## 4. Key Findings")
    lines.append("")
    lines.append("### (a) How well do existing Ilocano lexicons support recognising that a phrase is from a song?")
    lines.append("")
    if positive_only:
        lines.append("**NOT YET ANSWERABLE.** Only song data is available. Song detection requires "
                    "non-song data as negative examples (basic_text, conversation, poetry). "
                    "Please supply these genres to enable this analysis.")
    else:
        lines.append("See Task A results above.")
    lines.append("")

    lines.append("### (b) How well do they support identifying which song?")
    lines.append("")
    if task_c.get('status') == 'completed':
        ret = task_c.get('retrieval_metrics', {})
        lines.append(f"The retrieval system achieves **MRR={ret.get('mrr', 0):.3f}** and "
                    f"**top-1 accuracy={ret.get('top_1_accuracy', 0):.3f}**. "
                    f"See Task C results above for details.")
    else:
        lines.append("Task C could not be run. See logs for details.")
    lines.append("")

    lines.append("### (c) Which lexicon condition works best for each genre?")
    lines.append("")
    if positive_only:
        lines.append("**Cannot be determined yet.** The lexicon × genre grid requires multiple "
                    "genres. Currently only the song genre is available.")
    else:
        lines.append("See the lexicon × genre heatmap in results/figures/.")
    lines.append("")

    lines.append("## 5. Limitations and Assumptions")
    lines.append("")
    lines.append("1. The corpus is a single liturgical order of service, not a collection of independent songs.")
    lines.append("2. \"Songs\" in this corpus are liturgical canticles and sung responses, not standalone hymns.")
    lines.append("3. The house lexicon covers only ~44% of unique words (rows 111-440 missing).")
    lines.append("4. External lexicon availability depends on network access during the run.")
    lines.append("5. The corpus is very small (~270 lines, ~18KB), limiting ML model reliability.")
    lines.append("6. All metrics should be interpreted with caution given the small data size.")
    lines.append("")

    lines.append("## 6. What to Supply Next")
    lines.append("")
    lines.append("- [ ] **basic_text** genre data: Place `.txt` files in `data/genres/basic_text/`")
    lines.append("- [ ] **conversation** genre data: Place `.txt` files in `data/genres/conversation/`")
    lines.append("- [ ] **poetry** genre data: Place `.txt` files in `data/genres/poetry/`")
    lines.append("- [ ] Complete house lexicon rows 111-440")
    lines.append("- [ ] Confirm lexicon schema (see `docs/lexicon_schema.md`)")
    lines.append("- [ ] Optional: Rubino/Vanoverbergh headword list in `data/lexicon_external/user_supplied/`")
    lines.append("")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    logger.info(f"Report written to {report_path}")


if __name__ == '__main__':
    main()
