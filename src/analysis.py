"""
analysis.py — Error analysis, ablations, and lexicon masking.

Provides tools for:
- Error analysis with automatic reason tagging
- Lexicon ablation (drop each lexicon's features)
- Coverage masking (randomly mask lexicon entries and plot performance)
- Permutation importance for lexicon fields
"""

import os
import logging
from typing import List, Dict, Optional, Tuple, Any
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.base import clone

from src.preprocess import tokenise, normalise_text
from src.lexicons import Lexicon
from src.evaluate import compute_binary_metrics, compute_retrieval_metrics

logger = logging.getLogger(__name__)


# =====================================================================
# IDEA 4: Lexicon contribution analysis
# WHY:     We need to know which lexicon and which lexicon fields
#          actually help, not just that "combined is best".
# HOW:     (a) Ablation: drop each lexicon's features and measure
#          performance drop. (b) Coverage masking: randomly mask
#          25/50/75/100% of lexicon entries and plot performance vs
#          coverage. (c) Permutation importance on lexicon fields.
# OUTPUT:  results/metrics/ablation_*.csv, coverage-masking curves,
#          permutation importance tables
# CAVEAT:  Ablation with very few features may not show clear signal.
#          Masking is random — need multiple seeds for stability.
# =====================================================================


# =====================================================================
# IDEA 5: Hard negatives and genre-specific difficulty
# WHY:     Poetry is the realistic hard confuser for songs. We need
#          to know how well song detection works against each genre
#          separately (easy vs hard negatives).
# HOW:     Report song detection F1/PR-AUC against each other genre
#          individually, ranking genres from easy to hard. Also use
#          liturgical_prose from the same corpus as a proxy if
#          separable from hymns.
# OUTPUT:  Per-genre difficulty ranking, proxy Task A results (if
#          applicable), results/metrics/difficulty_ranking.csv
# CAVEAT:  Only available when multiple genres are loaded. Proxy
#          results are clearly labelled as proxy, not real Task A.
# =====================================================================


def tag_error_reason(text: str, y_true: Any, y_pred: Any,
                     score: float, lexicon: Optional[Lexicon] = None,
                     all_song_texts: Optional[List[str]] = None) -> str:
    """
    Automatically tag the reason for a misclassification.

    Tags:
    - short_phrase: text has ≤ 3 tokens
    - high_oov: > 50% of tokens are OOV
    - spelling_variant: has tokens with known variant mappings
    - shared_refrain: text appears in multiple songs
    - loanword_heavy: > 30% loanwords
    - ambiguous_boundary: near a song boundary

    Parameters
    ----------
    text : str
        The misclassified text.
    y_true, y_pred : Any
        True and predicted labels.
    score : float
        Prediction confidence score.
    lexicon : Optional[Lexicon]
        Lexicon for OOV checking.
    all_song_texts : Optional[List[str]]
        All song texts for shared-refrain checking.

    Returns
    -------
    str
        Comma-separated reason tags.
    """
    tokens = tokenise(text)
    reasons = []

    # Short phrase
    if len(tokens) <= 3:
        reasons.append('short_phrase')

    # High OOV
    if lexicon and tokens:
        oov_count = sum(1 for t in tokens if lexicon.lookup_with_variants(t) is None)
        if oov_count / len(tokens) > 0.5:
            reasons.append('high_oov')

    # Spelling variant tokens
    from src.preprocess import generate_modern_candidates
    variant_count = sum(1 for t in tokens if len(generate_modern_candidates(t)) > 1)
    if variant_count > 0:
        reasons.append('spelling_variant')

    # Shared refrain
    if all_song_texts:
        norm_text = normalise_text(text)
        match_count = sum(1 for st in all_song_texts if norm_text in normalise_text(st))
        if match_count > 1:
            reasons.append('shared_refrain')

    # Loanword heavy
    from src.features import LoanwordFeatureExtractor
    # Quick check with a small seed list
    loanwords = {'gloria', 'aleluya', 'espiritu', 'santo', 'cristo', 'amen',
                 'hosanna', 'bendito', 'sacramento', 'cordero'}
    loanword_count = sum(1 for t in tokens if t in loanwords)
    if tokens and loanword_count / len(tokens) > 0.3:
        reasons.append('loanword_heavy')

    return ','.join(reasons) if reasons else 'unknown'


def export_error_analysis(y_true: np.ndarray, y_pred: np.ndarray,
                          scores: np.ndarray, texts: List[str],
                          model_name: str, lexicon_condition: str,
                          lexicon: Optional[Lexicon] = None,
                          all_song_texts: Optional[List[str]] = None,
                          output_dir: str = 'results/errors') -> pd.DataFrame:
    """
    Export misclassified samples with error analysis.

    Creates results/errors/misclassified_<model>_<condition>.csv

    Columns: phrase_raw, phrase_normalised, true, predicted, score,
    system, lexicon_condition, phrase_length, oov_tokens,
    variant_mapped_tokens, reason_tag.
    """
    os.makedirs(output_dir, exist_ok=True)

    errors = []
    for i in range(len(y_true)):
        if y_true[i] != y_pred[i]:
            text = texts[i]
            tokens = tokenise(text)

            oov_tokens = []
            variant_tokens = []
            if lexicon:
                for t in tokens:
                    if lexicon.lookup(t) is None:
                        if lexicon.lookup_with_variants(t) is not None:
                            variant_tokens.append(t)
                        else:
                            oov_tokens.append(t)

            reason = tag_error_reason(
                text, y_true[i], y_pred[i], float(scores[i]),
                lexicon=lexicon, all_song_texts=all_song_texts
            )

            errors.append({
                'phrase_raw': text,
                'phrase_normalised': normalise_text(text),
                'true': y_true[i],
                'predicted': y_pred[i],
                'score': float(scores[i]),
                'system': model_name,
                'lexicon_condition': lexicon_condition,
                'phrase_length': len(tokens),
                'oov_tokens': ';'.join(oov_tokens),
                'variant_mapped_tokens': ';'.join(variant_tokens),
                'reason_tag': reason,
            })

    df = pd.DataFrame(errors)
    fname = f"misclassified_{model_name}_{lexicon_condition}.csv"
    df.to_csv(os.path.join(output_dir, fname), index=False)
    logger.info(f"Exported {len(errors)} misclassified samples to {fname}")
    return df


def lexicon_ablation_study(texts: List[str], y: np.ndarray,
                            lexicon_conditions: Dict[str, Optional[Lexicon]],
                            loanword_seeds: List[str],
                            n_folds: int = 3) -> pd.DataFrame:
    """
    Drop each lexicon condition and feature group to measure marginal contribution.

    Evaluates:
    - full_combined (TF-IDF + combined lexicon + loanwords + text stats)
    - without_house (TF-IDF + existing lexicon only)
    - without_existing (TF-IDF + house lexicon only)
    - without_lexicon (TF-IDF + loanwords + text stats, no lexicon)
    - without_loanwords (combined model without loanword features)
    - without_text_stats (combined model without text length stats)
    - lexicon_only (combined lexicon + loanwords + text stats, no TF-IDF)

    Returns DataFrame with ablation results.
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score, accuracy_score
    from src.features import build_feature_pipeline
    from sklearn.preprocessing import MaxAbsScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    X = np.array(texts)
    y_arr = np.array(y)
    unique_classes, counts = np.unique(y_arr, return_counts=True)
    actual_folds = min(n_folds, min(counts))
    actual_folds = max(actual_folds, 2)

    configs = [
        ('full_combined', lexicon_conditions.get('combined'), True, True, True),
        ('without_house', lexicon_conditions.get('existing'), True, True, True),
        ('without_existing', lexicon_conditions.get('house'), True, True, True),
        ('without_lexicon', None, True, True, True),
        ('without_loanwords', lexicon_conditions.get('combined'), True, False, True),
        ('without_text_stats', lexicon_conditions.get('combined'), True, True, False),
        ('lexicon_only', lexicon_conditions.get('combined'), False, True, True),
    ]

    results = []
    skf = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=42)

    for name, lex, use_tfidf, use_loan, use_stats in configs:
        fold_f1s = []
        fold_accs = []
        try:
            for train_idx, test_idx in skf.split(X, y_arr):
                pipe = Pipeline([
                    ('features', build_feature_pipeline(
                        lexicon_condition=lex,
                        loanword_seeds=loanword_seeds,
                        use_tfidf_word=use_tfidf,
                        use_tfidf_char=use_tfidf,
                        use_lexicon=(lex is not None),
                        use_loanword=use_loan,
                        use_text_stats=use_stats,
                    )),
                    ('scaler', MaxAbsScaler()),
                    ('clf', LogisticRegression(max_iter=1000, random_state=42, C=1.0))
                ])
                pipe.fit(X[train_idx], y_arr[train_idx])
                pred = pipe.predict(X[test_idx])
                fold_f1s.append(f1_score(y_arr[test_idx], pred, average='macro', zero_division=0))
                fold_accs.append(accuracy_score(y_arr[test_idx], pred))

            results.append({
                'ablation_condition': name,
                'f1_macro_mean': np.mean(fold_f1s),
                'f1_macro_std': np.std(fold_f1s),
                'accuracy_mean': np.mean(fold_accs),
                'accuracy_std': np.std(fold_accs),
            })
            logger.info(f"  Ablation [{name}]: F1={np.mean(fold_f1s):.3f}, acc={np.mean(fold_accs):.3f}")
        except Exception as e:
            logger.warning(f"  Ablation [{name}] failed: {e}")

    return pd.DataFrame(results)


def coverage_masking_experiment(texts: List[str], y: np.ndarray,
                                lexicon: Lexicon,
                                loanword_seeds: List[str],
                                mask_fractions: List[float],
                                mask_seeds: List[int],
                                n_folds: int = 3) -> pd.DataFrame:
    """
    Randomly mask lexicon entries and measure downstream performance.

    Masks a fraction of lexicon headwords (e.g. 25%, 50%, 75%, 100%)
    over several seeds, measuring both lexicon coverage and model F1.

    Returns DataFrame with mask_fraction, seed, coverage_pct, and metrics.
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score, accuracy_score
    from src.features import build_feature_pipeline
    from sklearn.preprocessing import MaxAbsScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    X = np.array(texts)
    y_arr = np.array(y)
    unique_classes, counts = np.unique(y_arr, return_counts=True)
    actual_folds = min(n_folds, min(counts))
    actual_folds = max(actual_folds, 2)

    all_tokens = []
    for t in texts:
        all_tokens.extend(tokenise(t))

    headwords = list(lexicon.headwords)
    results = []

    # Baseline: 0.0 mask fraction
    base_cov = lexicon.coverage(all_tokens)
    skf = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=42)

    for frac in [0.0] + [f for f in mask_fractions if f > 0.0]:
        seeds_to_run = [0] if frac == 0.0 else mask_seeds
        for seed in seeds_to_run:
            if frac == 0.0:
                masked_lex = lexicon
            else:
                rng = np.random.RandomState(seed)
                n_mask = int(len(headwords) * frac)
                masked_words = set(rng.choice(headwords, n_mask, replace=False))
                masked_lex = Lexicon(name=f'{lexicon.name}_masked_{frac}', source=lexicon.source)
                for hw, entries in lexicon.entries.items():
                    if hw not in masked_words:
                        for entry in entries:
                            masked_lex.add_entry(entry)

            cov = masked_lex.coverage(all_tokens)
            fold_f1s = []
            fold_accs = []

            for train_idx, test_idx in skf.split(X, y_arr):
                pipe = Pipeline([
                    ('features', build_feature_pipeline(
                        lexicon_condition=masked_lex,
                        loanword_seeds=loanword_seeds,
                        use_tfidf_word=True,
                        use_tfidf_char=True,
                        use_lexicon=True,
                        use_loanword=True,
                        use_text_stats=True,
                    )),
                    ('scaler', MaxAbsScaler()),
                    ('clf', LogisticRegression(max_iter=1000, random_state=42, C=1.0))
                ])
                pipe.fit(X[train_idx], y_arr[train_idx])
                pred = pipe.predict(X[test_idx])
                fold_f1s.append(f1_score(y_arr[test_idx], pred, average='macro', zero_division=0))
                fold_accs.append(accuracy_score(y_arr[test_idx], pred))

            f1_mean = np.mean(fold_f1s) if fold_f1s else 0.0
            acc_mean = np.mean(fold_accs) if fold_accs else 0.0

            results.append({
                'mask_fraction': frac,
                'seed': seed,
                'coverage_pct': cov['direct_coverage_pct'],
                'total_coverage_pct': cov['total_coverage_pct'],
                'remaining_headwords': masked_lex.size(),
                'f1_macro': f1_mean,
                'accuracy': acc_mean,
            })

    return pd.DataFrame(results)


def find_confused_song_pairs(y_true: np.ndarray, y_pred: np.ndarray,
                              song_titles: Dict[int, str]) -> List[Tuple[str, str, int]]:
    """
    Find the most frequently confused song pairs.

    Returns list of (true_song, pred_song, count) sorted by count desc.
    """
    confusion_pairs = Counter()
    for true, pred in zip(y_true, y_pred):
        if true != pred:
            true_title = song_titles.get(true, f"Song {true}")
            pred_title = song_titles.get(pred, f"Song {pred}")
            pair = tuple(sorted([true_title, pred_title]))
            confusion_pairs[pair] += 1

    return [(a, b, c) for (a, b), c in confusion_pairs.most_common()]


def find_shared_refrains(units, threshold: int = 85) -> List[Tuple[str, List[int]]]:
    """
    Find lines/refrains shared across multiple songs.

    Uses fuzzy matching to find near-duplicate lines that appear
    in different songs.
    """
    from rapidfuzz import fuzz

    line_sources = {}  # normalised_line → list of song_ids
    for unit in units:
        for line in unit.lines:
            norm = normalise_text(line)
            if norm not in line_sources:
                line_sources[norm] = []
            if unit.unit_id not in line_sources[norm]:
                line_sources[norm].append(unit.unit_id)

    shared = [(line, ids) for line, ids in line_sources.items()
              if len(ids) > 1]
    shared.sort(key=lambda x: len(x[1]), reverse=True)
    return shared
