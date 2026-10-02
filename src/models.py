"""
models.py — Model zoo and baselines for the benchmark.

Systems compared (identical splits for all):
- Reference baselines: majority-class, stratified-random
- Lexicon-only: coverage score thresholded; classifier on lexicon features
- Statistical: TF-IDF with LogReg, LinearSVM, NB, RF
- Hybrid: TF-IDF + lexicon features
"""

import logging
from typing import List, Dict, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler

from src.features import (
    build_feature_pipeline, LexiconFeatureExtractor,
    LoanwordFeatureExtractor, TextFeatureExtractor
)
from src.lexicons import Lexicon

logger = logging.getLogger(__name__)


# =============================================================================
# Model definitions
# =============================================================================

def get_model_zoo(lexicon: Optional[Lexicon],
                  loanword_seeds: List[str],
                  task: str = 'classification') -> Dict[str, Pipeline]:
    """
    Get the full model zoo for benchmarking.

    Parameters
    ----------
    lexicon : Optional[Lexicon]
        Lexicon for this condition (None for 'none' condition).
    loanword_seeds : List[str]
        Spanish/Latin loanword seed list.
    task : str
        'classification' for Tasks A/B, 'identification' for Task C.

    Returns
    -------
    Dict[str, Pipeline]
        Named pipelines ready for cross-validation.
    """
    models = {}

    # --- Reference baselines ---
    models['majority'] = Pipeline([
        ('clf', DummyClassifier(strategy='most_frequent'))
    ])

    models['stratified_random'] = Pipeline([
        ('clf', DummyClassifier(strategy='stratified', random_state=42))
    ])

    # --- Statistical: TF-IDF models ---
    # These use word + char TF-IDF but NO lexicon features
    use_lex = lexicon is not None

    # Logistic Regression with TF-IDF
    models['tfidf_logreg'] = Pipeline([
        ('features', build_feature_pipeline(
            lexicon_condition=None,  # No lexicon for pure TF-IDF
            loanword_seeds=loanword_seeds,
            use_tfidf_word=True, use_tfidf_char=True,
            use_lexicon=False, use_loanword=True, use_text_stats=True
        )),
        ('scaler', MaxAbsScaler()),
        ('clf', LogisticRegression(max_iter=1000, random_state=42, C=1.0))
    ])

    # Linear SVM (calibrated for probability outputs)
    models['tfidf_svm'] = Pipeline([
        ('features', build_feature_pipeline(
            lexicon_condition=None,
            loanword_seeds=loanword_seeds,
            use_tfidf_word=True, use_tfidf_char=True,
            use_lexicon=False, use_loanword=True, use_text_stats=True
        )),
        ('scaler', MaxAbsScaler()),
        ('clf', CalibratedClassifierCV(
            LinearSVC(max_iter=2000, random_state=42),
            cv=3
        ))
    ])

    # Complement Naive Bayes (better for imbalanced text)
    models['tfidf_cnb'] = Pipeline([
        ('features', build_feature_pipeline(
            lexicon_condition=None,
            loanword_seeds=loanword_seeds,
            use_tfidf_word=True, use_tfidf_char=True,
            use_lexicon=False, use_loanword=False, use_text_stats=False
        )),
        ('scaler', MaxAbsScaler()),
        ('clf', ComplementNB())
    ])

    # Random Forest
    models['tfidf_rf'] = Pipeline([
        ('features', build_feature_pipeline(
            lexicon_condition=None,
            loanword_seeds=loanword_seeds,
            use_tfidf_word=True, use_tfidf_char=True,
            use_lexicon=False, use_loanword=True, use_text_stats=True
        )),
        ('scaler', MaxAbsScaler()),
        ('clf', RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        ))
    ])

    # --- Hybrid: TF-IDF + lexicon features ---
    if use_lex:
        models['hybrid_logreg'] = Pipeline([
            ('features', build_feature_pipeline(
                lexicon_condition=lexicon,
                loanword_seeds=loanword_seeds,
                use_tfidf_word=True, use_tfidf_char=True,
                use_lexicon=True, use_loanword=True, use_text_stats=True
            )),
            ('scaler', MaxAbsScaler()),
            ('clf', LogisticRegression(max_iter=1000, random_state=42, C=1.0))
        ])

        models['hybrid_svm'] = Pipeline([
            ('features', build_feature_pipeline(
                lexicon_condition=lexicon,
                loanword_seeds=loanword_seeds,
                use_tfidf_word=True, use_tfidf_char=True,
                use_lexicon=True, use_loanword=True, use_text_stats=True
            )),
            ('scaler', MaxAbsScaler()),
            ('clf', CalibratedClassifierCV(
                LinearSVC(max_iter=2000, random_state=42),
                cv=3
            ))
        ])

        models['hybrid_rf'] = Pipeline([
            ('features', build_feature_pipeline(
                lexicon_condition=lexicon,
                loanword_seeds=loanword_seeds,
                use_tfidf_word=True, use_tfidf_char=True,
                use_lexicon=True, use_loanword=True, use_text_stats=True
            )),
            ('scaler', MaxAbsScaler()),
            ('clf', RandomForestClassifier(
                n_estimators=100, random_state=42, n_jobs=-1
            ))
        ])

        # Lexicon-only: just lexicon features + loanword, no TF-IDF
        models['lexicon_only_logreg'] = Pipeline([
            ('features', build_feature_pipeline(
                lexicon_condition=lexicon,
                loanword_seeds=loanword_seeds,
                use_tfidf_word=False, use_tfidf_char=False,
                use_lexicon=True, use_loanword=True, use_text_stats=True
            )),
            ('scaler', MaxAbsScaler()),
            ('clf', LogisticRegression(max_iter=1000, random_state=42))
        ])

    return models


def get_fast_model_zoo(lexicon: Optional[Lexicon],
                       loanword_seeds: List[str]) -> Dict[str, Pipeline]:
    """
    Get a reduced model zoo for fast/quick checks.

    Only includes LogReg and majority baseline.
    """
    models = {}

    models['majority'] = Pipeline([
        ('clf', DummyClassifier(strategy='most_frequent'))
    ])

    models['tfidf_logreg'] = Pipeline([
        ('features', build_feature_pipeline(
            lexicon_condition=None,
            loanword_seeds=loanword_seeds,
            use_tfidf_word=True, use_tfidf_char=True,
            use_lexicon=False, use_loanword=True, use_text_stats=True
        )),
        ('scaler', MaxAbsScaler()),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ])

    if lexicon is not None:
        models['hybrid_logreg'] = Pipeline([
            ('features', build_feature_pipeline(
                lexicon_condition=lexicon,
                loanword_seeds=loanword_seeds,
                use_tfidf_word=True, use_tfidf_char=True,
                use_lexicon=True, use_loanword=True, use_text_stats=True
            )),
            ('scaler', MaxAbsScaler()),
            ('clf', LogisticRegression(max_iter=1000, random_state=42))
        ])

    return models
