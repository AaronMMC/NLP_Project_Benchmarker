"""
features.py — Feature extraction for the benchmark.

Builds TF-IDF features, lexicon-based features, code-switching/loanword
features, and length-based features. All assembled via scikit-learn
Pipeline/ColumnTransformer to prevent leakage.
"""

import logging
import re
from typing import List, Dict, Optional, Set, Tuple

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler

from src.lexicons import Lexicon, LexiconConditions
from src.preprocess import tokenise, normalise_text, SpellingNormaliser

logger = logging.getLogger(__name__)


# =====================================================================
# IDEA 6: Code-switching / loanword features
# WHY:     Liturgical Ilocano text is heavily interspersed with Spanish
#          and Latin loanwords (gloria, aleluya, espiritu, domini).
#          The ratio of foreign words may be a strong genre discriminator
#          since songs/canticles have higher liturgical vocabulary density.
# HOW:     Count ratio of tokens matching a documented seed list of known
#          Spanish/Latin loanwords, plus any lexicon-annotated foreign
#          origin entries. Use as a feature and test via ablation.
# OUTPUT:  Loanword ratio feature column, ablation in analysis.py
# CAVEAT:  The seed list is incomplete. Some Ilocano words look Spanish
#          but are naturalised (e.g. 'pastor' is used in both languages).
# =====================================================================


class TextFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extract text-level statistical features from documents.

    Features:
    - num_tokens: number of tokens
    - num_chars: number of characters
    - avg_token_length: average token length
    - num_unique_tokens: type count
    - type_token_ratio: lexical diversity
    - num_lines: number of lines
    - avg_line_length: average tokens per line
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """X is a list of text strings."""
        features = []
        for text in X:
            tokens = tokenise(text)
            lines = [l for l in text.split('\n') if l.strip()]
            n_tokens = len(tokens)
            n_types = len(set(tokens))

            feat = {
                'num_tokens': n_tokens,
                'num_chars': len(text),
                'avg_token_length': np.mean([len(t) for t in tokens]) if tokens else 0,
                'num_unique_tokens': n_types,
                'type_token_ratio': n_types / max(n_tokens, 1),
                'num_lines': len(lines),
                'avg_line_length': n_tokens / max(len(lines), 1),
            }
            features.append(feat)

        # Convert to array
        keys = sorted(features[0].keys()) if features else []
        result = np.array([[f[k] for k in keys] for f in features])
        return result

    def get_feature_names_out(self, input_features=None):
        return ['avg_line_length', 'avg_token_length', 'num_chars',
                'num_lines', 'num_tokens', 'num_unique_tokens', 'type_token_ratio']


class LexiconFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extract lexicon-based features for each document.

    Features per lexicon condition:
    - coverage_ratio: fraction of tokens found in lexicon
    - oov_ratio: fraction of tokens NOT in lexicon
    - variant_coverage_ratio: coverage via spelling variants
    - avg_pos_diversity: how many different POS tags found
    - has_gloss_ratio: fraction of found tokens with glosses
    """

    def __init__(self, lexicon: Optional[Lexicon] = None):
        self.lexicon = lexicon

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """X is a list of text strings."""
        if self.lexicon is None:
            # No lexicon: return zeros
            return np.zeros((len(X), 5))

        features = []
        for text in X:
            tokens = tokenise(text)
            if not tokens:
                features.append([0.0] * 5)
                continue

            direct_found = 0
            variant_found = 0
            pos_tags = set()
            gloss_count = 0

            for tok in tokens:
                entries = self.lexicon.lookup(tok)
                if entries:
                    direct_found += 1
                    for e in entries:
                        if e.pos:
                            pos_tags.add(e.pos)
                        if e.gloss:
                            gloss_count += 1
                else:
                    entries = self.lexicon.lookup_with_variants(tok)
                    if entries:
                        variant_found += 1
                        for e in entries:
                            if e.pos:
                                pos_tags.add(e.pos)
                            if e.gloss:
                                gloss_count += 1

            n = len(tokens)
            total_found = direct_found + variant_found
            feat = [
                total_found / n,           # coverage_ratio
                1.0 - total_found / n,      # oov_ratio
                variant_found / n,          # variant_coverage_ratio
                len(pos_tags) / max(n, 1),  # avg_pos_diversity
                gloss_count / max(total_found, 1),  # has_gloss_ratio
            ]
            features.append(feat)

        return np.array(features)

    def get_feature_names_out(self, input_features=None):
        return ['coverage_ratio', 'oov_ratio', 'variant_coverage_ratio',
                'avg_pos_diversity', 'has_gloss_ratio']


class LoanwordFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extract code-switching / loanword features.

    Computes the ratio of Spanish/Latin loanwords in each document
    based on a documented seed list.
    """

    def __init__(self, loanword_seeds: Optional[List[str]] = None):
        self.loanword_seeds = loanword_seeds

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """X is a list of text strings."""
        seeds_set = set(w.lower() for w in (self.loanword_seeds or []))
        features = []
        for text in X:
            tokens = tokenise(text)
            if not tokens:
                features.append([0.0, 0])
                continue
            loanword_count = sum(1 for t in tokens if t in seeds_set)
            features.append([
                loanword_count / len(tokens),  # loanword_ratio
                loanword_count,                # loanword_count
            ])
        return np.array(features)

    def get_feature_names_out(self, input_features=None):
        return ['loanword_ratio', 'loanword_count']


# =====================================================================
# IDEA 8: Length analysis
# WHY:     Performance may vary dramatically by phrase length (single
#          word vs full stanza). Controlled-length evaluation reveals
#          which systems degrade on short inputs.
# HOW:     Generate evaluation phrases at controlled lengths (1 word,
#          3-5 words, full line, full stanza) from held-out test text
#          only. Add length as a feature and as a stratification axis.
# OUTPUT:  Performance vs phrase length plots, length features
# CAVEAT:  Short phrases have very little signal; expect high variance.
# =====================================================================


def build_feature_pipeline(lexicon_condition: Optional[Lexicon],
                           loanword_seeds: List[str],
                           use_tfidf_word: bool = True,
                           use_tfidf_char: bool = True,
                           use_lexicon: bool = True,
                           use_loanword: bool = True,
                           use_text_stats: bool = True) -> FeatureUnion:
    """
    Build a feature extraction pipeline.

    Combines TF-IDF (word and char n-grams), lexicon features,
    loanword features, and text statistics via FeatureUnion.

    All fitted inside the training fold to prevent leakage.

    Parameters
    ----------
    lexicon_condition : Optional[Lexicon]
        The lexicon to use, or None for 'no lexicon' condition.
    loanword_seeds : List[str]
        List of known Spanish/Latin loanwords.
    use_tfidf_word, use_tfidf_char, use_lexicon, use_loanword, use_text_stats : bool
        Flags to enable/disable feature groups.

    Returns
    -------
    FeatureUnion
        A sklearn FeatureUnion ready for Pipeline use.
    """
    transformers = []

    if use_tfidf_word:
        transformers.append(('tfidf_word', TfidfVectorizer(
            analyzer='word',
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True,
            min_df=1,
        )))

    if use_tfidf_char:
        transformers.append(('tfidf_char', TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(2, 5),
            max_features=5000,
            sublinear_tf=True,
            min_df=1,
        )))

    if use_lexicon:
        transformers.append(('lexicon', LexiconFeatureExtractor(
            lexicon=lexicon_condition
        )))

    if use_loanword:
        transformers.append(('loanword', LoanwordFeatureExtractor(
            loanword_seeds=loanword_seeds
        )))

    if use_text_stats:
        transformers.append(('text_stats', TextFeatureExtractor()))

    return FeatureUnion(transformers)
