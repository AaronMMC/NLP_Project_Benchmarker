"""
retrieval.py — Top-k song lookup tool (Task C).

Provides search(phrase, k=5) returning top-k songs with scores,
using char n-gram TF-IDF + cosine nearest neighbours, plus a
BM25-style word variant.

Includes CLI: python -m src.retrieval "phrase here"
"""

import os
import sys
import logging
import argparse
from typing import List, Tuple, Optional, Dict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.data_loader import LiturgicalUnit
from src.preprocess import normalise_text, tokenise

logger = logging.getLogger(__name__)


# =====================================================================
# IDEA 3: Retrieval mode
# WHY:     Song identification (Task C) is a natural retrieval problem.
#          Given a phrase, which song does it come from?
# HOW:     Char n-gram TF-IDF + cosine similarity for retrieval, plus
#          a BM25-style word-level variant. Returns top-k songs with
#          scores. CLI for interactive use.
# OUTPUT:  Top-1/3/5 accuracy, MRR in results/metrics/, CLI tool
# CAVEAT:  Small corpus means TF-IDF may overfit to rare n-grams.
#          BM25 parameters are not tuned (k1=1.5, b=0.75 defaults).
# =====================================================================


class SongRetriever:
    """
    Retrieval system for identifying which song a phrase comes from.

    Uses two methods:
    1. Char n-gram TF-IDF + cosine similarity
    2. BM25-style word-level scoring
    """

    def __init__(self, units: List[LiturgicalUnit]):
        """
        Initialize the retriever with song units.

        Parameters
        ----------
        units : List[LiturgicalUnit]
            The song/canticle units from the corpus.
        """
        self.units = units
        self.song_texts = [normalise_text(u.text) for u in units]
        self.song_ids = [u.unit_id for u in units]
        self.song_titles = [u.title for u in units]

        # Build char n-gram TF-IDF index
        self.char_vectoriser = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(2, 5),
            max_features=10000,
            sublinear_tf=True,
        )
        self.char_matrix = self.char_vectoriser.fit_transform(self.song_texts)

        # Build word TF-IDF index
        self.word_vectoriser = TfidfVectorizer(
            analyzer='word',
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True,
        )
        self.word_matrix = self.word_vectoriser.fit_transform(self.song_texts)

        # BM25 parameters
        self.k1 = 1.5
        self.b = 0.75
        self._build_bm25_index()

        logger.info(f"SongRetriever initialized with {len(units)} songs")

    def _build_bm25_index(self):
        """Build BM25 index from song texts."""
        self.song_tokens = [tokenise(text) for text in self.song_texts]
        self.avg_dl = np.mean([len(tokens) for tokens in self.song_tokens])

        # Document frequencies
        self.df = {}
        for tokens in self.song_tokens:
            seen = set()
            for tok in tokens:
                if tok not in seen:
                    self.df[tok] = self.df.get(tok, 0) + 1
                    seen.add(tok)

        self.n_docs = len(self.song_tokens)

    def _bm25_score(self, query_tokens: List[str], doc_idx: int) -> float:
        """Compute BM25 score for a query against a document."""
        doc_tokens = self.song_tokens[doc_idx]
        dl = len(doc_tokens)
        tf_dict = {}
        for tok in doc_tokens:
            tf_dict[tok] = tf_dict.get(tok, 0) + 1

        score = 0.0
        for qt in query_tokens:
            if qt not in self.df:
                continue
            idf = np.log((self.n_docs - self.df[qt] + 0.5) / (self.df[qt] + 0.5) + 1)
            tf = tf_dict.get(qt, 0)
            tf_norm = (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avg_dl))
            score += idf * tf_norm

        return score

    def search(self, phrase: str, k: int = 5,
               method: str = 'combined') -> List[Tuple[int, str, float]]:
        """
        Search for the top-k songs matching a phrase.

        Parameters
        ----------
        phrase : str
            The query phrase.
        k : int
            Number of results to return.
        method : str
            'char_tfidf', 'word_tfidf', 'bm25', or 'combined'.

        Returns
        -------
        List[Tuple[int, str, float]]
            List of (song_id, song_title, score) tuples, sorted by score desc.
        """
        norm_phrase = normalise_text(phrase)

        scores = np.zeros(len(self.units))

        if method in ('char_tfidf', 'combined'):
            query_vec = self.char_vectoriser.transform([norm_phrase])
            char_scores = cosine_similarity(query_vec, self.char_matrix).flatten()
            scores += char_scores

        if method in ('word_tfidf', 'combined'):
            query_vec = self.word_vectoriser.transform([norm_phrase])
            word_scores = cosine_similarity(query_vec, self.word_matrix).flatten()
            scores += word_scores

        if method in ('bm25', 'combined'):
            query_tokens = tokenise(norm_phrase)
            bm25_scores = np.array([self._bm25_score(query_tokens, i)
                                    for i in range(len(self.units))])
            # Normalise BM25 scores to [0, 1] range
            if bm25_scores.max() > 0:
                bm25_scores = bm25_scores / bm25_scores.max()
            scores += bm25_scores

        # Rank by score
        top_indices = np.argsort(-scores)[:k]
        results = []
        for idx in top_indices:
            results.append((
                self.song_ids[idx],
                self.song_titles[idx],
                float(scores[idx])
            ))

        return results

    def search_line_level(self, lines: List[str], k: int = 5) -> List[List[Tuple[int, str, float]]]:
        """
        Search for each line independently, returning top-k per line.

        Used for evaluation where each line has a known source song.
        """
        return [self.search(line, k=k) for line in lines]

    def evaluate(self, test_lines: List[str], test_song_ids: List[int],
                 top_k_values: List[int] = [1, 3, 5]) -> Dict[str, float]:
        """
        Evaluate retrieval performance.

        Parameters
        ----------
        test_lines : List[str]
            Query lines.
        test_song_ids : List[int]
            True song ID for each line.
        top_k_values : List[int]
            Values of k for top-k accuracy.

        Returns
        -------
        Dict[str, float]
            Retrieval metrics.
        """
        rankings = []
        for line in test_lines:
            results = self.search(line, k=max(top_k_values))
            ranked_ids = [r[0] for r in results]
            rankings.append(ranked_ids)

        from src.evaluate import compute_retrieval_metrics
        return compute_retrieval_metrics(
            np.array(test_song_ids), rankings, top_k_values
        )


def main():
    """CLI entry point for retrieval."""
    parser = argparse.ArgumentParser(
        description='Search for Ilocano songs by phrase'
    )
    parser.add_argument('phrase', type=str, help='Search phrase')
    parser.add_argument('-k', '--top-k', type=int, default=5,
                       help='Number of results (default: 5)')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Config file path')
    args = parser.parse_args()

    import yaml
    from src.data_loader import load_raw_corpus, detect_boundaries

    # Load config
    config_path = args.config
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    # Load corpus
    corpus_path = config.get('paths', {}).get('raw_corpus', 'data/raw/corpus_ilocano_liturgy.txt')
    raw_text = load_raw_corpus(corpus_path)
    units, method = detect_boundaries(raw_text, config)

    # Filter to song-type units
    song_units = [u for u in units if u.unit_type in ('hymn', 'canticle')]
    if not song_units:
        song_units = units  # Use all if no songs detected

    # Build retriever and search
    retriever = SongRetriever(song_units)
    results = retriever.search(args.phrase, k=args.top_k)

    print(f"\nSearch results for: '{args.phrase}'")
    print("=" * 60)
    for i, (song_id, title, score) in enumerate(results, 1):
        print(f"  {i}. [{score:.4f}] Song #{song_id}: {title}")
    print()


if __name__ == '__main__':
    main()
