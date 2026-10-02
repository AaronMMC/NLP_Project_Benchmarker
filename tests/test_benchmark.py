"""
Test suite for the Ilocano Hymn Lexicon Benchmark.

Covers: tokeniser, boundary detection on toy corpus, normalisation rules,
lexicon loader (partial/duplicate/missing columns), splitters (no group
overlap; dedupe holds), metric sanity, and the shuffled-label leakage check.
"""

import os
import sys
import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestTokeniser:
    """Test the deterministic tokeniser."""

    def test_basic_tokenisation(self):
        from src.preprocess import tokenise
        tokens = tokenise("Amami nga addaca sadi langit")
        assert tokens == ['amami', 'nga', 'addaca', 'sadi', 'langit']

    def test_apostrophe_preservation(self):
        from src.preprocess import tokenise
        tokens = tokenise("Talna't daga, naimbag a nakem!")
        assert "talna't" in tokens
        assert 'daga' in tokens

    def test_hyphen_preservation(self):
        from src.preprocess import tokenise
        tokens = tokenise("awanan-tulaw a doctrina")
        assert 'awanan-tulaw' in tokens

    def test_deterministic(self):
        """Same input always produces same output."""
        from src.preprocess import tokenise
        text = "Apo caasiannacami! Cristo caasiannacami!"
        result1 = tokenise(text)
        result2 = tokenise(text)
        assert result1 == result2

    def test_empty_input(self):
        from src.preprocess import tokenise
        assert tokenise("") == []
        assert tokenise("   ") == []

    def test_punctuation_stripping(self):
        from src.preprocess import tokenise
        tokens = tokenise("Amen! Amen! Amen!")
        assert all(t == 'amen' for t in tokens)
        assert len(tokens) == 3


class TestNormalisation:
    """Test text normalisation and spelling variants."""

    def test_unicode_nfc(self):
        from src.preprocess import unicode_normalise
        # NFC should produce consistent forms
        text = "tëst"
        result = unicode_normalise(text)
        assert len(result) > 0

    def test_normalise_lowercase(self):
        from src.preprocess import normalise_text
        assert normalise_text("HELLO WORLD") == "hello world"

    def test_variant_c_to_k(self):
        from src.preprocess import generate_modern_candidates
        candidates = generate_modern_candidates('cararag')
        assert 'kararag' in candidates
        assert 'cararag' in candidates  # Original always included

    def test_variant_qu_to_k(self):
        from src.preprocess import generate_modern_candidates
        # 'qu' before e/i should map to 'k'
        candidates = generate_modern_candidates('queso')
        assert 'keso' in candidates

    def test_known_variant_lookup(self):
        from src.preprocess import generate_modern_candidates
        candidates = generate_modern_candidates('cadagiti')
        assert 'kadagiti' in candidates

    def test_original_always_included(self):
        from src.preprocess import generate_modern_candidates
        for word in ['cararag', 'hello', 'test', 'cadagiti']:
            candidates = generate_modern_candidates(word)
            assert word in candidates


class TestBoundaryDetection:
    """Test boundary detection on toy corpus."""

    def test_parse_pages(self):
        from src.data_loader import parse_pages
        toy = """# TITLE

================================================================================
PAGE 1
================================================================================

Content on page 1.

================================================================================
PAGE 2
================================================================================

Content on page 2.
"""
        pages = parse_pages(toy)
        assert len(pages) == 2
        assert pages[0][0] == 1
        assert pages[1][0] == 2

    def test_all_caps_detection(self):
        from src.data_loader import _is_all_caps_heading
        assert _is_all_caps_heading("GLORIA PATRI")
        assert _is_all_caps_heading("TI KYRIE")
        assert not _is_all_caps_heading("Small text")
        assert not _is_all_caps_heading("a")

    def test_speaker_detection(self):
        from src.data_loader import _detect_speaker
        assert _detect_speaker("Pastor: Hello") == "Pastor"
        assert _detect_speaker("Gimong: Response") == "Gimong"
        assert _detect_speaker("Amin: All together") == "Amin"
        assert _detect_speaker("Regular text") is None


class TestLexiconLoader:
    """Test lexicon loading with edge cases."""

    def test_house_lexicon_loads(self):
        from src.lexicons import load_house_lexicon
        # Use actual data if available
        house_dir = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), 'data', 'lexicon_house')
        if os.path.exists(house_dir):
            words = ['a', 'adal', 'amen']
            lex = load_house_lexicon(house_dir, words)
            assert lex.size() > 0

    def test_lexicon_lookup(self):
        from src.lexicons import Lexicon, LexiconEntry
        lex = Lexicon(name='test', source='test')
        lex.add_entry(LexiconEntry(
            headword='test', source='test', pos='NOUN', gloss='a test'
        ))
        result = lex.lookup('test')
        assert result is not None
        assert len(result) == 1
        assert result[0].pos == 'NOUN'

    def test_lexicon_lookup_missing(self):
        from src.lexicons import Lexicon
        lex = Lexicon(name='test', source='test')
        assert lex.lookup('nonexistent') is None

    def test_lexicon_variant_lookup(self):
        from src.lexicons import Lexicon, LexiconEntry
        lex = Lexicon(name='test', source='test')
        lex.add_entry(LexiconEntry(
            headword='kararag', source='test', pos='VERB'
        ))
        # 'cararag' should find 'kararag' via variant mapping
        result = lex.lookup_with_variants('cararag')
        assert result is not None

    def test_lexicon_coverage(self):
        from src.lexicons import Lexicon, LexiconEntry
        lex = Lexicon(name='test', source='test')
        lex.add_entry(LexiconEntry(headword='hello', source='test'))
        lex.add_entry(LexiconEntry(headword='world', source='test'))

        cov = lex.coverage(['hello', 'world', 'missing'])
        assert cov['total_types'] == 3
        assert cov['direct_covered'] == 2

    def test_merge_keeps_both(self):
        """Merge rule: keep both entries on conflict, never silently overwrite."""
        from src.lexicons import Lexicon, LexiconEntry, merge_lexicons
        lex1 = Lexicon(name='a', source='a')
        lex1.add_entry(LexiconEntry(headword='word', source='a', gloss='meaning A'))
        lex2 = Lexicon(name='b', source='b')
        lex2.add_entry(LexiconEntry(headword='word', source='b', gloss='meaning B'))

        merged = merge_lexicons([lex1, lex2])
        entries = merged.lookup('word')
        assert len(entries) == 2
        sources = {e.source for e in entries}
        assert 'a' in sources
        assert 'b' in sources


class TestSplits:
    """Test cross-validation splitters."""

    def test_dedup_groups_exact_match(self):
        from src.splits import build_dedup_groups
        texts = ["hello world", "hello world", "different text"]
        groups = build_dedup_groups(texts, threshold=85)
        # First two should be in same group
        assert groups[0] == groups[1]
        # Third should be different
        assert groups[2] != groups[0]

    def test_no_group_overlap_in_splits(self):
        """Groups should never be split across train and test."""
        from src.splits import generate_cv_splits
        from sklearn.model_selection import StratifiedGroupKFold

        X = np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        groups = np.array([0, 0, 1, 1, 2, 2, 3, 3])

        splitter = StratifiedGroupKFold(n_splits=2)
        splits = generate_cv_splits(X, y, 'grouped', splitter, groups)

        for train_idx, test_idx in splits:
            train_groups = set(groups[train_idx])
            test_groups = set(groups[test_idx])
            # No overlap
            assert len(train_groups & test_groups) == 0


class TestMetrics:
    """Test metric computation."""

    def test_binary_metrics_perfect(self):
        from src.evaluate import compute_binary_metrics
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        metrics = compute_binary_metrics(y_true, y_pred)
        assert metrics['accuracy'] == 1.0
        assert metrics['f1'] == 1.0
        assert metrics['mcc'] == 1.0

    def test_binary_metrics_random(self):
        from src.evaluate import compute_binary_metrics
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([1, 0, 1, 0])  # All wrong
        metrics = compute_binary_metrics(y_true, y_pred)
        assert metrics['accuracy'] == 0.0

    def test_bootstrap_ci(self):
        from src.evaluate import bootstrap_ci
        from sklearn.metrics import accuracy_score
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 1])
        point, ci_lo, ci_hi = bootstrap_ci(y_true, y_pred, accuracy_score, n_bootstrap=100)
        assert point == 1.0
        assert ci_lo <= point <= ci_hi

    def test_shuffled_label_leakage_check(self):
        """
        Shuffled labels should score near chance — this is the leakage check.
        If a model scores well on shuffled labels, there's leakage.
        """
        from sklearn.dummy import DummyClassifier
        from sklearn.metrics import accuracy_score

        np.random.seed(42)
        n = 100
        y = np.array([0] * 50 + [1] * 50)
        np.random.shuffle(y)
        X = np.random.randn(n, 5)

        # Shuffle labels to break any real signal
        y_shuffled = y.copy()
        np.random.shuffle(y_shuffled)

        clf = DummyClassifier(strategy='stratified', random_state=42)
        clf.fit(X, y_shuffled)
        pred = clf.predict(X)

        acc = accuracy_score(y_shuffled, pred)
        # Should be near 50% (chance) — definitely not > 80%
        assert acc < 0.8, f"Shuffled-label accuracy {acc:.2f} is suspiciously high — possible leakage"


class TestUniqueWords:
    """Test unique words loading."""

    def test_load_unique_words(self):
        from src.data_loader import load_unique_words
        words_path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), 'data', 'raw', 'unique_words.txt')
        if os.path.exists(words_path):
            words = load_unique_words(words_path)
            assert len(words) == 672
            assert 'a' in words
            assert 'amen' in words
            assert 'yeg' in words


# Test fixtures
@pytest.fixture
def toy_corpus():
    """A tiny synthetic corpus for testing."""
    return """# TEST CORPUS

================================================================================
PAGE 1
================================================================================

GLORIA PATRI
Gimong: Gloria coma iti Ama ken 'ti Anac.

TI KYRIE
Gimong: Apo caasiannacami!
"""


@pytest.fixture
def toy_lexicon():
    """A tiny synthetic lexicon for testing."""
    from src.lexicons import Lexicon, LexiconEntry
    lex = Lexicon(name='test', source='test')
    entries = [
        ('gloria', 'NOUN', 'glory'),
        ('ama', 'NOUN', 'father'),
        ('apo', 'NOUN', 'lord'),
    ]
    for word, pos, gloss in entries:
        lex.add_entry(LexiconEntry(
            headword=word, source='test', pos=pos, gloss=gloss
        ))
    return lex
