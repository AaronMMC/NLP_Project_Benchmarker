"""
lexicons.py — Lexicon interface, house + external lexicon loaders.

Provides a unified Lexicon interface for all lexicon types (house-made,
Wiktionary, PanLex, user-supplied). Each lexicon supports lookup, coverage,
and feature extraction.

Merge rule for 'combined' condition: union of headwords; on conflicts,
keep both entries and tag each with its source — never silently overwrite.
"""

import os
import re
import glob
import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple, Any
from collections import defaultdict

import pandas as pd
from rapidfuzz import fuzz, process

from src.preprocess import generate_modern_candidates, generate_old_candidates

logger = logging.getLogger(__name__)


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class LexiconEntry:
    """A single entry in any lexicon."""
    headword: str
    source: str  # 'house', 'wiktionary', 'panlex', 'user_supplied'
    pos: Optional[str] = None
    gloss: Optional[str] = None
    lemma: Optional[str] = None
    normalised_form: Optional[str] = None
    context: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Lexicon interface
# =============================================================================

class Lexicon:
    """
    Unified lexicon interface used by all lexicon types.

    Supports:
    - lookup(word) → LexiconEntry | None
    - coverage(tokens) → coverage stats
    - features(token) → feature dict for ML
    """

    def __init__(self, name: str, source: str):
        self.name = name
        self.source = source
        self.entries: Dict[str, List[LexiconEntry]] = defaultdict(list)
        self._headwords: Set[str] = set()

    def add_entry(self, entry: LexiconEntry):
        """Add an entry to the lexicon."""
        hw = entry.headword.lower().strip()
        self.entries[hw].append(entry)
        self._headwords.add(hw)

    @property
    def headwords(self) -> Set[str]:
        """Return set of all headwords."""
        return self._headwords

    def size(self) -> int:
        """Return number of unique headwords."""
        return len(self._headwords)

    def lookup(self, word: str) -> Optional[List[LexiconEntry]]:
        """
        Look up a word in the lexicon.

        Parameters
        ----------
        word : str
            Word to look up (case-insensitive).

        Returns
        -------
        Optional[List[LexiconEntry]]
            List of matching entries, or None if not found.
        """
        w = word.lower().strip()
        if w in self.entries:
            return self.entries[w]
        return None

    def lookup_with_variants(self, word: str) -> Optional[List[LexiconEntry]]:
        """
        Look up a word, trying spelling variants if direct lookup fails.

        Generates modern and old spelling candidates and checks each.
        """
        # Direct lookup first
        result = self.lookup(word)
        if result:
            return result

        # Try modern candidates (corpus word → modern lexicon)
        for candidate in generate_modern_candidates(word.lower()):
            result = self.lookup(candidate)
            if result:
                return result

        # Try old candidates (modern word → old lexicon)
        for candidate in generate_old_candidates(word.lower()):
            result = self.lookup(candidate)
            if result:
                return result

        return None

    def coverage(self, tokens: List[str]) -> Dict:
        """
        Compute coverage statistics for a list of tokens.

        Parameters
        ----------
        tokens : List[str]
            Tokens to check coverage for.

        Returns
        -------
        Dict
            Coverage statistics including counts and percentages.
        """
        unique_tokens = set(t.lower() for t in tokens)
        covered = unique_tokens & self._headwords
        oov = unique_tokens - self._headwords

        # Also check with variants
        variant_covered = set()
        for tok in oov:
            if self.lookup_with_variants(tok) is not None:
                variant_covered.add(tok)

        return {
            'total_types': len(unique_tokens),
            'direct_covered': len(covered),
            'variant_covered': len(variant_covered),
            'total_covered': len(covered) + len(variant_covered),
            'oov': len(oov) - len(variant_covered),
            'direct_coverage_pct': len(covered) / max(len(unique_tokens), 1) * 100,
            'total_coverage_pct': (len(covered) + len(variant_covered)) / max(len(unique_tokens), 1) * 100,
        }

    def features(self, token: str) -> Dict[str, float]:
        """
        Extract feature values for a token from this lexicon.

        Features:
        - in_lexicon: 1.0 if found, 0.0 otherwise
        - in_lexicon_variant: 1.0 if found via variant, 0.0 otherwise
        - has_pos: 1.0 if POS tag available
        - has_gloss: 1.0 if gloss available
        - pos_* one-hot features for each POS tag
        """
        feats = {
            'in_lexicon': 0.0,
            'in_lexicon_variant': 0.0,
            'has_pos': 0.0,
            'has_gloss': 0.0,
        }

        entries = self.lookup(token)
        if entries:
            feats['in_lexicon'] = 1.0
        else:
            entries = self.lookup_with_variants(token)
            if entries:
                feats['in_lexicon_variant'] = 1.0

        if entries:
            for entry in entries:
                if entry.pos:
                    feats['has_pos'] = 1.0
                    feats[f'pos_{entry.pos.lower()}'] = 1.0
                if entry.gloss:
                    feats['has_gloss'] = 1.0

        return feats


# =============================================================================
# House-made lexicon loader
# =============================================================================

def load_house_lexicon(house_dir: str, unique_words: List[str]) -> Lexicon:
    """
    Load the house-made annotated lexicon from CSV files.

    Per AGENTS.md:
    - Glob all *.csv files in the directory (sorted)
    - Tolerate differing column order, extra/missing columns
    - Detect and report duplicate headwords across files
    - Report coverage against unique_words.txt

    Parameters
    ----------
    house_dir : str
        Path to the lexicon_house directory.
    unique_words : List[str]
        List of unique words from the corpus.

    Returns
    -------
    Lexicon
        The loaded house-made lexicon.
    """
    lexicon = Lexicon(name='house', source='house')

    csv_files = sorted(glob.glob(os.path.join(house_dir, '*.csv')))
    if not csv_files:
        logger.warning(f"No CSV files found in {house_dir}")
        return lexicon

    logger.info(f"Loading house lexicon from {len(csv_files)} CSV files: "
               f"{[os.path.basename(f) for f in csv_files]}")

    all_dfs = []
    seen_headwords = {}  # word → filename where first seen
    duplicates = []

    for csv_file in csv_files:
        fname = os.path.basename(csv_file)
        try:
            df = pd.read_csv(csv_file, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(csv_file, encoding='latin-1')
            except Exception as e:
                logger.warning(f"Could not read {fname}: {e}")
                continue

        logger.info(f"  {fname}: {len(df)} rows, columns={list(df.columns)}")

        # Standardise column names (case-insensitive matching)
        col_map = {}
        for col in df.columns:
            cl = col.lower().strip()
            if 'number' in cl:
                col_map[col] = 'Number'
            elif cl == 'word':
                col_map[col] = 'Word'
            elif 'normalized' in cl or 'normalised' in cl:
                col_map[col] = 'Normalized_Ilocano'
            elif 'lemma' in cl or 'root' in cl:
                col_map[col] = 'Lemma_Root'
            elif 'upos' in cl or 'pos' in cl:
                col_map[col] = 'UPOS_Tag'
            elif 'english' in cl or 'translation' in cl:
                col_map[col] = 'English_Translation'
            elif 'context' in cl or 'example' in cl:
                col_map[col] = 'Context_Example'
            else:
                col_map[col] = col  # keep unknown columns

        df = df.rename(columns=col_map)

        # Check for duplicates
        if 'Word' in df.columns:
            for _, row in df.iterrows():
                word = str(row.get('Word', '')).lower().strip()
                if not word or word == 'nan':
                    continue
                if word in seen_headwords:
                    duplicates.append((word, seen_headwords[word], fname))
                    # Keep the first occurrence (documented rule)
                    logger.warning(f"  Duplicate headword '{word}': first in "
                                 f"{seen_headwords[word]}, also in {fname} — keeping first")
                else:
                    seen_headwords[word] = fname

        all_dfs.append((fname, df))

    # Combine and create entries
    processed_words = set()
    for fname, df in all_dfs:
        for _, row in df.iterrows():
            word = str(row.get('Word', '')).lower().strip()
            if not word or word == 'nan' or word in processed_words:
                # Skip duplicates (keep first per documented rule)
                if word in processed_words:
                    continue
                continue

            entry = LexiconEntry(
                headword=word,
                source='house',
                pos=str(row.get('UPOS_Tag', '')) if pd.notna(row.get('UPOS_Tag')) else None,
                gloss=str(row.get('English_Translation', '')) if pd.notna(row.get('English_Translation')) else None,
                lemma=str(row.get('Lemma_Root', '')) if pd.notna(row.get('Lemma_Root')) else None,
                normalised_form=str(row.get('Normalized_Ilocano', '')) if pd.notna(row.get('Normalized_Ilocano')) else None,
                context=str(row.get('Context_Example', '')) if pd.notna(row.get('Context_Example')) else None,
            )
            lexicon.add_entry(entry)
            processed_words.add(word)

    # Report coverage against unique_words
    unique_set = set(w.lower() for w in unique_words)
    covered = unique_set & lexicon.headwords
    uncovered = unique_set - lexicon.headwords

    # Determine missing ranges
    word_numbers = {}
    for i, w in enumerate(unique_words, 1):
        word_numbers[w.lower()] = i

    uncovered_numbers = sorted(word_numbers[w] for w in uncovered if w in word_numbers)
    missing_ranges = _find_ranges(uncovered_numbers)

    logger.info(f"House lexicon loaded: {lexicon.size()} entries")
    logger.info(f"Coverage against unique_words.txt: "
               f"{len(covered)}/{len(unique_set)} ({len(covered)/len(unique_set)*100:.1f}%)")
    logger.info(f"Uncovered: {len(uncovered)} words")
    logger.info(f"Missing ranges: {missing_ranges}")
    if duplicates:
        logger.info(f"Duplicates found across files: {len(duplicates)}")
        for word, first, second in duplicates[:5]:
            logger.info(f"  '{word}': {first} vs {second}")

    return lexicon


def _find_ranges(numbers: List[int]) -> List[str]:
    """Find contiguous ranges in a sorted list of numbers."""
    if not numbers:
        return []
    ranges = []
    start = numbers[0]
    end = numbers[0]
    for n in numbers[1:]:
        if n == end + 1:
            end = n
        else:
            if start == end:
                ranges.append(str(start))
            else:
                ranges.append(f"{start}-{end}")
            start = end = n
    if start == end:
        ranges.append(str(start))
    else:
        ranges.append(f"{start}-{end}")
    return ranges


# =============================================================================
# External lexicon loaders
# =============================================================================

def load_wiktionary_lexicon(external_dir: str) -> Optional[Lexicon]:
    """
    Load Wiktionary Ilocano entries from cached JSONL.

    Attempts to download from kaikki.org if not cached. If network
    is unavailable, prints manual download instructions and skips.

    Parameters
    ----------
    external_dir : str
        Path to lexicon_external directory.

    Returns
    -------
    Optional[Lexicon]
        The Wiktionary lexicon, or None if unavailable.
    """
    cache_path = os.path.join(external_dir, 'wiktionary_ilocano.jsonl')

    if not os.path.exists(cache_path):
        # Try to download
        logger.info("Wiktionary cache not found. Attempting download...")
        try:
            import urllib.request
            url = "https://kaikki.org/dictionary/Ilocano/kaikki.org-dictionary-Ilocano.jsonl"
            logger.info(f"Downloading from {url}...")
            urllib.request.urlretrieve(url, cache_path)
            logger.info(f"Downloaded Wiktionary Ilocano data to {cache_path}")
        except Exception as e:
            logger.warning(
                f"Could not download Wiktionary data: {e}\n"
                f"Manual download instructions:\n"
                f"  1. Go to https://kaikki.org/dictionary/Ilocano/\n"
                f"  2. Download the JSONL file\n"
                f"  3. Save as {cache_path}\n"
                f"Skipping Wiktionary lexicon."
            )
            return None

    # Parse the JSONL
    lexicon = Lexicon(name='wiktionary', source='wiktionary')
    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    word = data.get('word', '').lower().strip()
                    if not word:
                        continue

                    pos = data.get('pos', None)
                    glosses = []
                    for sense in data.get('senses', []):
                        for gloss in sense.get('glosses', []):
                            glosses.append(gloss)

                    entry = LexiconEntry(
                        headword=word,
                        source='wiktionary',
                        pos=pos,
                        gloss='; '.join(glosses) if glosses else None,
                        extra=data,
                    )
                    lexicon.add_entry(entry)
                except json.JSONDecodeError:
                    continue

        logger.info(f"Wiktionary lexicon loaded: {lexicon.size()} entries")
    except Exception as e:
        logger.warning(f"Error parsing Wiktionary data: {e}")
        return None

    return lexicon


def load_panlex_lexicon(external_dir: str) -> Optional[Lexicon]:
    """
    Load PanLex Ilocano entries.

    Attempts to query PanLex API or load from cache. If unavailable,
    prints instructions and skips.

    Parameters
    ----------
    external_dir : str
        Path to lexicon_external directory.

    Returns
    -------
    Optional[Lexicon]
        The PanLex lexicon, or None if unavailable.
    """
    cache_path = os.path.join(external_dir, 'panlex_ilocano.json')

    if not os.path.exists(cache_path):
        logger.info("PanLex cache not found. Attempting API query...")
        try:
            import urllib.request
            # PanLex API v2: query for Ilocano (ilo) expressions
            url = "https://api.panlex.org/v2/expr"
            payload = json.dumps({
                "langvar": "ilo-000",
                "limit": 10000,
                "include": "trans_langvar"
            }).encode('utf-8')
            req = urllib.request.Request(
                url,
                data=payload,
                headers={'Content-Type': 'application/json'},
                method='POST'
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode('utf-8'))

            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"Downloaded PanLex data to {cache_path}")
        except Exception as e:
            logger.warning(
                f"Could not query PanLex API: {e}\n"
                f"Manual instructions:\n"
                f"  1. Visit https://panlex.org/ and search for Ilocano\n"
                f"  2. Or use the PanLex data dump: https://panlex.org/snapshot/\n"
                f"  3. Save processed data as {cache_path}\n"
                f"Skipping PanLex lexicon."
            )
            return None

    lexicon = Lexicon(name='panlex', source='panlex')
    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        results = data.get('result', data) if isinstance(data, dict) else data
        if isinstance(results, list):
            for item in results:
                word = ''
                if isinstance(item, dict):
                    word = item.get('txt', item.get('word', '')).lower().strip()
                elif isinstance(item, str):
                    word = item.lower().strip()
                if word:
                    entry = LexiconEntry(
                        headword=word,
                        source='panlex',
                        extra=item if isinstance(item, dict) else {},
                    )
                    lexicon.add_entry(entry)

        logger.info(f"PanLex lexicon loaded: {lexicon.size()} entries")
    except Exception as e:
        logger.warning(f"Error parsing PanLex data: {e}")
        return None

    return lexicon if lexicon.size() > 0 else None


def load_user_supplied_lexicon(user_dir: str) -> Optional[Lexicon]:
    """
    Load user-supplied lexicon (e.g. Rubino or Vanoverbergh headword list).

    Only loaded if user places files in data/lexicon_external/user_supplied/.
    Does not scrape or reproduce copyrighted content.
    """
    if not os.path.exists(user_dir):
        logger.info("No user-supplied lexicon directory found. Skipping.")
        return None

    files = glob.glob(os.path.join(user_dir, '*'))
    if not files:
        logger.info("No user-supplied lexicon files found. Skipping.")
        return None

    lexicon = Lexicon(name='user_supplied', source='user_supplied')

    for fpath in files:
        fname = os.path.basename(fpath)
        logger.info(f"Loading user-supplied lexicon from {fname}")
        try:
            if fpath.endswith('.csv'):
                df = pd.read_csv(fpath, encoding='utf-8')
                for _, row in df.iterrows():
                    word = str(row.iloc[0]).lower().strip()
                    if word and word != 'nan':
                        entry = LexiconEntry(
                            headword=word,
                            source='user_supplied',
                            gloss=str(row.iloc[1]) if len(row) > 1 else None,
                        )
                        lexicon.add_entry(entry)
            elif fpath.endswith('.txt'):
                with open(fpath, 'r', encoding='utf-8') as f:
                    for line in f:
                        word = line.strip().lower()
                        if word:
                            entry = LexiconEntry(
                                headword=word,
                                source='user_supplied',
                            )
                            lexicon.add_entry(entry)
        except Exception as e:
            logger.warning(f"Error loading {fname}: {e}")

    logger.info(f"User-supplied lexicon loaded: {lexicon.size()} entries")
    return lexicon if lexicon.size() > 0 else None


# =============================================================================
# Lexicon combination
# =============================================================================

def merge_lexicons(lexicons: List[Lexicon], name: str = 'combined') -> Lexicon:
    """
    Merge multiple lexicons into one.

    Merge rule (documented per AGENTS.md):
    - Union of headwords
    - On conflicts (same headword in multiple sources), keep BOTH entries
      and tag each with its source
    - Never silently overwrite

    Parameters
    ----------
    lexicons : List[Lexicon]
        Lexicons to merge.
    name : str
        Name for the merged lexicon.

    Returns
    -------
    Lexicon
        The merged lexicon.
    """
    merged = Lexicon(name=name, source='merged')
    conflicts = 0

    for lex in lexicons:
        for headword, entries in lex.entries.items():
            existing = merged.lookup(headword)
            if existing:
                conflicts += 1
            for entry in entries:
                merged.add_entry(entry)

    logger.info(f"Merged {len(lexicons)} lexicons into '{name}': "
               f"{merged.size()} headwords, {conflicts} conflicts (both kept)")
    return merged


# =============================================================================
# Lexicon condition manager
# =============================================================================

class LexiconConditions:
    """
    Manages all lexicon conditions for the benchmark.

    Conditions:
    - 'none': no lexicon features
    - 'house': only house-made lexicon
    - 'existing': all external lexicons pooled
    - 'combined': house + existing
    - Individual: 'wiktionary', 'panlex', 'user_supplied'
    """

    def __init__(self):
        self.conditions: Dict[str, Optional[Lexicon]] = {}
        self.house: Optional[Lexicon] = None
        self.externals: Dict[str, Lexicon] = {}

    def setup(self, house_dir: str, external_dir: str,
              user_dir: str, unique_words: List[str]):
        """
        Load all lexicons and set up conditions.

        Parameters
        ----------
        house_dir : str
            Path to house lexicon directory.
        external_dir : str
            Path to external lexicon directory.
        user_dir : str
            Path to user-supplied lexicon directory.
        unique_words : List[str]
            Unique words from corpus for coverage reporting.
        """
        # Load house lexicon
        self.house = load_house_lexicon(house_dir, unique_words)
        self.conditions['house'] = self.house

        # Load external lexicons
        wikt = load_wiktionary_lexicon(external_dir)
        if wikt:
            self.externals['wiktionary'] = wikt
            self.conditions['wiktionary'] = wikt

        panlex = load_panlex_lexicon(external_dir)
        if panlex:
            self.externals['panlex'] = panlex
            self.conditions['panlex'] = panlex

        user = load_user_supplied_lexicon(user_dir)
        if user:
            self.externals['user_supplied'] = user
            self.conditions['user_supplied'] = user

        # 'existing': pool all externals
        if self.externals:
            existing = merge_lexicons(list(self.externals.values()), name='existing')
            self.conditions['existing'] = existing
        else:
            logger.warning("No external lexicons available. 'existing' condition will be empty.")
            self.conditions['existing'] = Lexicon(name='existing', source='merged')

        # 'combined': house + existing
        all_lexicons = []
        if self.house and self.house.size() > 0:
            all_lexicons.append(self.house)
        if self.externals:
            all_lexicons.extend(self.externals.values())
        if all_lexicons:
            combined = merge_lexicons(all_lexicons, name='combined')
            self.conditions['combined'] = combined
        else:
            self.conditions['combined'] = Lexicon(name='combined', source='merged')

        # 'none': no lexicon
        self.conditions['none'] = None

        logger.info(f"Lexicon conditions set up: {list(self.conditions.keys())}")
        for name, lex in self.conditions.items():
            if lex:
                logger.info(f"  {name}: {lex.size()} headwords")
            else:
                logger.info(f"  {name}: (no lexicon)")

    def get_condition(self, name: str) -> Optional[Lexicon]:
        """Get a specific lexicon condition."""
        return self.conditions.get(name)

    def get_headline_conditions(self) -> List[str]:
        """Get the four headline condition names."""
        return ['none', 'house', 'existing', 'combined']

    def get_all_conditions(self) -> List[str]:
        """Get all condition names including individual externals."""
        return list(self.conditions.keys())
