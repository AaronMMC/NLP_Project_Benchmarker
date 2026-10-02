"""
preprocess.py — Text normalisation and spelling-variant mapping.

Handles Unicode normalisation, lowercase, punctuation, hyphen/apostrophe
handling, and the critical old↔modern Ilocano spelling variant mapping
(c→k, qu→k, etc.).
"""

import re
import os
import logging
import unicodedata
from typing import List, Dict, Set, Tuple, Optional
from collections import Counter

logger = logging.getLogger(__name__)


# =====================================================================
# IDEA 2: Spelling-variant handling
# WHY:     The corpus uses older orthography (c/qu) while modern lexicons
#          use k-based spelling. Without mapping, lexicon lookups miss
#          most entries, producing misleadingly low coverage.
# HOW:     Rule-based candidate generation maps each corpus token to a
#          set of modern spelling candidates. Also supports fuzzy matching
#          via edit distance (done at lookup time in lexicons.py).
#          Coverage is reported before and after normalisation.
# OUTPUT:  logs/normalisation_rules.md, coverage stats in REPORT.md
# CAVEAT:  Rules are based on known Ilocano orthography shifts; may
#          over-generate candidates for some words. Not all old spellings
#          follow regular patterns.
# =====================================================================


# =============================================================================
# Normalisation rules
# =============================================================================

def unicode_normalise(text: str) -> str:
    """Apply Unicode NFC normalisation."""
    return unicodedata.normalize('NFC', text)


def normalise_text(text: str) -> str:
    """
    Apply standard text normalisation.

    Steps:
    1. Unicode NFC
    2. Lowercase
    3. Normalise whitespace
    4. Handle apostrophes and hyphens (preserve them as meaningful)
    5. Remove extraneous punctuation but keep sentence structure
    """
    text = unicode_normalise(text)
    text = text.lower()
    # Normalise various apostrophe characters to standard
    text = re.sub(r'[\u2018\u2019\u0060\u00B4]', "'", text)
    # Normalise various dash characters to standard hyphen
    text = re.sub(r'[\u2013\u2014\u2015]', '-', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenise(text: str) -> List[str]:
    """
    Deterministic tokeniser for Ilocano liturgical text.

    Rules:
    - Split on whitespace
    - Preserve hyphens within words (e.g. 'jesu-cristo', 'ay-ayatec')
    - Preserve apostrophes within words (e.g. "talna't", "napno't")
    - Strip leading/trailing punctuation except apostrophe and hyphen
    - Remove empty tokens

    This tokeniser is deterministic: same input always produces same output.
    """
    text = normalise_text(text)
    raw_tokens = text.split()
    tokens = []
    for tok in raw_tokens:
        # Strip leading punctuation (except apostrophe/hyphen)
        tok = re.sub(r"^[^\w'-]+", '', tok)
        # Strip trailing punctuation (except apostrophe/hyphen)
        tok = re.sub(r"[^\w'-]+$", '', tok)
        # Handle trailing apostrophe-t pattern (e.g. "talna't" → keep as-is)
        # Handle trailing period or comma (already stripped above)
        if tok:
            tokens.append(tok)
    return tokens


# =============================================================================
# Spelling variant mapping: Old Ilocano ↔ Modern Ilocano
# =============================================================================

# The old orthography used in the corpus follows Spanish conventions:
# - 'c' before a, o, u → 'k' in modern (cararag → kararag)
# - 'qu' before e, i → 'k' in modern (querer → not applicable, but pattern exists)
# - 'c' before e, i → 's' in some Spanish loanwords (ceremonia → seremonia)
#   but in Ilocano native words, 'c' before e/i is less common
# - 'ñ' can appear in Spanish loans
# - 'gui' → 'gi' in some contexts

# Rules as (pattern, replacement, description) — applied to generate candidates
VARIANT_RULES = [
    # Core c→k rules
    (r'c(?=[aou])', 'k', 'c→k before a/o/u'),
    (r'c(?=[^aeiou\s])', 'k', 'c→k before consonant'),
    (r'qu(?=[ei])', 'k', 'qu→k before e/i'),
    (r'(?<=[aeiou])c(?=[aou])', 'k', 'medial c→k before a/o/u'),

    # Less common but attested
    (r'gu(?=[ei])', 'g', 'gu→g before e/i (soft g)'),
    (r'ñ', 'ny', 'ñ→ny'),
]

# Direct known mappings from corpus inspection (high-confidence)
KNOWN_VARIANTS = {
    # Format: old_form → modern_form
    'cararag': 'kararag',
    'caasi': 'kaasi',
    'cuna': 'kuna',
    'cadagiti': 'kadagiti',
    'cadatayo': 'kadatayo',
    'cadacayo': 'kadakayo',
    'cadacami': 'kadakami',
    'cadacuada': 'kadakuada',
    'cas': 'kas',
    'casta': 'kasta',
    'coma': 'koma',
    'canayon': 'kanayon',
    'caniac': 'kaniak',
    'cantaen': 'kantaen',
    'canenyo': 'kanenyo',
    'cangatoan': 'kangatoan',
    'comulgar': 'komulgar',
    'copa': 'kopa',
    'calpasan': 'kalpasan',
    'cucuam': 'kukuam',
    'cuam': 'kuam',
    'cunana': 'kunana',
    'cuncunana': 'kunkunana',
    'cunaen': 'kunaen',
    'cunaenmi': 'kunaenmi',
    'caanoman': 'kaanoman',
    'caasiannacami': 'kaasiannakami',
    'caasyannac': 'kaasiannak',
    'cabaelanmi': 'kabaelanmi',
    'cabusormi': 'kabusormi',
    'cadi': 'kadi',
    'cayatmo': 'kayatmo',
    'cayo': 'kayo',
    'collect': 'kollek',
    'creda': 'kreda',
    'cristiana': 'kristiana',
    'cristo': 'kristo',
    'cruz': 'krus',
    'cuaresma': 'kuaresma',
    'cuma': 'kuma',
    'dacami': 'dakami',
    'dacayo': 'dakayo',
    'siac': 'siak',
    'bagic': 'bagik',
    'ipudnoc': 'ipudnok',
    'basbasolco': 'basbasolko',
    'lablabsingco': 'lablabsingko',
    'apoc': 'apok',
    'rebbengco': 'rebbengko',
    'kinunac': 'kinunak',
    'bigbigec': 'bigbigek',
    'dawatec': 'dawatek',
    'saadco': 'saadko',
    'paglabsingac': 'paglabsingak',
    'ipacdaarco': 'ipakdaarko',
    'panaglagidingitco': 'panaglagidingitko',
}


def generate_modern_candidates(word: str) -> Set[str]:
    """
    Generate candidate modern spellings for an old-orthography word.

    Uses rule-based transformations to produce multiple candidate
    modern spellings. Does NOT destructively replace — returns a set
    of candidates that may include the original.

    Parameters
    ----------
    word : str
        The word in old orthography (lowercase).

    Returns
    -------
    Set[str]
        Set of candidate modern spellings, always includes the original.
    """
    candidates = {word}  # Always include original

    # Check known variants first
    if word in KNOWN_VARIANTS:
        candidates.add(KNOWN_VARIANTS[word])

    # Apply rules to generate candidates
    for pattern, replacement, _ in VARIANT_RULES:
        try:
            modernised = re.sub(pattern, replacement, word)
            if modernised != word:
                candidates.add(modernised)
        except re.error:
            continue

    # Also try: if word ends in 'c', try replacing with 'k'
    if word.endswith('c') and len(word) > 1:
        candidates.add(word[:-1] + 'k')

    return candidates


def generate_old_candidates(word: str) -> Set[str]:
    """
    Generate candidate old spellings for a modern-orthography word.

    Reverse of generate_modern_candidates. Useful for looking up
    modern lexicon entries from old corpus words.

    Parameters
    ----------
    word : str
        The word in modern orthography (lowercase).

    Returns
    -------
    Set[str]
        Set of candidate old spellings.
    """
    candidates = {word}

    # Reverse known variants
    reverse_map = {v: k for k, v in KNOWN_VARIANTS.items()}
    if word in reverse_map:
        candidates.add(reverse_map[word])

    # Reverse rules: k→c before a/o/u
    candidates.add(re.sub(r'k(?=[aou])', 'c', word))
    # k→qu before e/i
    candidates.add(re.sub(r'k(?=[ei])', 'qu', word))
    # Final k→c
    if word.endswith('k') and len(word) > 1:
        candidates.add(word[:-1] + 'c')

    return candidates


class SpellingNormaliser:
    """
    Manages spelling variant mapping and tracks rule usage.

    Produces a normalisation log with rule hit counts for transparency.
    """

    def __init__(self):
        self.rule_hits = Counter()
        self.variant_map: Dict[str, Set[str]] = {}  # word → set of candidates
        self._processed = False

    def build_variant_map(self, tokens: List[str]) -> Dict[str, Set[str]]:
        """
        Build a variant map for all corpus tokens.

        Parameters
        ----------
        tokens : List[str]
            All tokens from the corpus.

        Returns
        -------
        Dict[str, Set[str]]
            Mapping from each token to its set of spelling candidates.
        """
        unique_tokens = set(tokens)
        for tok in unique_tokens:
            candidates = generate_modern_candidates(tok)
            self.variant_map[tok] = candidates

            # Track which rules fired
            for pattern, replacement, desc in VARIANT_RULES:
                try:
                    if re.sub(pattern, replacement, tok) != tok:
                        self.rule_hits[desc] += 1
                except re.error:
                    continue

            if tok in KNOWN_VARIANTS:
                self.rule_hits['known_variant_lookup'] += 1

        self._processed = True
        logger.info(f"Built variant map for {len(unique_tokens)} unique tokens, "
                   f"{sum(1 for v in self.variant_map.values() if len(v) > 1)} have variants")
        return self.variant_map

    def get_candidates(self, word: str) -> Set[str]:
        """Get modern spelling candidates for a word."""
        if word in self.variant_map:
            return self.variant_map[word]
        return generate_modern_candidates(word)

    def write_rule_log(self, log_dir: str = 'logs'):
        """
        Write normalisation rules log with examples and hit counts.

        Creates logs/normalisation_rules.md
        """
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, 'normalisation_rules.md')

        lines = []
        lines.append("# Normalisation Rules Log")
        lines.append("")
        lines.append("## Rules applied")
        lines.append("")
        lines.append("| Rule | Pattern | Replacement | Hit Count |")
        lines.append("|------|---------|-------------|-----------|")
        for pattern, replacement, desc in VARIANT_RULES:
            count = self.rule_hits.get(desc, 0)
            lines.append(f"| {desc} | `{pattern}` | `{replacement}` | {count} |")
        lines.append(f"| known_variant_lookup | (dict) | (dict) | "
                    f"{self.rule_hits.get('known_variant_lookup', 0)} |")

        lines.append("")
        lines.append("## Example mappings (first 20)")
        lines.append("")
        lines.append("| Original | Candidates |")
        lines.append("|----------|------------|")
        count = 0
        for word, candidates in sorted(self.variant_map.items()):
            if len(candidates) > 1:
                others = ', '.join(sorted(candidates - {word}))
                lines.append(f"| {word} | {others} |")
                count += 1
                if count >= 20:
                    break

        lines.append("")
        lines.append(f"Total tokens with variants: "
                    f"{sum(1 for v in self.variant_map.values() if len(v) > 1)}")
        lines.append(f"Total unique tokens processed: {len(self.variant_map)}")

        with open(log_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        logger.info(f"Normalisation rules log written to {log_path}")


def compute_coverage_change(tokens: List[str], lexicon_headwords: Set[str],
                           normaliser: SpellingNormaliser) -> Dict:
    """
    Compute lexicon coverage before and after spelling normalisation.

    This is a key metric: the orthography gap between corpus and lexicon
    is itself a headline finding.

    Parameters
    ----------
    tokens : List[str]
        Corpus tokens.
    lexicon_headwords : Set[str]
        Set of headwords in the lexicon (modern spelling).
    normaliser : SpellingNormaliser
        The normaliser with variant map built.

    Returns
    -------
    Dict
        Coverage statistics before and after normalisation.
    """
    unique_tokens = set(tokens)

    # Before normalisation: direct match
    before_covered = unique_tokens & lexicon_headwords
    before_oov = unique_tokens - lexicon_headwords

    # After normalisation: any candidate matches
    after_covered = set()
    after_oov = set()
    for tok in unique_tokens:
        candidates = normaliser.get_candidates(tok)
        if candidates & lexicon_headwords:
            after_covered.add(tok)
        else:
            after_oov.add(tok)

    return {
        'total_types': len(unique_tokens),
        'before_covered': len(before_covered),
        'before_oov': len(before_oov),
        'before_coverage_pct': len(before_covered) / max(len(unique_tokens), 1) * 100,
        'after_covered': len(after_covered),
        'after_oov': len(after_oov),
        'after_coverage_pct': len(after_covered) / max(len(unique_tokens), 1) * 100,
        'coverage_gain': len(after_covered) - len(before_covered),
        'coverage_gain_pct': (len(after_covered) - len(before_covered)) / max(len(unique_tokens), 1) * 100,
    }
