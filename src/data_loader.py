"""
data_loader.py — Corpus parsing and song-boundary detection.

Handles loading the Ilocano liturgical corpus, detecting song/section
boundaries using heuristics, and producing structured data for downstream
tasks.
"""

import re
import os
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from collections import Counter

import yaml

logger = logging.getLogger(__name__)


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class LiturgicalUnit:
    """A detected unit (song/canticle/prayer/prose) from the corpus."""
    unit_id: int
    title: str
    unit_type: str  # 'hymn', 'canticle', 'prayer', 'liturgical_prose', 'rubric'
    lines: List[str]
    raw_text: str
    page: Optional[int] = None
    speaker: Optional[str] = None  # 'Pastor', 'Gimong', 'Amin', None
    heuristic_used: str = ''
    metadata: Dict = field(default_factory=dict)

    @property
    def num_lines(self):
        return len(self.lines)

    @property
    def num_tokens(self):
        return sum(len(line.split()) for line in self.lines)

    @property
    def text(self):
        return '\n'.join(self.lines)


@dataclass
class CorpusStats:
    """Summary statistics for the corpus."""
    total_lines: int = 0
    total_tokens: int = 0
    total_types: int = 0
    num_units: int = 0
    num_hymns: int = 0
    num_canticles: int = 0
    num_prayers: int = 0
    num_prose: int = 0
    num_rubrics: int = 0
    lines_per_unit: List[int] = field(default_factory=list)
    tokens_per_unit: List[int] = field(default_factory=list)
    duplicate_lines: List[Tuple[str, int]] = field(default_factory=list)
    page_range: Tuple[int, int] = (0, 0)


# =============================================================================
# Corpus loader
# =============================================================================

def load_raw_corpus(filepath: str) -> str:
    """
    Load the raw corpus file, detecting encoding.

    Parameters
    ----------
    filepath : str
        Path to the corpus text file.

    Returns
    -------
    str
        The full raw text of the corpus.
    """
    # Try UTF-8 first, fall back to latin-1
    for enc in ('utf-8', 'utf-8-sig', 'latin-1', 'cp1252'):
        try:
            with open(filepath, 'r', encoding=enc) as f:
                text = f.read()
            logger.info(f"Corpus loaded with encoding '{enc}', {len(text)} chars, "
                       f"{text.count(chr(10))} lines")
            return text
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError(f"Could not decode corpus file: {filepath}")


def parse_pages(raw_text: str) -> List[Tuple[int, str]]:
    """
    Split corpus by page markers.

    The corpus uses '=' delimited PAGE N markers. Returns list of
    (page_number, page_content) tuples.
    """
    page_pattern = re.compile(
        r'={10,}\s*\nPAGE\s+(\d+)\s*\n={10,}\s*\n',
        re.MULTILINE
    )
    parts = page_pattern.split(raw_text)

    pages = []
    # parts[0] is text before first page marker (the title line)
    # Then alternating: page_number, page_content, page_number, page_content...
    i = 1
    while i < len(parts) - 1:
        page_num = int(parts[i])
        page_content = parts[i + 1].strip()
        pages.append((page_num, page_content))
        i += 2

    logger.info(f"Parsed {len(pages)} pages: {[p[0] for p in pages]}")
    return pages


# =============================================================================
# Boundary detection heuristics
# =============================================================================

# Liturgical section titles in ALL CAPS that denote major sections
KNOWN_SECTIONS = {
    'URNOS TI PANAGDAYDAYAW': 'liturgical_prose',
    'GLORIA PATRI': 'canticle',
    'TI KYRIE': 'canticle',
    'TI GLORIA EXCELSIS': 'canticle',
    'TI INTROITA': 'rubric',
    'CREDA NICIA': 'prayer',
    'TI SERMON': 'rubric',
    'URNOS TI CEREMONIA TI COMULGAR': 'liturgical_prose',
    'TI CARARAG NGA AMAMI': 'prayer',
    'TI PAX DOMINI': 'liturgical_prose',
    'TI AGNUS DEI': 'canticle',
    'TI PANAGIWARAS': 'liturgical_prose',
    'TI NUNC DIMITTIS': 'canticle',
    'TI PAMMENDICION': 'liturgical_prose',
}

# Liturgical seasons/occasions used as sub-headings in the Preface section
SEASON_HEADINGS = {
    'Adviento', 'Pascua', 'Cuaresma', 'Panagungar',
    'Aldaw Dagiti Tallo Nga Ari', 'Aldaw Ti Iyu-uli',
    'Aldaw Ti Pentecostes', 'Trinidad',
    'Aldaw Dagiti Apostoles Ken Evangelistas',
}


def _is_all_caps_heading(line: str) -> bool:
    """Check if a line is an ALL-CAPS section heading."""
    stripped = line.strip()
    if not stripped or len(stripped) < 3:
        return False
    # Remove parenthetical notes and check
    cleaned = re.sub(r'\([^)]*\)', '', stripped).strip()
    if not cleaned:
        return False
    # Check if the cleaned line is predominantly uppercase
    alpha_chars = [c for c in cleaned if c.isalpha()]
    if not alpha_chars:
        return False
    upper_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
    return upper_ratio >= 0.8 and len(alpha_chars) >= 3


def _detect_speaker(line: str) -> Optional[str]:
    """Detect if a line begins with a speaker label (Pastor:, Gimong:, Amin:)."""
    match = re.match(r'^(Pastor|Gimong|Amin)\s*:', line)
    return match.group(1) if match else None


def _classify_unit_type(title: str, lines: List[str], speaker: Optional[str]) -> str:
    """
    Classify a liturgical unit as hymn, canticle, prayer, prose, or rubric.

    Decision rules (documented per AGENTS.md requirement):
    1. If title matches a KNOWN_SECTIONS entry, use that classification.
    2. If lines contain 'Cantaen ti Himno' (= 'Sing the Hymn'), it's a rubric
       referencing a hymn to be sung (the actual hymn text is not in the corpus).
    3. If the title contains 'Gloria', 'Kyrie', 'Agnus', 'Nunc', classify as canticle.
    4. If the title contains 'Creda' or 'Cararag' or 'Amami', classify as prayer.
    5. If text is dialogue (Pastor/Gimong), classify as liturgical_prose.
    6. If 'Amin:' speaker and text has rhythmic/poetic structure, classify as canticle.
    7. Default: liturgical_prose.
    """
    title_upper = title.upper().strip()

    # Rule 1: known sections
    for known, utype in KNOWN_SECTIONS.items():
        if known in title_upper or title_upper in known:
            return utype

    # Rule 2: hymn reference
    full_text = '\n'.join(lines)
    if 'Cantaen ti Himno' in full_text:
        return 'rubric'

    # Rule 3: canticle keywords
    canticle_keywords = ['GLORIA', 'KYRIE', 'AGNUS', 'NUNC', 'DIMITTIS', 'SANTO, SANTO']
    for kw in canticle_keywords:
        if kw in title_upper:
            return 'canticle'

    # Rule 4: prayer keywords
    prayer_keywords = ['CREDA', 'CARARAG', 'AMAMI', 'MAMATIAC']
    for kw in prayer_keywords:
        if kw in title_upper or kw in full_text.upper()[:100]:
            return 'prayer'

    # Rule 5/6: check speaker
    if speaker == 'Amin':
        return 'canticle'

    return 'liturgical_prose'


def detect_boundaries(raw_text: str, config: dict) -> Tuple[List[LiturgicalUnit], str]:
    """
    Discover song/section boundaries in the corpus.

    Tries heuristics in order of reliability:
    (a) Explicit section headings (ALL CAPS titles like 'TI GLORIA EXCELSIS')
    (b) Speaker-label blocks (Pastor:, Gimong:, Amin:)
    (c) Blank-line separated blocks
    (d) Fallback: fixed stanza heuristics

    The best heuristic is selected based on plausibility scoring.

    Parameters
    ----------
    raw_text : str
        The full corpus text.
    config : dict
        Configuration dictionary (may contain boundary overrides).

    Returns
    -------
    Tuple[List[LiturgicalUnit], str]
        List of detected units and the name of the heuristic used.
    """
    boundary_config = config.get('boundary_detection', {})
    method = boundary_config.get('method', 'auto')

    # Check for manual boundary file override
    if method == 'manual':
        manual_file = boundary_config.get('manual_boundary_file')
        if manual_file and os.path.exists(manual_file):
            logger.info(f"Using manual boundary file: {manual_file}")
            return _load_manual_boundaries(raw_text, manual_file), 'manual'

    # Check for regex override
    if method == 'regex':
        regex_pattern = boundary_config.get('regex_pattern')
        if regex_pattern:
            logger.info(f"Using regex boundary pattern: {regex_pattern}")
            return _detect_by_regex(raw_text, regex_pattern), 'regex'

    # Auto-detection: try heuristics in order
    logger.info("Auto-detecting boundaries using heuristic cascade...")

    # Parse pages first to get page context
    pages = parse_pages(raw_text)

    # Heuristic (a): ALL-CAPS section headings — this is the best for this corpus
    units_a = _detect_by_headings(pages)
    score_a = _score_plausibility(units_a, 'headings')

    # Heuristic (c): blank-line blocks (simpler, less semantic)
    units_c = _detect_by_blank_lines(pages)
    score_c = _score_plausibility(units_c, 'blank_lines')

    # Select best
    candidates = [
        (units_a, score_a, 'all_caps_headings'),
        (units_c, score_c, 'blank_line_blocks'),
    ]
    candidates.sort(key=lambda x: x[1], reverse=True)
    best_units, best_score, best_method = candidates[0]

    logger.info(f"Boundary detection scores: "
               f"headings={score_a:.2f}, blank_lines={score_c:.2f}")
    logger.info(f"Selected method: '{best_method}' (score={best_score:.2f})")

    # Reliability check
    if len(best_units) <= 1:
        logger.warning("⚠ BOUNDARY DETECTION WARNING: Only 1 unit detected! "
                       "The corpus may not have clear boundaries.")
    elif len(best_units) > 100:
        logger.warning(f"⚠ BOUNDARY DETECTION WARNING: {len(best_units)} units detected! "
                       "This seems like too many — boundaries may be too granular.")

    # Log what we found
    for u in best_units:
        logger.info(f"  Unit {u.unit_id}: [{u.unit_type}] '{u.title}' "
                   f"({u.num_lines} lines, {u.num_tokens} tokens, p.{u.page})")

    return best_units, best_method


def _detect_by_headings(pages: List[Tuple[int, str]]) -> List[LiturgicalUnit]:
    """
    Heuristic (a): Split by ALL-CAPS section headings.

    This corpus uses headings like 'GLORIA PATRI', 'TI KYRIE', etc.
    to delimit liturgical sections. Some sections also begin with
    speaker labels (Pastor:, Gimong:, Amin:).
    """
    units = []
    unit_id = 0

    for page_num, page_content in pages:
        # Split page into lines
        page_lines = page_content.split('\n')

        current_title = None
        current_lines = []
        current_speaker = None

        for line in page_lines:
            stripped = line.strip()
            if not stripped:
                continue

            # Check if this is a section heading
            if _is_all_caps_heading(stripped) and len(stripped) > 5:
                # Save previous unit if any
                if current_lines:
                    unit_type = _classify_unit_type(
                        current_title or f"Page {page_num} block",
                        current_lines,
                        current_speaker
                    )
                    units.append(LiturgicalUnit(
                        unit_id=unit_id,
                        title=current_title or f"Page {page_num} block",
                        unit_type=unit_type,
                        lines=current_lines,
                        raw_text='\n'.join(current_lines),
                        page=page_num,
                        speaker=current_speaker,
                        heuristic_used='all_caps_headings',
                    ))
                    unit_id += 1

                current_title = stripped
                current_lines = []
                current_speaker = _detect_speaker(stripped)
            else:
                # Detect speaker for this line
                sp = _detect_speaker(stripped)
                if sp and not current_speaker:
                    current_speaker = sp
                current_lines.append(stripped)

        # Save final block on this page
        if current_lines:
            unit_type = _classify_unit_type(
                current_title or f"Page {page_num} block",
                current_lines,
                current_speaker
            )
            units.append(LiturgicalUnit(
                unit_id=unit_id,
                title=current_title or f"Page {page_num} block",
                unit_type=unit_type,
                lines=current_lines,
                raw_text='\n'.join(current_lines),
                page=page_num,
                speaker=current_speaker,
                heuristic_used='all_caps_headings',
            ))
            unit_id += 1

    return units


def _detect_by_blank_lines(pages: List[Tuple[int, str]]) -> List[LiturgicalUnit]:
    """Heuristic (c): Split by blank-line separated blocks."""
    units = []
    unit_id = 0

    for page_num, page_content in pages:
        blocks = re.split(r'\n\s*\n', page_content)
        for block in blocks:
            lines = [l.strip() for l in block.strip().split('\n') if l.strip()]
            if not lines:
                continue

            title = lines[0] if _is_all_caps_heading(lines[0]) else f"Block {unit_id}"
            speaker = None
            for line in lines:
                sp = _detect_speaker(line)
                if sp:
                    speaker = sp
                    break

            unit_type = _classify_unit_type(title, lines, speaker)
            units.append(LiturgicalUnit(
                unit_id=unit_id,
                title=title,
                unit_type=unit_type,
                lines=lines,
                raw_text='\n'.join(lines),
                page=page_num,
                speaker=speaker,
                heuristic_used='blank_line_blocks',
            ))
            unit_id += 1

    return units


def _detect_by_regex(raw_text: str, pattern: str) -> List[LiturgicalUnit]:
    """Heuristic: Split by user-supplied regex."""
    pages = parse_pages(raw_text)
    full_text = '\n'.join(content for _, content in pages)
    parts = re.split(pattern, full_text)
    units = []
    for i, part in enumerate(parts):
        lines = [l.strip() for l in part.strip().split('\n') if l.strip()]
        if not lines:
            continue
        units.append(LiturgicalUnit(
            unit_id=i,
            title=lines[0][:60],
            unit_type='unknown',
            lines=lines,
            raw_text=part.strip(),
            heuristic_used='regex',
        ))
    return units


def _load_manual_boundaries(raw_text: str, boundary_file: str) -> List[LiturgicalUnit]:
    """Load boundaries from a manual CSV file."""
    import pandas as pd
    df = pd.read_csv(boundary_file)
    logger.info(f"Loaded {len(df)} manual boundaries from {boundary_file}")
    # Expect columns: title, start_line, end_line, unit_type
    # This is a stub — real implementation would slice raw_text by line numbers
    units = []
    lines = raw_text.split('\n')
    for i, row in df.iterrows():
        start = int(row.get('start_line', 0))
        end = int(row.get('end_line', len(lines)))
        unit_lines = [l.strip() for l in lines[start:end] if l.strip()]
        units.append(LiturgicalUnit(
            unit_id=i,
            title=str(row.get('title', f'Unit {i}')),
            unit_type=str(row.get('unit_type', 'unknown')),
            lines=unit_lines,
            raw_text='\n'.join(unit_lines),
            heuristic_used='manual',
        ))
    return units


def _score_plausibility(units: List[LiturgicalUnit], method: str) -> float:
    """
    Score how plausible a boundary detection result is.

    Criteria:
    - Reasonable number of units (5-30 is ideal for this corpus)
    - Units have reasonable size (not 1-line or 200-line)
    - Variety of unit types detected
    """
    if not units:
        return 0.0

    n = len(units)
    sizes = [u.num_lines for u in units]
    avg_size = sum(sizes) / len(sizes) if sizes else 0
    types_found = len(set(u.unit_type for u in units))

    # Number of units score: peak around 10-20
    if 5 <= n <= 30:
        n_score = 1.0
    elif 3 <= n <= 50:
        n_score = 0.6
    else:
        n_score = 0.2

    # Average size score: peak around 5-20 lines
    if 3 <= avg_size <= 30:
        size_score = 1.0
    elif 1 <= avg_size <= 50:
        size_score = 0.5
    else:
        size_score = 0.2

    # Type diversity score
    type_score = min(types_found / 3.0, 1.0)

    # Penalty for tiny units
    tiny_count = sum(1 for s in sizes if s <= 1)
    tiny_penalty = max(0, 1.0 - tiny_count / max(n, 1))

    score = (n_score * 0.3 + size_score * 0.3 + type_score * 0.2 + tiny_penalty * 0.2)
    return score


# =============================================================================
# Unique words loader
# =============================================================================

def load_unique_words(filepath: str) -> List[str]:
    """
    Load the unique_words.txt file, stripping the 'N. word' numbering.

    Parameters
    ----------
    filepath : str
        Path to unique_words.txt.

    Returns
    -------
    List[str]
        Sorted list of unique words (lowercased).
    """
    words = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Strip "N. " prefix
            match = re.match(r'^\d+\.\s+(.+)$', line)
            if match:
                words.append(match.group(1).strip().lower())
            else:
                words.append(line.lower())

    logger.info(f"Loaded {len(words)} unique words from {filepath}")
    return words


# =============================================================================
# Corpus statistics
# =============================================================================

def compute_corpus_stats(units: List[LiturgicalUnit], unique_words: List[str]) -> CorpusStats:
    """
    Compute comprehensive statistics about the corpus.

    Parameters
    ----------
    units : List[LiturgicalUnit]
        Detected liturgical units.
    unique_words : List[str]
        Known unique words from unique_words.txt.

    Returns
    -------
    CorpusStats
        Statistics summary.
    """
    all_lines = []
    all_tokens = []
    for u in units:
        all_lines.extend(u.lines)
        for line in u.lines:
            all_tokens.extend(line.lower().split())

    line_counts = Counter(l.strip().lower() for l in all_lines if l.strip())
    duplicate_lines = [(line, count) for line, count in line_counts.most_common()
                       if count > 1]

    types = set(all_tokens)

    stats = CorpusStats(
        total_lines=len(all_lines),
        total_tokens=len(all_tokens),
        total_types=len(types),
        num_units=len(units),
        num_hymns=sum(1 for u in units if u.unit_type == 'hymn'),
        num_canticles=sum(1 for u in units if u.unit_type == 'canticle'),
        num_prayers=sum(1 for u in units if u.unit_type == 'prayer'),
        num_prose=sum(1 for u in units if u.unit_type == 'liturgical_prose'),
        num_rubrics=sum(1 for u in units if u.unit_type == 'rubric'),
        lines_per_unit=[u.num_lines for u in units],
        tokens_per_unit=[u.num_tokens for u in units],
        duplicate_lines=duplicate_lines,
        page_range=(min(u.page for u in units if u.page),
                    max(u.page for u in units if u.page)) if units else (0, 0),
    )

    return stats


def print_corpus_inspection(raw_text: str, units: List[LiturgicalUnit],
                            stats: CorpusStats, unique_words: List[str],
                            log_dir: str = 'logs'):
    """
    Print detailed corpus inspection results (Phase 1 requirement).

    Logs encoding, size, line count, blank-line patterns, numbering,
    uppercase headings, repeated lines, Latin/Spanish markers, bracketed
    rubrics, and illustrative snippets.
    """
    os.makedirs(log_dir, exist_ok=True)

    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("CORPUS INSPECTION REPORT")
    report_lines.append("=" * 70)
    report_lines.append(f"Total characters: {len(raw_text)}")
    report_lines.append(f"Total raw lines: {raw_text.count(chr(10)) + 1}")
    report_lines.append(f"Total content lines (in units): {stats.total_lines}")
    report_lines.append(f"Total tokens: {stats.total_tokens}")
    report_lines.append(f"Unique types: {stats.total_types}")
    report_lines.append(f"Unique words from wordlist: {len(unique_words)}")
    report_lines.append(f"Page range: {stats.page_range[0]}-{stats.page_range[1]}")
    report_lines.append(f"")
    report_lines.append(f"--- Units detected: {stats.num_units} ---")
    report_lines.append(f"  Hymns: {stats.num_hymns}")
    report_lines.append(f"  Canticles: {stats.num_canticles}")
    report_lines.append(f"  Prayers: {stats.num_prayers}")
    report_lines.append(f"  Liturgical prose: {stats.num_prose}")
    report_lines.append(f"  Rubrics: {stats.num_rubrics}")
    report_lines.append(f"")

    if stats.lines_per_unit:
        report_lines.append(f"Lines per unit: min={min(stats.lines_per_unit)}, "
                          f"max={max(stats.lines_per_unit)}, "
                          f"mean={sum(stats.lines_per_unit)/len(stats.lines_per_unit):.1f}")
        report_lines.append(f"Tokens per unit: min={min(stats.tokens_per_unit)}, "
                          f"max={max(stats.tokens_per_unit)}, "
                          f"mean={sum(stats.tokens_per_unit)/len(stats.tokens_per_unit):.1f}")

    report_lines.append(f"")
    report_lines.append(f"--- Duplicate lines (top 10): ---")
    for line, count in stats.duplicate_lines[:10]:
        report_lines.append(f"  [{count}x] {line[:80]}")

    report_lines.append(f"")
    report_lines.append(f"--- Unit details: ---")
    for u in units:
        report_lines.append(f"  #{u.unit_id} [{u.unit_type}] p.{u.page} "
                          f"'{u.title[:50]}' ({u.num_lines} lines, {u.num_tokens} tokens)")
        if u.lines:
            report_lines.append(f"    First line: {u.lines[0][:80]}")

    report_lines.append(f"")
    report_lines.append(f"--- Illustrative snippets: ---")
    for u in units[:3]:
        report_lines.append(f"  Unit '{u.title}':")
        for line in u.lines[:3]:
            report_lines.append(f"    | {line[:80]}")
        report_lines.append(f"    ...")

    report = '\n'.join(report_lines)
    logger.info(report)

    # Write to log file
    log_path = os.path.join(log_dir, 'corpus_inspection.md')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(report)
    logger.info(f"Corpus inspection written to {log_path}")

    return report
