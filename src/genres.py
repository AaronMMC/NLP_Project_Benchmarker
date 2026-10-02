"""
genres.py — Genre data loading and TODO stubs for future non-song data.

Only the `song` genre exists today. basic_text, conversation, and poetry
will be supplied later and are handled under the TODO policy.
"""

import os
import glob
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import pandas as pd

from src.data_loader import LiturgicalUnit

logger = logging.getLogger(__name__)


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class GenreDocument:
    """A single document/passage with a genre label."""
    doc_id: str
    text: str
    lines: List[str]
    genre: str  # 'song', 'basic_text', 'conversation', 'poetry'
    source: str  # e.g. 'corpus_ilocano_liturgy', 'news_source_1'
    unit_id: Optional[int] = None
    metadata: Dict = field(default_factory=dict)


# =============================================================================
# Genre loader
# =============================================================================

def load_genres(genres_dir: str, song_units: List[LiturgicalUnit]) -> Dict[str, List[GenreDocument]]:
    """
    Load all available genre data.

    Currently only 'song' is available from the corpus. Other genres
    are loaded if files exist in the expected format.

    Expected format for non-song genres:
        data/genres/<genre>/<source_name>.txt
        One unit per blank-line-separated block.

    Parameters
    ----------
    genres_dir : str
        Path to data/genres/ directory.
    song_units : List[LiturgicalUnit]
        Liturgical units classified as songs/canticles from the corpus.

    Returns
    -------
    Dict[str, List[GenreDocument]]
        Mapping from genre name to list of documents.
    """
    genres: Dict[str, List[GenreDocument]] = {}

    # Load songs from the liturgical corpus
    # Songs include both 'hymn' and 'canticle' unit types since both are
    # sung texts that should be detected as songs
    song_docs = []
    for unit in song_units:
        doc = GenreDocument(
            doc_id=f"song_{unit.unit_id}",
            text=unit.text,
            lines=unit.lines,
            genre='song',
            source='corpus_ilocano_liturgy',
            unit_id=unit.unit_id,
            metadata={
                'title': unit.title,
                'unit_type': unit.unit_type,
                'page': unit.page,
                'speaker': unit.speaker,
            }
        )
        song_docs.append(doc)

    genres['song'] = song_docs
    logger.info(f"Loaded {len(song_docs)} song documents from liturgical corpus")

    # TODO(negatives): Load basic_text genre data
    # Expected: data/genres/basic_text/<source_name>.txt
    # Format: one passage per blank-line-separated block
    # Content: news articles, essays, informational prose in Ilocano
    basic_text_docs = _load_genre_dir(os.path.join(genres_dir, 'basic_text'), 'basic_text')
    if basic_text_docs:
        genres['basic_text'] = basic_text_docs
        logger.info(f"Loaded {len(basic_text_docs)} basic_text documents")

    # TODO(negatives): Load conversation genre data
    # Expected: data/genres/conversation/<source_name>.txt
    # Format: one dialogue per blank-line-separated block
    # Content: dialogue / spoken-style Ilocano text
    conversation_docs = _load_genre_dir(os.path.join(genres_dir, 'conversation'), 'conversation')
    if conversation_docs:
        genres['conversation'] = conversation_docs
        logger.info(f"Loaded {len(conversation_docs)} conversation documents")

    # TODO(negatives): Load poetry genre data
    # Expected: data/genres/poetry/<source_name>.txt
    # Format: one poem per blank-line-separated block
    # Content: Ilocano poetry (a hard confuser for songs)
    poetry_docs = _load_genre_dir(os.path.join(genres_dir, 'poetry'), 'poetry')
    if poetry_docs:
        genres['poetry'] = poetry_docs
        logger.info(f"Loaded {len(poetry_docs)} poetry documents")

    # Report what's available
    available = [g for g in genres if genres[g]]
    missing = [g for g in ['song', 'basic_text', 'conversation', 'poetry'] if g not in available]

    if missing:
        logger.warning(
            f"⚠ GENRE DATA WARNING: Only {available} genres available. "
            f"Missing: {missing}. "
            f"Tasks A, B, D cannot be fully run without all genres. "
            f"Running in POSITIVE-ONLY mode."
        )

    return genres


def _load_genre_dir(genre_dir: str, genre_name: str) -> List[GenreDocument]:
    """Load documents from a genre directory."""
    docs = []
    if not os.path.exists(genre_dir):
        return docs

    txt_files = glob.glob(os.path.join(genre_dir, '*.txt'))
    for fpath in txt_files:
        source_name = os.path.splitext(os.path.basename(fpath))[0]
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                content = f.read()

            # Split by blank lines into passages
            blocks = content.split('\n\n')
            for i, block in enumerate(blocks):
                lines = [l.strip() for l in block.strip().split('\n') if l.strip()]
                if not lines:
                    continue
                doc = GenreDocument(
                    doc_id=f"{genre_name}_{source_name}_{i}",
                    text='\n'.join(lines),
                    lines=lines,
                    genre=genre_name,
                    source=source_name,
                )
                docs.append(doc)
        except Exception as e:
            logger.warning(f"Error loading {fpath}: {e}")

    return docs


def is_positive_only_mode(genres: Dict[str, List[GenreDocument]]) -> bool:
    """
    Check if we're in positive-only mode (only song data available).

    Returns True if only 'song' genre has data.
    """
    available = [g for g, docs in genres.items() if docs]
    return available == ['song']


def get_available_genre_pairs(genres: Dict[str, List[GenreDocument]]) -> List[Tuple[str, str]]:
    """
    Get all available genre pairs for pairwise comparison.

    Returns list of (genre_a, genre_b) tuples where data exists for both.
    """
    available = [g for g, docs in genres.items() if docs]
    pairs = []
    for i, a in enumerate(available):
        for b in available[i+1:]:
            pairs.append((a, b))
    return pairs
