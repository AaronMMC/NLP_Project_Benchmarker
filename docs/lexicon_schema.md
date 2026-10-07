# Inferred Lexicon Schema

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

## Current Inventory & Coverage
- Source files detected: `AnnotatedIlocanoLexicon.csv`
- Total headwords loaded: 670
- Unique words coverage: 669/672 (99.6%)
- Uncovered word ranges: `['221', '229', '540']`
- Duplicate headwords across files: 0
- All files use the same column schema
- POS tags follow Universal Dependencies (UPOS) conventions
