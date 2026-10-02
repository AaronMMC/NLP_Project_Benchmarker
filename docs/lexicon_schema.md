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

## Notes
- Three CSV files cover rows 1-110, 441-550, and 550-672
- Gap: rows 111-440 are not yet annotated
- Row 550 (papigsaen) appears in both the 441-550 and 550-672 files
- The 441-550 file has a filename typo: 'AnnonatedLexicon' (missing 't')
- All files use the same column schema
- POS tags follow Universal Dependencies (UPOS) conventions
