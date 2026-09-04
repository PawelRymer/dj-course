# Project Guidance: M1/tokenizer

## Project Overview

This is a Polish NLP educational project for building custom tokenizers using the `tokenizers` library (PyPI). The project works with Polish language corpora (NKJP, Wolne Lektury) to train and test tokenizers for large language models.

**Key entrypoints:**
- `tokenizer-build.py` - build and train custom tokenizers (BPE model)
- `tokenize-pan-tadeusz.py` - example tokenization of Polish text
- `tokenize-visualize.py` - visualize tokenization output
- `corpora.py` - utility to access corpus files (NKJP, WOLNELEKTURY, PAN_TADEUSZ, ALL)
- `tokenizers/` - directory storing trained tokenizer JSON files

## Python Setup

### Dependencies

```bash
pip install -r requirements.txt
```

Current dependencies: `tokenizers` (PyPI). No additional test framework or build system configured.

### Running Scripts

All scripts are standalone Python files. Run them directly:

```bash
python tokenizer-build.py          # Train a tokenizer
python tokenize-pan-tadeusz.py     # Tokenize specific text
python tokenize-visualize.py       # Visualize tokenization
python corpora.py                  # Check available corpora
```

### Corpus Access

Corpora are accessed via `corpora.py`. Requires these sibling directories to exist:
- `../korpus-nkjp/output/` - NKJP corpus files (*.txt)
- `../korpus-wolnelektury/` - Wolne Lektury corpus files (*.txt)

Without these, `tokenizer-build.py` will fail when calling `get_corpus_file()`. If testing locally, corpora paths may not exist; handle gracefully or skip corpus-dependent code.

### No Tests or Build System

This project has no pytest, unittest, or build configuration. Verification is done by running scripts directly and checking output or visualizations.

## Coding Conventions

- Files use Polish comments and variable names (e.g., `corpora.py`, `latarnik.txt`)
- BPE tokenizer configuration is defined in `tokenizer-build.py` (vocab_size, min_frequency, special tokens)
- Trained tokenizers are saved as JSON files in `tokenizers/` directory
- Terms of Use: Bielik models require acceptance of terms at https://huggingface.co/ and https://bielik.ai/terms/

## Common Tasks

**Train a custom tokenizer:**
1. Edit `tokenizer-build.py` to configure BPE trainer (vocab_size, special_tokens, etc.)
2. Point `FILES` to desired corpus via `get_corpus_file()`
3. Update `TOKENIZER_OUTPUT_FILE` output path
4. Run `python tokenizer-build.py`

**Tokenize new text:**
1. Load a tokenizer from `tokenizers/*.json`
2. Call `tokenizer.encode(text)` to get tokens and IDs
3. Decode with `tokenizer.decode(ids)` if needed

**Debug corpus paths:**
- Run `python corpora.py` to verify corpus availability
- Uses `pathlib.Path` relative to the tokenizer directory
- Adjust `CORPORA_DIRS` in `corpora.py` if paths change

## Quirks and Gotchas

- **Corpus paths are relative:** `corpora.py` uses relative paths (`../korpus-*`). Scripts assume they're run from the `M1/tokenizer/` directory; running from elsewhere breaks corpus loading.
- **No error handling on missing corpora:** If corpora directories don't exist, `glob()` returns empty lists silently. Train scripts may fail later with unclear errors.
- **Tokenizer output is in `tokenizers/` directory:** Always save trained tokenizers there to keep them organized.
- **Polish text encoding:** Ensure files are UTF-8 encoded (the corpus files are).
