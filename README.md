# Bengali BPE Tokenizer

> **Byte Pair Encoding from scratch for Bengali (Bangla) and other non-Latin scripts.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## What is Byte Pair Encoding?

**Byte Pair Encoding (BPE)** is a subword tokenisation algorithm that sits
between character-level and word-level tokenisation. Instead of splitting
text into individual characters or whole words, it learns a vocabulary of
**variable-length subword units** by iteratively merging the most frequent
adjacent symbol pairs in a corpus.

### Key properties

| Property    | Detail                                                      |
| ----------- | ----------------------------------------------------------- |
| Origin      | Gage (1994, data compression) NLP by Sennrich et al. (2016) |
| Vocabulary  | Fixed-size, learned from corpus data                        |
| Handles OOV | Yes, unknown words are decomposed into known subwords       |
| Used by     | GPT-2, GPT-3, GPT-4, RoBERTa, BART, and many others         |

## Why Tokenisation Matters

A Tokenizer is the **first** and **last** component of every NLP pipeline.
Its quality directly affects:

- **Model vocabulary efficiency** - a good Tokenizer avoids wasting ID slots.
- **OOV handling** - subword models can represent any input, unlike word-level models.
- **Training data efficiency** - better tokenisation → shorter sequences → faster training.
- **Downstream accuracy** - poorly tokenised input degrades all downstream tasks.

## Why Non-Latin Languages Are Challenging

Most popular Tokenizers were designed and evaluated on **English** and
other Latin-script languages. Non-Latin scripts present unique challenges:

1. **Multi-byte UTF-8 characters**: Bengali characters span 2-3 bytes.
   Naively splitting at bytes produces nonsensical, non-linguistic fragments.

2. **Character composition**: Many scripts have combining characters,
   diacritics, and dependent vowels that only make sense attached to
   a base character.

3. **No spaces inside words**: In some scripts, word boundaries are less
   clear-cut than in Latin text.

4. **Rich morphology**: Bengali, Arabic, Turkish, Finnish, etc. have highly
   inflected word forms, producing many unique surface forms per root.

## Bengali: Token Inflation Problem

Bengali suffers from a particularly severe **token inflation** problem with
English-centric Tokenizers:

- **GPT-2 Tokenizer** (trained mostly on English): A common Bengali word like
  `বাংলাদেশ` (Bangladesh) can be split into **10-15 tokens** because the
  Tokenizer's vocabulary contains few Bengali subwords.
- This means Bengali text uses **3-5× more tokens** than equivalent English
  text, leading to:
  - Truncated context windows
  - Higher inference costs
  - Degraded model performance

This project addresses the problem by training BPE **directly on Bengali
text**, producing a vocabulary where common Bengali morphemes become single
tokens.

## Installation

### From source (recommended)

```bash
git clone https://github.com/your-org/bengali-bpe.git
cd bengali-bpe

# Create a virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # Linux / macOS

# Install the package in editable mode
pip install -e ".[dev]"
```

### Minimal install (no dev tools)

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Train a Tokenizer

```bash
python bpe.py train \
    --file examples/corpus.txt \
    --vocab-size 2000 \
    --output output/
```

### 2. Encode text

```bash
python bpe.py encode \
    --model output/ \
    --text "আমি বাংলাদেশে থাকি।" \
    --show-tokens
```

### 3. Decode tokens

```bash
python bpe.py decode \
    --model output/ \
    --tokens 245 891 7 42
```

### 4. View statistics

```bash
python bpe.py stats \
    --model output/ \
    --top-merges 30 \
    --corpus-file examples/corpus.txt
```

### 5. Inspect a word step-by-step

```bash
python bpe.py inspect \
    --model output/ \
    --word বাংলাদেশ
```

**Example output:**

```
──────────────────────────────────────────────────────────
  BPE Inspection — 'বাংলাদেশ'
──────────────────────────────────────────────────────────

  Step 0  [            Initial             ]  →  ব + া + ং + ল + া + দ + ে + শ + </w>
  Step 1  [    Merge ('া' + 'ং') → াং     ]  →  ব + াং + ল + া + দ + ে + শ + </w>
  Step 2  [    Merge ('ব' + 'াং') → বাং     ]  →  বাং + ল + া + দ + ে + শ + </w>
  Step 3  [     Merge ('ল' + 'া') → লা     ]   →  বাং + লা + দ + ে + শ + </w>
  Step 4  [   Merge ('বাং' + 'লা') → বাংলা   ]  →  বাংলা + দ + ে + শ + </w>
  Step 5  [     Merge ('দ' + 'ে') → দে     ]   →  বাংলা + দে + শ + </w>
  Step 6  [    Merge ('দে' + 'শ') → দেশ    ]  →  বাংলা + দেশ + </w>
  Step 7  [Merge ('দেশ' + '</w>') → দেশ</w>]  →  বাংলা + দেশ</w>

  Final tokens : ['বাংলা', 'দেশ</w>']
  Token IDs    : [892, 1034]
  Merge steps  : 7
```

## CLI Reference

```
usage: bpe [-h] [--log-level {DEBUG,INFO,WARNING,ERROR}] <command> ...

Commands:
  train    Train a BPE model from a UTF-8 corpus file.
  encode   Encode text using a trained model.
  decode   Decode token IDs back to text.
  stats    Display vocabulary and training statistics.
  inspect  Visualise BPE merge steps for a single word.
```

### `train`

```
python bpe.py train
    --file PATH          UTF-8 corpus file (required)
    --vocab-size N       Target vocabulary size (default: 8000)
    --output DIR         Output directory (required)
    --language LANG      Language tag for metadata (default: bengali)
    --min-freq N         Min word frequency (default: 2)
    --normalization FORM Unicode form: NFC|NFD|NFKC|NFKD (default: NFC)
    --no-eow             Disable end-of-word marker
    --no-progress        Suppress progress bar
    --export-vocab-csv   Also write vocab.csv
```

### `encode`

```
python bpe.py encode
    --model DIR          Model directory (required)
    --text TEXT          Text to encode (or use --file)
    --file PATH          Encode entire file
    --show-tokens        Print token strings alongside IDs
```

### `decode`

```
python bpe.py decode
    --model DIR          Model directory (required)
    --tokens ID [ID...]  Token IDs to decode
```

### `stats`

```
python bpe.py stats
    --model DIR          Model directory (required)
    --top-merges N       Show top N merge rules (default: 20)
    --export-csv         Write vocab.csv to model directory
    --corpus-file PATH   Compute compression ratio from original corpus
    --plot               Plot merge frequency curve (requires matplotlib)
```

### `inspect`

```
python bpe.py inspect
    --model DIR          Model directory (required)
    --word WORD          Single word to inspect
```

## Python API

```python
from bpe import BPETrainer, BPETokenizer, TrainerConfig

# Train
config = TrainerConfig(vocab_size=8000, language="bengali")
trainer = BPETrainer(config)
with open("corpus.txt", encoding="utf-8") as f:
    text = f.read()
model = trainer.train(text)
model.save("output/")

# Load and use
tok = BPETokenizer("output/")
ids = tok.encode("আমি বাংলাদেশে থাকি।")
print(ids)                # [45, 120, 7, ...]
print(tok.decode(ids))    # "আমি বাংলাদেশে থাকি।"
print(tok.tokenize("বাংলাদেশ"))  # ['বাংলা', 'দেশ</w>']
```

## Project Structure

```
bengali-bpe/
├── bpe.py                     # Entry-point script
├── pyproject.toml             # Package metadata and tool config
├── requirements.txt
├── LICENSE
├── README.md
│
├── src/bpe/
│   ├── __init__.py            # Public package API
│   ├── unicode_utils.py       # Unicode/UTF-8 utilities
│   ├── vocabulary.py          # Token ↔ ID mapping
│   ├── trainer.py             # BPE training algorithm
│   ├── encoder.py             # Encode text → token IDs
│   ├── decoder.py             # Decode token IDs → text
│   ├── tokenizer.py           # High-level tokenizer class
│   ├── cli.py                 # Command-line interface
│   └── utils.py               # File I/O, timing, metrics
│
├── examples/
│   └── corpus.txt             # Sample Bengali corpus
│
├── output/                    # Trained model files (gitignored)
│
├── docs/
│   └── algorithm.md           # Algorithm documentation
│
└── tests/
    ├── test_unicode_utils.py
    ├── test_vocabulary.py
    ├── test_trainer.py
    └── test_tokenizer.py
```

## Implementation Details

### Unicode (the correct way)

This Tokenizer operates exclusively on **Unicode code points** (Python `str`).
It never touches raw UTF-8 bytes. This means:

- `list("বাং")` → `['ব', 'া', 'ং']` — always correct for any script.
- No byte-splitting that would break multi-byte characters.
- All regex operations use `re.UNICODE` flag.

### Bengali Unicode block (U+0980-U+09FF)

| Code point     | Character | Description                                  |
| -------------- | --------- | -------------------------------------------- |
| U+09CD | ্ | Virama / Hasanta — suppresses inherent vowel |
| U+0982 | ং | Anusvara — nasalisation |
| U+0983 | ঃ | Visarga |
| U+0981 | ঁ | Chandrabindu |
| U+09BC | ় | Nukta |
| U+09BE-U+09CC | া-ৌ | Dependent vowel signs |
| U+09E6-U+09EF | ০-৯ | Bengali digits |

### Conjuncts

Bengali conjuncts (যুক্তবর্ণ) are formed as:

```
consonant + virama (্) + consonant
```

Example: `ক` + `্` + `ষ` = `ক্ষ` (three code points, rendered as one glyph).

The Tokenizer preserves virama and learns to merge conjuncts as natural
subword units — no special handling required.

### NFC normalisation

Bengali vowel signs have both composed and decomposed forms.
NFC normalisation canonicalises these before training, ensuring consistent
tokenisation.

## Algorithm Complexity

| Step                     | Time      | Space   |
| ------------------------ | --------- | ------- |
| Pre-tokenisation | O(N) | O(N) |
| Initial vocab build | O(N) | O(C) |
| Pair counting | O(W·L) | O(P) |
| Each merge (incremental) | O(C·L) | O(W·L) |
| Total training | O(V·W·L) | O(W·L) |
| Encoding one word | O(M·L) | O(L) |

Where:

- N = corpus characters
- W = unique word types
- L = average word length (characters)
- C = word types containing the merged pair
- P = unique pairs
- V = number of merges
- M = number of merge rules

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

## Future Improvements

- **Byte-level BPE**: Fall back to UTF-8 bytes for unknown characters
  (guaranteed lossless encoding of any input).
- **Unigram language model tokenisation** (alternative to BPE).
- **Multiprocessing** for pair-counting on large corpora.
- **Sentencepiece-compatible output** for interoperability.
- **Script-aware pre-tokenisation** to keep Bengali and Latin words separate.
- **Grapheme cluster awareness** using the `unicodedata` grapheme break algorithm.
- **Streaming training** for corpora too large to fit in memory.
- **Vocabulary pruning** to remove low-frequency subword units.

## References

1. Sennrich, R., Haddow, B., & Birch, A. (2016). _Neural Machine Translation
   of Rare Words with Subword Units_. ACL 2016.
   [arXiv:1508.07909](https://arxiv.org/abs/1508.07909)

2. Gage, P. (1994). _A New Algorithm for Data Compression_. C Users Journal.

3. Kudo, T., & Richardson, J. (2018). _SentencePiece: A simple and language
   independent subword tokenizer_. EMNLP 2018.

4. Radford, A., et al. (2019). _Language Models are Unsupervised Multitask
   Learners_. (GPT-2 byte-level BPE).

## License

MIT © 2026 Bengali BPE Tokenizer. See [LICENSE](LICENSE).
