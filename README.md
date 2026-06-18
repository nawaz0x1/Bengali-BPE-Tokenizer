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

## Motivation: Better LLMs for Under-Resourced Languages

Languages like Bengali have over 230 million native speakers yet remain
**severely under-served** by existing LLM tokenizers. When an LLM trained
primarily on English text is applied to Bengali, the tokenizer's vocabulary
contains almost no Bengali subwords causing every word to fragment into
dozens of meaningless byte-level tokens. This:

- **Inflates sequence length**, filling context windows with noise.
- **Raises inference cost**, since every extra token costs compute.
- **Degrades model quality**, as the model must spend capacity reconstructing
  characters rather than learning language.

Training BPE directly on a large Bengali corpus gives the model a vocabulary
where real morphemes become single tokens, the same advantage English enjoys
by default.

## Training Corpus

The tokenizer was trained on the **CC-100 Bengali** dataset (`bn.txt`), a
monolingual corpus extracted from January–December 2018 CommonCrawl snapshots
using the [CC-Net](https://github.com/facebookresearch/cc_net) pipeline.

| Property | Value |
| -------- | ----- |
| Dataset | CC-100 Bengali (`bn.txt.xz`) |
| Source | [data.statmt.org/cc-100/](http://data.statmt.org/cc-100/) |
| Corpus size | 3,479,160,474 characters |
| Word count | 525,195,961 |
| Unique word types (≥ min\_freq) | 1,659,662 |
| Unicode codepoints seen | 6,116 |
| Bengali characters | 2,794,501,647 |
| Latin characters | 82,354,819 |

**Training run summary**

| Parameter | Value |
| --------- | ----- |
| Target vocabulary size | 8,000 |
| Character vocabulary size | 3,737 |
| Merge operations | 4,263 |
| Training time | 2 h 21 m 54 s |
| Merge throughput | ~1.05 merges / s |

> Dataset source: [data.statmt.org/cc-100/](http://data.statmt.org/cc-100/) — see [[5]](#references) for citation.

## Benchmark: Bengali BPE vs tiktoken

Measured on `examples/corpus.txt` (3,795 chars, 523 words, 427 Bengali words,
300 unique Bengali word types). Vocabulary trained on CC-100 Bengali (8,000 tokens).
Reproduce with:

```bash
pip install tiktoken
python benchmark.py
```

### Token count - full corpus (lower is better)

| Tokenizer | Vocab size | Tokens | vs Bengali-BPE |
| --------- | ---------: | -----: | -------------- |
| **Bengali-BPE (ours)** | **8,000** | **1,046** |  |
| tiktoken cl100k / GPT-4 | 100,277 | 4,712 | 4.50× more |
| tiktoken GPT-2 | 50,257 | 7,663 | 7.33× more |

### Compression ratio - chars per token (higher is better)

| Tokenizer | Chars / token |
| --------- | ------------: |
| **Bengali-BPE (ours)** | **3.63** |
| tiktoken cl100k / GPT-4 | 0.81 |
| tiktoken GPT-2 | 0.50 |

### Average tokens per Bengali word (lower is better)

| Tokenizer | Tokens / word | vs Bengali-BPE |
| --------- | ------------: | -------------- |
| **Bengali-BPE (ours)** | **2.03** |  |
| tiktoken cl100k / GPT-4 | 8.71 | 4.28× more |
| tiktoken GPT-2 | 13.65 | 6.71× more |

### Word-level spotlight - top 15 worst cases for GPT-2

| Bengali word | Bengali-BPE | GPT-2 | GPT-4 | Saving vs GPT-2 |
| ------------ | ----------: | ----: | ----: | --------------- |
| উল্লেখযোগ্যভাবে | **4** | 35 | 22 | **8.75× fewer** |
| প্রক্রিয়াকরণের | **5** | 33 | 20 | **6.60× fewer** |
| গণপ্রজাতন্ত্রী | **4** | 32 | 19 | **8.00× fewer** |
| মুক্তিযুদ্ধের | **2** | 31 | 18 | **15.50× fewer** |
| শিক্ষাব্যবস্থা | **2** | 31 | 17 | **15.50× fewer** |
| বিশ্ববিদ্যালয় | **1** | 30 | 18 | **30.00× fewer** |
| শীর্ষস্থানীয় | **3** | 30 | 19 | **10.00× fewer** |
| গুরুত্বপূর্ণ | **1** | 29 | 18 | **29.00× fewer** |
| বিশ্বসাহিত্যে | **4** | 29 | 15 | **7.25× fewer** |
| তাৎপর্যপূর্ণ | **4** | 28 | 17 | **7.00× fewer** |
| মর্যাদাপূর্ণ | **4** | 27 | 14 | **6.75× fewer** |
| সম্প্রদায়ের | **2** | 27 | 16 | **13.50× fewer** |
| উচ্চশিক্ষায় | **3** | 26 | 16 | **8.67× fewer** |
| রাষ্ট্রভাষার | **3** | 26 | 13 | **8.67× fewer** |
| ট্রান্সফরমার | **5** | 26 | 15 | **5.20× fewer** |

> **Takeaway:** Bengali-BPE uses **7.33× fewer tokens** than GPT-2 and **4.50× fewer**
> than GPT-4 on the full corpus. Per-word, it requires **6.71× fewer tokens** than
> GPT-2 and **4.28× fewer** than GPT-4. Common words like `গুরুত্বপূর্ণ` and
> `বিশ্ববিদ্যালয়` cost a single token instead of 29–30, directly reducing
> context window usage, inference cost, and sequence length for downstream models.

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
  BPE Inspection - 'বাংলাদেশ'
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

- `list("বাং")` → `['ব', 'া', 'ং']` - always correct for any script.
- No byte-splitting that would break multi-byte characters.
- All regex operations use `re.UNICODE` flag.

### Bengali Unicode block (U+0980-U+09FF)

| Code point     | Character | Description                                  |
| -------------- | --------- | -------------------------------------------- |
| U+09CD | ্ | Virama / Hasanta - suppresses inherent vowel |
| U+0982 | ং | Anusvara - nasalisation |
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
subword units - no special handling required.

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

5. Conneau, A., Khandelwal, K., Goyal, N., Chaudhary, V., Wenzek, G.,
   Guzmán, F., Grave, E., Ott, M., Zettlemoyer, L., & Stoyanov, V. (2020).
   _Unsupervised Cross-lingual Representation Learning at Scale_. ACL 2020.
   [arXiv:1911.02116](https://arxiv.org/abs/1911.02116)
   _(Source of the CC-100 Bengali training corpus.)_

6. Wenzek, G., Lachaux, M.-A., Conneau, A., Chaudhary, V., Guzmán, F.,
   Joulin, A., & Grave, E. (2020). _CCNet: Extracting High Quality Monolingual
   Datasets from Web Crawl Data_. LREC 2020.
   [arXiv:1911.00359](https://arxiv.org/abs/1911.00359)
   _(CC-Net pipeline used to build CC-100.)_

## License

MIT © 2026 Bengali BPE Tokenizer. See [LICENSE](LICENSE).
