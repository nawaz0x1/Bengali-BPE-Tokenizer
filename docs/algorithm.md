# BPE Algorithm Implementation Details

This document explains the Byte Pair Encoding algorithm as implemented in
this repository, with special attention to Bengali (Bangla) and other
non-Latin scripts.

## 1. What is Byte Pair Encoding?

Byte Pair Encoding (BPE) is a **data compression** algorithm originally
described by Gage (1994) and adapted for neural machine translation
subword tokenisation by Sennrich, Haddow & Birch (2016).

The core idea: **repeatedly merge the most frequent adjacent symbol pair**
until the vocabulary reaches a target size.

## 2. Algorithm Steps

### Phase 1 - Pre-tokenisation

Before BPE is applied, the corpus is split into **pre-tokens** (roughly
"words"). This is done by splitting at whitespace and punctuation.

For Bengali:

- Split at ASCII whitespace, Bengali danda `।` (U+0964), double danda `॥` (U+0965).
- Keep Bengali digits `০–৯` as separate units.

### Phase 2 - Initial vocabulary

Each word is represented as a **sequence of Unicode code points** plus an
end-of-word marker `</w>`.

```
"বাংলা" → ('ব', 'া', 'ং', 'ল', 'া', '</w>')
```

All unique characters across the corpus form the initial vocabulary.

### Phase 3 - Iterative merging

Repeat until target vocabulary size is reached:

1. **Count pair frequencies**: For every word's symbol sequence, count how
   often each adjacent pair appears (weighted by word frequency).

2. **Select best pair**: The most frequent pair wins. Ties are broken
   lexicographically for determinism.

3. **Merge**: Replace every occurrence of `(A, B)` in all word sequences
   with `AB`.

4. **Update pair counts incrementally**: Only the pair counts affected by
   the merge are updated - O(k·C) per step.

5. **Add to vocabulary**: The new symbol `AB` is added to the vocabulary.

### Phase 4 - Save model

The trained model is saved as:

- `vocab.json` - token → ID mapping
- `merges.txt` - ordered merge rules (tab-separated)
- `metadata.json` - training statistics

## 3. Encoding New Text

Given a trained model, encoding a word uses the **greedy merge** algorithm:

1. Split the word into characters + `</w>`.
2. Find the pair with the **lowest merge rank** (i.e. the pair that was
   merged earliest during training).
3. Merge **all** occurrences of that pair.
4. Repeat until no ranked pair remains.
5. Look up each final symbol in the vocabulary → integer ID.

This greedy algorithm guarantees that the encoding is consistent with the
training procedure.

## 4. Decoding

Decoding is simple:

1. Map each integer ID → token string.
2. Concatenate all tokens.
3. Replace `</w>` with a space.

## 5. Bengali-Specific Considerations

### Unicode code points, not bytes

This Tokenizer **never** operates on raw UTF-8 bytes. Bengali characters
span 2–3 bytes in UTF-8 encoding, so byte-level operations would produce
nonsensical sub-character tokens. Python's `str` type is a sequence of
Unicode code points, so `list(text)` always gives the correct character split.

### NFC normalisation

Bengali has composed and decomposed representations for certain vowel signs.
For example, the vowel `ো` (U+09CB) can be represented as a single code point
**or** as `া` (U+09BE) + `ে` (U+09CB). NFC normalisation ensures a canonical
form before training.

### Virama and conjuncts

The virama `্` (U+09CD) suppresses the inherent vowel of a consonant and forms
conjuncts:

```
ক + ্ + ষ = ক্ষ  (three code points, one visual unit)
```

This Tokenizer treats the virama as an atomic code point. BPE will
naturally learn to merge common conjunct sequences into single tokens.

### Zero-width characters

ZWNJ (U+200C) prevents conjunct formation. ZWJ (U+200D) forces it. The
Tokenizer preserves these characters by default (removing them changes
orthographic meaning).

## 6. Complexity Analysis

| Operation          | Time complexity | Notes                            |
| ------------------ | --------------- | -------------------------------- |
| Build word freq    | O(N)            | N = corpus characters            |
| Initial pair count | O(W·L)          | W = unique words, L = avg length |
| Each merge step    | O(C·L)          | C = words containing the pair    |
| Total training     | O(V·W·L)        | V = number of merges             |
| Encoding one word  | O(L²)           | L = word length                  |

In practice, for Bengali (L ≈ 10, W ≈ 50K, V ≈ 16K), training takes
seconds to a few minutes.

## 7. References

1. Sennrich, R., Haddow, B., & Birch, A. (2016). _Neural Machine Translation
   of Rare Words with Subword Units_. ACL 2016.
   [arXiv:1508.07909](https://arxiv.org/abs/1508.07909)

2. Gage, P. (1994). _A New Algorithm for Data Compression_. C Users Journal.

3. Kudo, T., & Richardson, J. (2018). _SentencePiece: A simple and language
   independent subword tokenizer and detokenizer for Neural Text Processing_.
   EMNLP 2018. [arXiv:1808.06226](https://arxiv.org/abs/1808.06226)

4. Radford, A., et al. (2019). _Language Models are Unsupervised Multitask
   Learners_. (GPT-2 paper - describes byte-level BPE).
