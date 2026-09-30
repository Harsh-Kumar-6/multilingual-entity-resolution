*Multilingual Entity Resolution*

Scalable multilingual entity resolution using multi-pass blocking, fuzzy similarity features, and XGBoost.

It is an entity-matching pipeline designed to identify records that refer to the same real-world entity across multiple data sources.

The project combines **multilingual text normalization, heuristic blocking, fuzzy string matching, feature engineering, and gradient-boosted classification** into a memory-conscious pipeline that can work with large datasets.

---

## Pipeline

```text
Raw Data
   │
   ▼
Text Normalization
   │
   ├── Unicode normalization
   ├── Indic script transliteration
   ├── Accent removal
   ├── Stopword removal
   └── Text cleaning
   │
   ▼
Blocking
   │
   ├── Multiple heuristic blocking keys
   ├── Candidate generation
   ├── Frequency caps
   └── Duplicate removal
   │
   ▼
Feature Engineering
   │
   ├── Jaro-Winkler similarity
   ├── Levenshtein similarity
   ├── Token Jaccard similarity
   ├── Number overlap
   ├── Name length ratio
   └── Name subset features
   │
   ▼
XGBoost Classifier
   │
   ├── Group-aware validation
   ├── Early stopping
   └── Threshold tuning
   │
   ▼
Final Entity Matches
```

## Why Blocking?

Comparing every record against every other record quickly becomes impractical when working with millions of records.

Instead of performing an exhaustive comparison, our solution first generates a smaller set of **candidate pairs** using multiple heuristic blocking keys.

The blocking stage uses 15 different keys based on combinations of:

* Business names
* Address components
* Numbers
* Character patterns
* Word prefixes
* First/last tokens

Large blocks are capped to avoid creating huge candidate sets.

This significantly reduces the number of comparisons that need to reach the more expensive feature-engineering and classification stages.

---

## Multilingual Text Normalization

Entity names and addresses can appear in different scripts and formats.

The normalization pipeline handles several Indic scripts, including:

* Bengali
* Devanagari
* Gujarati
* Gurmukhi
* Kannada
* Malayalam
* Odia
* Tamil
* Telugu

The text-processing pipeline performs operations such as:

1. Unicode normalization
2. Indic-script transliteration
3. Lowercasing
4. Accent/diacritic removal
5. Entity/address stopword removal
6. Punctuation cleanup
7. Whitespace normalization

Normalized records are cached as pickle files so that expensive preprocessing does not need to be repeated unnecessarily.

---

## Feature Engineering

After candidate generation, each candidate pair is converted into a set of numerical similarity features.

### Name similarity

**Jaro-Winkler similarity**

Measures character-level similarity between the two names.

**Levenshtein/fuzzy similarity**

Captures general edit-based similarity between names.

**Token Jaccard similarity**

Measures the overlap between the token sets of the two names.

### Address similarity

A fuzzy string similarity is calculated between the normalized addresses.

### Numerical overlap

Numbers appearing in addresses are extracted and compared.


### Additional features

The model also receives:

* Name length ratio
* Name subset relationship
* Number overlap

The final feature set contains **7 features**:

```text
name_jw
name_lev
addr_lev
name_jaccard
num_overlap
name_len_ratio
name_subset
```

---

## Memory-Aware Processing

The pipeline is designed for large datasets rather than assuming that everything can comfortably fit into memory at once.

Several stages use:

* Multiprocessing
* Chunked processing
* Candidate frequency caps
* Integer hashing of entity IDs
* Temporary Parquet files
* Streaming candidate generation

Feature engineering processes candidates in chunks of **300,000 rows**, writes intermediate Parquet files, and combines them after processing.

This makes the pipeline more practical for datasets containing millions of records.

---

## Model

The classification stage uses **XGBoost** to determine whether a generated candidate pair represents the same entity.

The model uses:

```text
n_estimators = 300
max_depth = 7
learning_rate = 0.05
subsample = 0.8
colsample_bytree = 0.8
```

It uses the histogram-based tree method and early stopping.

### Group-aware validation

The train/validation split uses `GroupShuffleSplit`, with the source-1 entity ID as the grouping variable.

This prevents records belonging to the same source entity from being arbitrarily distributed between the training and validation sets.

---

## Threshold Optimization

Instead of automatically using `0.5` as the classification threshold, our solution evaluates several thresholds:

```text
0.10
0.30
0.50
0.60
0.70
0.80
0.85
0.90
0.95
0.98
0.99
```

For every threshold, the pipeline calculates:

* Accuracy
* Precision
* Recall
* F0.5 score

The threshold producing the highest validation **F0.5** score is stored with the model and reused during inference.

This separates the model's probability estimation from the final decision threshold.

---

## Inference

During inference:

1. The trained XGBoost model is loaded.
2. Test features are loaded.
3. Match probabilities are generated.
4. The saved optimal threshold is applied.
5. Predicted matches are grouped by source-1 entity.
6. Every source-1 entity is included in the final output.

The final result is written to:

```text
output/matching_results.tsv
```

The pipeline also applies a **singleton rule**, ensuring that source-1 entities with no predicted match are still present in the final output with an empty candidate field.

---

