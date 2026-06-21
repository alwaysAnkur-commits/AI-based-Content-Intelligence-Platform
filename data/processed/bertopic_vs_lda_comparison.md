# BERTopic vs LDA Comparison

BERTopic discovered 2 topics (excluding outlier topic -1).
LDA discovered 11 topics.

## Key Differences
- BERTopic uses dense embeddings + clustering (HDBSCAN), capturing semantic similarity directly.
- LDA uses bag-of-words probabilistic modeling — purely word co-occurrence statistics.
- BERTopic can leave documents unassigned (-1 'outlier' topic); LDA always assigns every document.
- BERTopic topics tend to be more semantically coherent for short/noisy text.