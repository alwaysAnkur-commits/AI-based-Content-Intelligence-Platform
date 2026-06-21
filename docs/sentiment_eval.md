# Sentiment Model Evaluation

Test set: 2 docs

| model      |   accuracy |   weighted_f1 |   avg_inference_time_ms |
|:-----------|-----------:|--------------:|------------------------:|
| VADER      |        0.5 |         0.333 |                    0.08 |
| TextBlob   |        0.5 |         0.333 |                   43.6  |
| DistilBERT |        1   |         1     |                   48.3  |