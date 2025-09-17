# Program Layout
- `download_dataset.py`: Downloads the dataset from the BEIR benchmark
- `sparse_retrieval.py`: Implements the sparse retrieval algorithm
- `dense_retrieval.py`: Implements the dense retrieval algorithm
- `evaluation.py`: Evaluates the retrieval systems

## Results
Sparse Retrieval:
```
Evaluation Scores:
[
    {
        "NDCG@10": 0.65189,
        "NDCG@100": 0.67586
    },
    {
        "MAP@10": 0.60697,
        "MAP@100": 0.61318
    },
    {
        "Recall@10": 0.774,
        "Recall@100": 0.87306
    },
    {
        "P@10": 0.085,
        "P@100": 0.0098
    }
]
```

Dense Retrieval:
```
Evaluation Scores:
[
    {
        "NDCG@10": 0.64508,
        "NDCG@100": 0.67665
    },
    {
        "MAP@10": 0.59593,
        "MAP@100": 0.60307
    },
    {
        "Recall@10": 0.78333,
        "Recall@100": 0.925
    },
    {
        "P@10": 0.08833,
        "P@100": 0.01053
    }
]
```

### Results Analysis
- The sparse retrieval system (BM25) has a slightly higher NDCG and MAP score, indicating that it is able to retrieve the most relevant documents at the top (better ranking quality)
- However, the dense retrieval system (DPR) has a higher recall and precision score, indicating that it is able to retrieve more relevant documents overall (better retrieval coverage)
