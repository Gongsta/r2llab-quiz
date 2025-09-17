# Program Layout
- `download_dataset.py`: Downloads the dataset from the BEIR benchmark
- `sparse_retrieval.py`: Implements the sparse retrieval algorithm
- `dense_retrieval.py`: Implements the dense retrieval algorithm
- `evaluation.py`: Evaluates the retrieval systems

## Geting Started
Set up a virtual environment and install the dependencies:
```bash
pip install -r requirements.txt
```

Note that this installs the `faiss-cpu` package, which is a CPU-only version of the `faiss` library. If you want to use the GPU version, refer to the [faiss documentation](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md).


## Results
Sparse Retrieval (BM25):
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

Dense Retrieval (DPR):
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
- Ranking quality: The sparse retrieval system (BM25) has a slightly higher NDCG and MAP score, indicating that it is able to retrieve the most relevant documents at the top
- Precision and recall (retrieval coverage): The dense retrieval system (DPR) has a higher recall and precision score, indicating that it is able to retrieve more relevant documents overall
- Speed: BM25 is much faster since it doesn't use a pre-trained LM as the encoder unlike DPR.
- Memory: DPR required additional memory to store a dense matrix of embeddings (dim=384 per document), plus transient memory while encoding; BM25’s indices are light because they use
sparse vectors (inverted indices). At larger scales, DPR’s memory grows linearly with corpus size × embedding dim.
