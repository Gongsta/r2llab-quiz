import argparse
from typing import Dict, List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import re

from utils import load_scifact_split, save_results


def build_bm25_index(corpus: Dict[str, Dict[str, str]]) -> Tuple[BM25Okapi, List[str]]:
    doc_ids: List[str] = []
    tokenized_docs: List[List[str]] = []

    for doc_id, doc in corpus.items():
        content = doc["title"] + " " + doc["text"]
        tokens = re.findall(r"\w+", content.lower())   # note that content.split() works worse than the regex, since it counts punctuation as part of the word
        doc_ids.append(doc_id)
        tokenized_docs.append(tokens)

    bm25 = BM25Okapi(tokenized_docs)
    return bm25, doc_ids


def retrieve_top_k(
    bm25: BM25Okapi,
    doc_ids: List[str],
    queries: Dict[str, str],
    top_k: int = 100,
) -> Dict[str, Dict[str, float]]:
    """
    Retrieve the top-k documents for each query using the BM25 index
    """
    results: Dict[str, Dict[str, float]] = {}
    for qid, qtext in tqdm(queries.items(), desc="Retrieving", unit="query"):
        q_tokens = re.findall(r"\w+", qtext.lower())
        scores = bm25.get_scores(q_tokens)
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        results[qid] = {doc_ids[i]: float(scores[i]) for i in top_indices}
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sparse retrieval with BM25 on BEIR SciFact. "
            "Loads the test split queries, indexes the corpus, retrieves top-k, and saves JSON results."
        )
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="datasets/scifact",
        help="Path to the BEIR dataset folder, e.g., datasets/scifact",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="results/sparse_results.json",
        help="Path to write results JSON, e.g., results/sparse_results.json",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "dev", "test"],
        help="Dataset split to use for queries (default: test)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=100,
        help="Number of documents to retrieve per query (default: 100)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load scifact
    corpus, queries = load_scifact_split(args.dataset_path, split=args.split)

    # Build the BM25 index and get the list of document IDs
    bm25, doc_ids = build_bm25_index(corpus)

    # Retrieve the top-k documents for each query
    results = retrieve_top_k(bm25, doc_ids, queries, top_k=args.top_k)

    # Save retrieval results to JSON
    save_results(results, args.output_path)
    print(f"Saved results to {args.output_path}")


if __name__ == "__main__":
    main()
