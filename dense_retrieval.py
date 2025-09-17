import argparse
from typing import Dict, List, Tuple

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from utils import load_scifact_split, save_results


def encode_corpus(
    model: SentenceTransformer,
    corpus: Dict[str, Dict[str, str]],
    batch_size: int = 128,
) -> Tuple[np.ndarray, List[str]]:
    doc_ids: List[str] = []
    texts: List[str] = []

    for doc_id, doc in corpus.items():
        content = doc["title"] + " " + doc["text"]
        doc_ids.append(doc_id)
        texts.append(content)

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    embeddings = embeddings.astype(np.float32, copy=False)
    return embeddings, doc_ids


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def retrieve_top_k(
    index: faiss.Index,
    doc_ids: List[str],
    queries: Dict[str, str],
    model: SentenceTransformer,
    top_k: int = 100,
    batch_size: int = 128,
) -> Dict[str, Dict[str, float]]:
    if len(doc_ids) == 0:
        return {qid: {} for qid in queries.keys()}

    qids: List[str] = []
    qtexts: List[str] = []
    for qid, qtext in queries.items():
        qids.append(qid)
        qtexts.append(qtext or "")

    q_embeddings = model.encode(
        qtexts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32, copy=False)

    all_scores: Dict[str, Dict[str, float]] = {}
    # Process in batches
    batch_start = 0
    pbar = tqdm(total=len(qids), desc="Retrieving", unit="query")
    while batch_start < len(qids):
        batch_end = min(batch_start + batch_size, len(qids))
        batch_emb = q_embeddings[batch_start:batch_end]
        scores, indices = index.search(batch_emb, top_k)
        for row, qidx in enumerate(range(batch_start, batch_end)):
            qid = qids[qidx]
            doc_scores: Dict[str, float] = {}
            for rank, doc_index in enumerate(indices[row]):
                doc_id = doc_ids[doc_index]
                doc_scores[doc_id] = float(scores[row][rank])
            all_scores[qid] = doc_scores
        pbar.update(batch_end - batch_start)
        batch_start = batch_end
    pbar.close()

    return all_scores


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    # Prefer MPS on Apple Silicon if available
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Dense retrieval with Sentence-Transformers + FAISS on BEIR SciFact. "
            "Loads the split queries, encodes corpus, retrieves top-k, and saves JSON results."
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
        default="results/dense_results.json",
        help="Path to write results JSON, e.g., results/dense_results.json",
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
    parser.add_argument(
        "--model_name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help=(
            "Sentence-Transformers model to use (default: sentence-transformers/all-MiniLM-L6-v2)"
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for encoding queries and documents (default: 128)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = _detect_device()

    # Load scifact
    corpus, queries = load_scifact_split(args.dataset_path, split=args.split)

    # Load model
    model = SentenceTransformer(args.model_name, device=device)

    # Encode corpus
    corpus_embeddings, doc_ids = encode_corpus(
        model, corpus, batch_size=args.batch_size
    )
    print(
        f"Encoded {len(doc_ids)} documents. Embedding dim = {corpus_embeddings.shape[1]}"
    )

    # Build FAISS index
    index = build_faiss_index(corpus_embeddings)

    # Retrieve top-k documents for each query
    results = retrieve_top_k(
        index,
        doc_ids,
        queries,
        model,
        top_k=args.top_k,
        batch_size=args.batch_size,
    )

    # Save results to JSON
    save_results(results, args.output_path)
    print(f"Saved results to {args.output_path}")


if __name__ == "__main__":
    main()
