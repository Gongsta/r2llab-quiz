from beir.datasets.data_loader import GenericDataLoader
from typing import Dict, Tuple
import json
import os
import re


def load_scifact_split(
    dataset_path: str, split: str = "test"
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str]]:
    corpus, queries, _ = GenericDataLoader(data_folder=dataset_path).load(split=split)
    return corpus, queries


def save_results(results: Dict[str, Dict[str, float]], output_path: str) -> None:
    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f)
