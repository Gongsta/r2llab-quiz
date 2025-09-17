import argparse
import shutil
from pathlib import Path

from beir import util
from beir.datasets.data_loader import GenericDataLoader

SCIFACT_URL = (
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
)


def download_dataset(output_dir: Path, force: bool = False) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = output_dir / "scifact"

    if dataset_dir.exists() and not force:
        print(
            f"Dataset already exists at {dataset_dir}. Skipping download. Use --force to re-download."
        )
        return dataset_dir

    if dataset_dir.exists():
        print(f"Removing existing dataset directory: {dataset_dir}")
        shutil.rmtree(dataset_dir)

    print(
        f"Downloading and extracting 'scifact' from {SCIFACT_URL} into {output_dir} ..."
    )
    util.download_and_unzip(SCIFACT_URL, str(output_dir))
    print(f"Done. Dataset available at: {dataset_dir}")
    return dataset_dir


def verify_dataset(dataset_dir: Path) -> None:
    # Will raise an error if the dataset is malformed or missing files
    print(
        f"Verifying dataset at {dataset_dir} by loading the 'train' and 'test' split ..."
    )
    GenericDataLoader(data_folder=str(dataset_dir)).load(split="train")
    GenericDataLoader(data_folder=str(dataset_dir)).load(split="test")
    print("Verification successful: dataset can be loaded.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the Scifact dataset to an output directory."
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="datasets",
        help="Directory where the dataset folder will be placed (default: datasets).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and overwrite existing dataset directory if it exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = download_dataset(Path(args.out_dir), force=args.force)
    verify_dataset(dataset_dir)


if __name__ == "__main__":
    main()
