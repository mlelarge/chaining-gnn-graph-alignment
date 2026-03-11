"""Self-contained inference script that downloads pre-trained GNNs from GitHub
releases and runs the chaining inference loop on synthetic graph pairs.

Usage:
    python run_inference.py --release v1.0.0-er500-d4-pn0.22
    python run_inference.py --release v1.0.0-er500-d80-pn0.24 --noise 0.3
    python run_inference.py --release v1.0.0-reg500-d10-pn0.11 --num_examples 50

Available releases:
    v1.0.0-er500-d4-pn0.22   Sparse Erdos-Renyi (500 nodes, avg degree 4, noise 0.22)
    v1.0.0-er500-d80-pn0.24  Dense Erdos-Renyi (500 nodes, avg degree 80, noise 0.24)
    v1.0.0-reg500-d10-pn0.11 Regular graphs (500 nodes, degree 10, noise 0.11)
"""

import argparse
import json
import os
import urllib.request

from omegaconf import OmegaConf

from models.pipeline import Chaining

REPO = "mlelarge/chaining-gnn-graph-alignment"
API_URL = f"https://api.github.com/repos/{REPO}/releases/tags"
DOWNLOAD_URL = f"https://github.com/{REPO}/releases/download"


def get_release_assets(tag: str) -> list[str]:
    """Fetch the list of asset filenames for a given release tag."""
    url = f"{API_URL}/{tag}"
    req = urllib.request.Request(
        url, headers={"Accept": "application/vnd.github.v3+json"}
    )
    with urllib.request.urlopen(req) as resp:
        release = json.loads(resp.read().decode())
    return [asset["name"] for asset in release["assets"]]


def _local_models_ready(path_models: str) -> bool:
    """Check if checkpoints and config.json already exist locally."""
    if not os.path.isdir(path_models):
        return False
    files = os.listdir(path_models)
    has_config = "config.json" in files
    has_ckpt = any(f.endswith(".ckpt") for f in files)
    return has_config and has_ckpt


def download_release(tag: str, checkpoint_dir: str) -> str:
    """Download all assets for a release into checkpoint_dir/<tag>/.

    Skips the download entirely if checkpoints and config.json already
    exist locally (useful for offline/cluster environments).
    Returns the path to the directory containing the checkpoints.
    """
    path_models = os.path.join(checkpoint_dir, tag)

    if _local_models_ready(path_models):
        print(f"Models already available in {path_models}")
        return path_models

    os.makedirs(path_models, exist_ok=True)

    assets = get_release_assets(tag)
    relevant = [a for a in assets if a.endswith(".ckpt") or a == "config.json"]

    for filename in relevant:
        dest = os.path.join(path_models, filename)
        if os.path.exists(dest):
            print(f"  [cached] {filename}")
            continue
        url = f"{DOWNLOAD_URL}/{tag}/{filename}"
        print(f"  Downloading {filename} ...")
        urllib.request.urlretrieve(url, dest)

    print(f"Models ready in {path_models}")
    return path_models


def build_dataset_config(
    config: dict, noise: float | None, num_examples: int
) -> OmegaConf:
    """Build a minimal OmegaConf dataset config from the saved config.json."""
    ds = config["dataset"]
    cfg = OmegaConf.create(
        {
            "type": ds.get("type", "synthetic"),
            "n_vertices": ds["n_vertices"],
            "generative_model": ds["generative_model"],
            "noise_model": ds["noise_model"],
            "edge_density": ds["edge_density"],
            "noise": noise if noise is not None else ds["noise"],
            "test": {"num_examples": num_examples},
        }
    )
    return cfg


def main():
    parser = argparse.ArgumentParser(
        description="Download pre-trained GNNs and run chaining inference."
    )
    parser.add_argument(
        "--release",
        required=True,
        help="GitHub release tag (e.g. v1.0.0-er500-d4-pn0.22)",
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=None,
        help="Override noise level (default: from config.json)",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=100,
        help="Number of test examples to generate (default: 100)",
    )
    parser.add_argument(
        "--L",
        type=int,
        default=None,
        help="Max number of chaining models to use (default: all available)",
    )
    parser.add_argument(
        "--N_max",
        type=int,
        default=80,
        help="Max refinement iterations with best model (default: 80)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        default="./checkpoints",
        help="Directory to cache downloaded models (default: ./checkpoints)",
    )
    parser.add_argument(
        "--data_dir",
        default="./data",
        help="Directory to cache generated datasets (default: ./data)",
    )
    args = parser.parse_args()

    # 1. Download models
    print(f"Fetching release {args.release} ...")
    path_models = download_release(args.release, args.checkpoint_dir)

    # 2. Build dataset config from the release's config.json
    with open(os.path.join(path_models, "config.json")) as f:
        config = json.load(f)
    cfg_data = build_dataset_config(config, args.noise, args.num_examples)
    noise_used = cfg_data.noise
    print(
        f"Dataset: {cfg_data.generative_model}, n={cfg_data.n_vertices}, "
        f"density={cfg_data.edge_density}, noise={noise_used}, "
        f"examples={args.num_examples}"
    )

    # 3. Run chaining inference (data cached in --data_dir for reuse)
    chain = Chaining(path_models)
    data_dir = args.data_dir
    os.makedirs(data_dir, exist_ok=True)

    print("\nRunning chaining inference ...")
    result = chain.loop(cfg_data, data_dir, L=args.L, N_max=args.N_max)

    # 4. Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Release:         {args.release}")
    print(f"Noise:           {noise_used}")
    print(f"Test examples:   {args.num_examples}")
    print(f"Best chain iter: {result.best_nloop}")
    print(f"QAP score (avg): {result.all_qap.mean():.4f}")
    print("=" * 60)
    print(f"\nDataset cached in: {data_dir}")
    print(f"Run FAQ baseline with: python run_baseline.py --data_dir {data_dir} --release {args.release}")


if __name__ == "__main__":
    main()
