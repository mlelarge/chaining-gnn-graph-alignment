"""
Generate reference test fixtures from the current working code.

Creates 10 test graph pairs with fixed seeds and records the outputs
of every key function. Future code changes must reproduce these outputs.

Usage:
    python -m tests.generate_fixtures

Outputs saved to: tests/fixtures/
"""

import os
import sys
import json
import torch
import numpy as np
import random

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

FIXTURES_DIR = os.path.join(PROJECT_ROOT, "tests", "fixtures")
N_EXAMPLES = 10
N_VERTICES = 50  # Small graphs for fast tests
EDGE_DENSITY = 0.1
NOISE = 0.2
SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_tensor(name, tensor):
    path = os.path.join(FIXTURES_DIR, f"{name}.pt")
    torch.save(tensor, path)
    print(f"  Saved {name}.pt  shape={getattr(tensor, 'shape', '?')}")


def save_numpy(name, arr):
    path = os.path.join(FIXTURES_DIR, f"{name}.npy")
    np.save(path, arr)
    print(f"  Saved {name}.npy  shape={arr.shape}")


def save_json(name, obj):
    path = os.path.join(FIXTURES_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"  Saved {name}.json")


def save_list_of_tuples(name, data):
    """Save a list of (tensor, tensor, tensor) tuples."""
    path = os.path.join(FIXTURES_DIR, f"{name}.pt")
    torch.save(data, path)
    n = len(data)
    shapes = [tuple(t.shape) for t in data[0]] if n > 0 else []
    print(f"  Saved {name}.pt  len={n}  item_shapes={shapes}")


# ─── 1. Graph Generation ─────────────────────────────────────────────

def generate_graph_fixtures():
    """Generate graph pairs using the data_generator functions directly."""
    from loaders.data_generator import (
        generate_erdos_renyi_netx,
        noise_erdos_renyi,
        adjacency_matrix_to_tensor_representation,
    )

    print("\n=== 1. Graph Generation ===")
    set_seed(SEED)

    graphs_A = []
    graphs_B = []
    adjacencies_W = []
    adjacencies_W_noise = []
    perms = []

    for i in range(N_EXAMPLES):
        g, W, _ = generate_erdos_renyi_netx(EDGE_DENSITY, N_VERTICES)
        W_noise = noise_erdos_renyi(g, W, NOISE, EDGE_DENSITY)

        B = adjacency_matrix_to_tensor_representation(W)
        B_noise = adjacency_matrix_to_tensor_representation(W_noise)

        # Create a random permutation (like the dataset does)
        perm = np.random.permutation(N_VERTICES)
        perm_mat = torch.zeros(N_VERTICES, N_VERTICES)
        for j in range(N_VERTICES):
            perm_mat[j, perm[j]] = 1.0

        graphs_A.append(B)
        graphs_B.append(B_noise)
        adjacencies_W.append(W)
        adjacencies_W_noise.append(W_noise)
        perms.append(perm_mat)

    graphs_A = torch.stack(graphs_A)
    graphs_B = torch.stack(graphs_B)
    adjacencies_W = torch.stack(adjacencies_W)
    adjacencies_W_noise = torch.stack(adjacencies_W_noise)
    perms = torch.stack(perms)

    save_tensor("graphs_A", graphs_A)
    save_tensor("graphs_B", graphs_B)
    save_tensor("adjacencies_W", adjacencies_W)
    save_tensor("adjacencies_W_noise", adjacencies_W_noise)
    save_tensor("permutations", perms)

    return graphs_A, graphs_B, adjacencies_W, perms


# ─── 2. Tensor Representation Functions ───────────────────────────────

def generate_tensor_repr_fixtures(adjacencies_W):
    """Test adjacency_matrix_to_tensor_representation and _ind variants."""
    from loaders.data_generator import (
        adjacency_matrix_to_tensor_representation,
        adjacency_matrix_to_tensor_representation_ind,
    )

    print("\n=== 2. Tensor Representation ===")
    set_seed(SEED + 1)

    # Standard representation
    repr_results = []
    for i in range(min(3, len(adjacencies_W))):
        B = adjacency_matrix_to_tensor_representation(adjacencies_W[i])
        repr_results.append(B)
    repr_results = torch.stack(repr_results)
    save_tensor("tensor_repr_standard", repr_results)

    # Representation with index relabeling
    repr_ind_results = []
    for i in range(min(3, len(adjacencies_W))):
        ind = np.random.permutation(N_VERTICES)
        # adjacency_matrix_to_tensor_representation_ind expects a 2-channel tensor
        W_2ch = torch.zeros(2, N_VERTICES, N_VERTICES)
        W_2ch[0] = adjacencies_W[i]
        B_ind = adjacency_matrix_to_tensor_representation_ind(W_2ch, ind)
        repr_ind_results.append(B_ind)
    repr_ind_results = torch.stack(repr_ind_results)
    save_tensor("tensor_repr_ind", repr_ind_results)

    return repr_results


# ─── 3. Data Relabeling (make_data_from_ind_label) ────────────────────

def generate_relabel_fixtures(graphs_A, graphs_B, perms):
    """Test the chaining relabeling step."""
    from loaders.data_generator import make_data_from_ind_label

    print("\n=== 3. Data Relabeling ===")
    set_seed(SEED + 2)

    # Create mock data in the format expected: list of (graph_A, graph_B, perm)
    data = [(graphs_A[i], graphs_B[i], perms[i]) for i in range(min(5, len(graphs_A)))]

    # Create mock index pairs (simulating model predictions)
    ind_pairs = []
    for _ in range(len(data)):
        ind1 = np.random.permutation(N_VERTICES)
        ind2 = np.random.permutation(N_VERTICES)
        ind_pairs.append((ind1, ind2))

    relabeled = make_data_from_ind_label(data, ind_pairs)
    save_list_of_tuples("relabeled_data", relabeled)

    # Also save the index pairs for reproducibility
    save_numpy("relabel_ind1", np.array([ip[0] for ip in ind_pairs]))
    save_numpy("relabel_ind2", np.array([ip[1] for ip in ind_pairs]))

    return relabeled


# ─── 4. Collate Function ─────────────────────────────────────────────

def generate_collate_fixtures(graphs_A, graphs_B, perms):
    """Test the DataLoader collate function."""
    from loaders import collate_fn

    print("\n=== 4. Collate Function ===")

    # Create a batch of 4 samples
    samples = [(graphs_A[i], graphs_B[i], perms[i]) for i in range(4)]
    batch = collate_fn(samples)

    save_tensor("collate_input1", batch[0]["input"])
    save_tensor("collate_input2", batch[1]["input"])
    save_tensor("collate_target", batch[2])


# ─── 5. Masking (no_seed) ────────────────────────────────────────────

def generate_masking_fixtures(graphs_A):
    """Test masking_noseed which zeros out the positional channel."""
    from loaders.load_utils import masking_noseed

    print("\n=== 5. Masking (no_seed) ===")

    # Clone so we don't modify the originals
    test_graph = graphs_A[0].clone()
    save_tensor("masking_before", test_graph.clone())
    masking_noseed(test_graph)
    save_tensor("masking_after", test_graph)


# ─── 6. Metrics Functions ────────────────────────────────────────────

def generate_metrics_fixtures():
    """Test accuracy_max, get_ranking, get_perm."""
    from toolbox.metrics import accuracy_max, get_ranking, get_perm

    print("\n=== 6. Metrics ===")
    set_seed(SEED + 3)

    n = N_VERTICES

    # accuracy_max: create synthetic scores and labels
    bs = 4
    # Scores where argmax along dim 1 gives identity perm (mostly)
    scores = torch.randn(bs, n, n)
    for i in range(bs):
        scores[i] += torch.eye(n) * 3.0  # bias toward diagonal
    labels = torch.eye(n).unsqueeze(0).expand(bs, -1, -1)

    acc, total = accuracy_max(scores, labels, aggregate_score=True)
    per_sample = accuracy_max(scores, labels, aggregate_score=False)

    save_tensor("metrics_scores", scores)
    save_json("metrics_accuracy", {
        "acc": int(acc),
        "total": int(total),
        "per_sample": per_sample,
    })

    # get_ranking: test with a single weight matrix and graphs
    weight = np.random.randn(n, n).astype(np.float64)
    g1 = (np.random.rand(n, n) > 0.8).astype(np.float64)
    g1 = np.triu(g1, 1)
    g1 = g1 + g1.T  # symmetric
    g2 = (np.random.rand(n, n) > 0.8).astype(np.float64)
    g2 = np.triu(g2, 1)
    g2 = g2 + g2.T

    row_ordering, col_ind = get_ranking(weight, g1, g2, use_faq=False)
    row_ordering_faq, col_ind_faq = get_ranking(weight, g1, g2, use_faq=True)

    save_numpy("metrics_weight", weight)
    save_numpy("metrics_g1", g1)
    save_numpy("metrics_g2", g2)
    save_numpy("metrics_ranking_order", row_ordering)
    save_numpy("metrics_ranking_col_ind", col_ind)
    save_numpy("metrics_ranking_order_faq", row_ordering_faq)
    save_numpy("metrics_ranking_col_ind_faq", col_ind_faq)

    # get_perm: test with synthetic index pairs
    ind0 = np.array([2, 0, 3, 1, 4])
    ind1 = np.array([4, 1, 0, 3, 2])
    perm = get_perm((ind0, ind1))
    save_numpy("metrics_get_perm_ind0", ind0)
    save_numpy("metrics_get_perm_ind1", ind1)
    save_numpy("metrics_get_perm_result", perm)


# ─── 7. Model Forward Pass ───────────────────────────────────────────

def generate_model_fixtures(graphs_A, graphs_B, perms):
    """Test model creation and forward pass with random weights."""
    from models import get_model, get_siamese
    from loaders import collate_fn

    print("\n=== 7. Model Forward Pass ===")
    set_seed(SEED + 4)

    # Use smaller model for test speed
    cfg_model = {
        "type": "node_embedding_node_pos",
        "block_inside": "block_res_mem",
        "num_blocks": 1,  # fewer blocks for speed
        "in_features": 32,  # smaller features for speed
        "depth_of_mlp": 1,
    }

    node_embedder = get_model(cfg_model)
    siamese = get_siamese(node_embedder)
    siamese.eval()

    # Create a small batch
    samples = [(graphs_A[i], graphs_B[i], perms[i]) for i in range(3)]
    batch = collate_fn(samples)

    with torch.no_grad():
        raw_scores = siamese(batch[0], batch[1])
        raw_scores_v, emb1, emb2 = siamese.forward_with_features(batch[0], batch[1])

    save_json("model_config", cfg_model)
    save_tensor("model_state_dict", siamese.state_dict())
    save_tensor("model_input1", batch[0]["input"])
    save_tensor("model_input2", batch[1]["input"])
    save_tensor("model_raw_scores", raw_scores)
    save_tensor("model_emb1", emb1)
    save_tensor("model_emb2", emb2)

    # Also test loss computation
    loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
    loss = loss_fn(raw_scores, batch[2])
    save_json("model_loss", {"loss": float(loss.item())})


# ─── 8. all_ind function ─────────────────────────────────────────────

def generate_all_ind_fixtures(graphs_A, graphs_B, perms):
    """Test the all_ind inference function."""
    from models import get_model, get_siamese
    from loaders import collate_fn, siamese_loader
    from loaders.data_generator import all_ind

    print("\n=== 8. all_ind Inference ===")
    set_seed(SEED + 5)

    cfg_model = {
        "type": "node_embedding_node_pos",
        "block_inside": "block_res_mem",
        "num_blocks": 1,
        "in_features": 32,
        "depth_of_mlp": 1,
    }

    node_embedder = get_model(cfg_model)
    siamese = get_siamese(node_embedder)
    siamese.eval()

    # Save model weights for reproducibility
    save_tensor("all_ind_model_state", siamese.state_dict())

    # Create dataset and loader
    data = [(graphs_A[i], graphs_B[i], perms[i]) for i in range(5)]
    loader = siamese_loader(data, batch_size=2, shuffle=False)

    device = "cpu"
    ind_data, nce, faq = all_ind(
        loader, siamese, device,
        compute_nce=True,
        compute_faq=True,
        verbose=False,
        size_seed=0,
    )

    # Save index pairs
    ind1_arr = np.array([ip[0] for ip in ind_data])
    ind2_arr = np.array([ip[1] for ip in ind_data])
    save_numpy("all_ind_ind1", ind1_arr)
    save_numpy("all_ind_ind2", ind2_arr)
    save_numpy("all_ind_nce", nce)
    if faq is not None:
        save_numpy("all_ind_faq", faq)


# ─── 9. Metadata ─────────────────────────────────────────────────────

def save_metadata():
    """Save generation parameters for documentation."""
    save_json("metadata", {
        "n_examples": N_EXAMPLES,
        "n_vertices": N_VERTICES,
        "edge_density": EDGE_DENSITY,
        "noise": NOISE,
        "seed": SEED,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
    })


# ─── Main ────────────────────────────────────────────────────────────

def main():
    os.makedirs(FIXTURES_DIR, exist_ok=True)
    print(f"Generating test fixtures in {FIXTURES_DIR}")

    save_metadata()

    graphs_A, graphs_B, adjacencies_W, perms = generate_graph_fixtures()
    generate_tensor_repr_fixtures(adjacencies_W)
    generate_relabel_fixtures(graphs_A, graphs_B, perms)
    generate_collate_fixtures(graphs_A, graphs_B, perms)
    generate_masking_fixtures(graphs_A)
    generate_metrics_fixtures()
    generate_model_fixtures(graphs_A, graphs_B, perms)
    generate_all_ind_fixtures(graphs_A, graphs_B, perms)

    print(f"\n=== Done! All fixtures saved to {FIXTURES_DIR} ===")
    print("Run 'python -m pytest tests/test_fixtures.py -v' to validate.")


if __name__ == "__main__":
    main()
