"""Three balanced K-means clusters within each synthesis type -> nine experts."""

import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans


def balanced_labels(features, seed):
    kmeans = KMeans(n_clusters=3, n_init=10, random_state=seed).fit(features)
    distances = kmeans.transform(features)
    # Greedy nearest-center assignment subject to floor/ceil capacities.
    capacity = np.full(3, len(features) // 3, dtype=int)
    capacity[: len(features) % 3] += 1
    labels = np.empty(len(features), dtype=int)
    for index in np.argsort(distances.min(1), kind="stable"):
        for cluster in np.argsort(distances[index]):
            if capacity[cluster] > 0:
                labels[index] = cluster
                capacity[cluster] -= 1
                break
    return labels, kmeans.cluster_centers_


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embeddings", required=True, help="NPZ from extract_queries.py"
    )
    parser.add_argument("--output", default="data/expert_assignments.json")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    data = np.load(args.embeddings, allow_pickle=False)
    indices, features, types = data["indices"], data["embeddings"], data["types"]
    if len(set(indices.tolist())) != len(indices):
        raise ValueError("Embedding indices contain duplicates")
    if set(types.tolist()) != {"GE", "CM", "SP"}:
        raise ValueError("Expected all three forgery synthesis types")
    assignments, centers = {}, {}
    for type_id, name in enumerate(["GE", "CM", "SP"]):
        chosen = np.flatnonzero(types == name)
        labels, center = balanced_labels(features[chosen], args.seed)
        centers[name] = center
        for index, label in zip(indices[chosen], labels):
            assignments[str(int(index))] = int(type_id * 3 + label)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(assignments, indent=2))
    np.savez(output.with_suffix(".centers.npz"), **centers)
    print(f"Saved {len(assignments)} fixed expert assignments to {output}")


if __name__ == "__main__":
    main()
