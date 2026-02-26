from pathlib import Path
import numpy as np


PathLike = str | Path

def filter_by_label(dataset_vectors, label):
    return np.array([vec for vec in dataset_vectors if Path(vec["name"]).name.startswith(label)])


def load_img_embeddings(file_path):
    feat_array = np.load(file_path, allow_pickle=True)
    
    print(len(feat_array))
    if feat_array.dtype.kind != 'V':
        vector_len = len(feat_array)
        struct = np.dtype([('name', 'U100'), ('vector', 'f4', (vector_len,))])
        feat_array = np.array([(name, vec) for name, vec in feat_array], dtype=struct)

    return feat_array


def k_center_greedy_fast(X, k, seed=0):
    N, D = X.shape
    rng = np.random.default_rng(seed)

    centers = np.empty(k, dtype=np.int64)
    centers[0] = rng.integers(0, N)

    # distance^2 to nearest center
    dist = 2 - 2 * (X @ X[centers[0]])

    for i in range(1, k):
        centers[i] = np.argmax(dist)
        new_dist = 2 - 2 * (X @ X[centers[i]])
        dist = np.minimum(dist, new_dist)

    return centers

def k_center_per_class(embeddings, labels, k, seed=0):
    """
    embeddings: np.ndarray (N, D)
    labels: np.ndarray (N,)
    k: centers per class

    Returns:
        dict[class_id] -> indices into original array
    """
    selected = {}
    unique_classes = np.unique(labels)

    for cls in unique_classes:
        idx = np.where(labels == cls)[0]
        X_cls = embeddings[idx]

        assert len(X_cls) >= k, f"Class {cls} has < k samples"

        centers_local = k_center_greedy_fast(X_cls, k, seed=seed)
        selected[cls] = idx[centers_local]

    return selected


def k_sampling(path, k=50):

    embeddings = load_img_embeddings(path)
    labels = ["ADI","DEB","LYM","MUC","MUS","NORM","STR","TUM"]
    
    selected_img = {}

    for i in labels:
        class_embeddings = filter_by_label(embeddings, i)
        class_embeddings_images = class_embeddings["vector"]
        class_embeddings_images /= np.linalg.norm(class_embeddings_images, axis=1, keepdims=True)

        
        selected_indices = k_center_greedy_fast(class_embeddings_images, k, seed=0)
        selected_img[i]= [class_embeddings[idx]["name"] for idx in selected_indices]


    return selected_img

def main():
    path = "test"
    k = 50
    selected_img = k_sampling(path, k)
    print(selected_img)
if __name__ == "__main__":
    main()