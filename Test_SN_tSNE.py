# test_SN_tSNE.py
import os
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.cluster as cluster
from sklearn import svm
from sklearn.model_selection import train_test_split

from datasets import load_dataset
from Clustering_evaluation import Accuracy, NMI, SC, DB
from SN_tSNE import SNTSNE  # or import your own SN-tSNE class

def main():
    parser = argparse.ArgumentParser(description="Run SN-tSNE or PN-tSNE on multiple datasets.")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["isolet", "uci_har", "fashion_mnist", "usps_h5",
                                 "coil20", "olivetti", "digits"],
                        help="Dataset name to load.")
    parser.add_argument("--data-root", type=str, default="Datasets",
                        help="Root directory containing datasets.")
    parser.add_argument("--perplexity", type=float, default=3.0,
                        help="t-SNE perplexity value.")
    parser.add_argument("--n-iter", type=int, default=150,
                        help="Number of optimization iterations.")
    parser.add_argument("--learning-rate", type=float, default=200,
                        help="Learning rate for t-SNE optimization.")
    parser.add_argument("--plot-out", type=str, default=None,
                        help="File path to save output plot (PNG).")
    args = parser.parse_args()

    # --- Load dataset ---
    dataset_args = {}
    if args.dataset == "usps_h5":
        if args.usps_path is None:
            raise ValueError("Please provide --usps-path for USPS dataset.")
        dataset_args["path"] = args.usps_path
    elif args.dataset in ["isolet", "uci_har", "coil20"]:
        dataset_args["root"] = args.data_root

    X, y = load_dataset(args.dataset, **dataset_args)
    print(f"Loaded dataset: {args.dataset.upper()} | X={X.shape}, y={y.shape}, "
          f"Classes={len(np.unique(y))}")

    # --- Run SN-tSNE ---
    model = SNTSNE(
        n_components=2,
        perplexity=args.perplexity,
        n_iter=args.n_iter,
        early_exaggeration=12.0,
        init='pca',
        random_state=42,
        verbose=1
    )

    t0 = time.time()
    embedding, error, history = model.fit(X)
    runtime = time.time() - t0

    print(f"\nExecution Time: {runtime:.3f}s")
    print(f"Final Optimization Error: {error:.5f}")

    # --- Compute Non-Negative Ratio ---
    nn_ratio = np.sum(embedding >= 0) / embedding.size
    print(f"Non-Negative Points (NNP) Ratio: {nn_ratio:.4f}")

    # --- K-Means Clustering Evaluation ---
    n_clusters = len(np.unique(y))
    std = embedding.std(axis=0, keepdims=True) + 1e-8  # NEW
    embedding_km = embedding / std  # NEW
    kmeans = cluster.KMeans(n_clusters=n_clusters, n_init=50, random_state=42)
    y_pred = kmeans.fit_predict(embedding_km)

    print("\nK-Means Clustering Results:")
    print(f"  Accuracy     = {Accuracy(y, y_pred):.4f}")
    print(f"  NMI          = {NMI(y, y_pred):.4f}")
    print(f"  Silhouette   = {SC(embedding, y_pred):.4f}")
    print(f"  DBI          = {DB(embedding, y_pred):.4f}")

    # --- SVM Classification Evaluation ---
    X_train, X_test, y_train, y_test = train_test_split(embedding, y, test_size=0.25, random_state=42)
    clf = svm.SVC()
    clf.fit(X_train, y_train)
    y_pred_svm = clf.predict(X_test)

    print("\nSVM Classification Results:")
    print(f"  Accuracy     = {Accuracy(y_test, y_pred_svm):.4f}")

    # --- Visualization ---
    df = pd.DataFrame(embedding, columns=["x", "y"])
    df["target"] = y

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="x", y="y", hue="target",
                    palette=sns.color_palette("hsv", len(np.unique(y))),
                    s=15, linewidth=0)
    plt.title(f"SN-tSNE Embedding on {args.dataset.upper()}")
    plt.tight_layout()

    if args.plot_out:
        os.makedirs(os.path.dirname(args.plot_out), exist_ok=True)
        plt.savefig(args.plot_out, dpi=200)
        print(f"Plot saved at: {args.plot_out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
