#!/usr/bin/env python
"""
Pancreas benchmark illustrating ACPCA vs PCA on a real single-cell dataset.

This script expects the SCIB human pancreas integration dataset at
data/human_pancreas_norm_complexBatch.h5ad (downloaded separately).
"""

import csv
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

from collections import Counter

import numpy as np
from matplotlib.patches import Ellipse
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler, quantile_transform

from acpca import ACPCA


def _get_pyplot():
    """Load matplotlib.pyplot using the Agg backend if not already imported."""
    import matplotlib

    if "matplotlib.pyplot" not in sys.modules:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def pretty_method(name: str) -> str:
    if name == "PCA":
        return "PCA"
    if name == "QuantileNorm_PCA":
        return "Quantile-normalized PCA"
    if name.startswith("ACPCA_lambda_"):
        return f"ACPCA (λ={name.split('_lambda_')[1]})"
    return name


def cluster_analysis(embedding, true_labels, n_clusters, cluster_dims):
    """Run KMeans clustering and compute agreement metrics and per-cluster summaries."""
    coords = embedding[:, :cluster_dims]
    coords2d = embedding[:, :2]
    model = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
    assignments = model.fit_predict(coords)
    ari = adjusted_rand_score(true_labels, assignments)
    nmi = normalized_mutual_info_score(true_labels, assignments)

    details = []
    for cluster_id in range(n_clusters):
        mask = assignments == cluster_id
        if not np.any(mask):
            continue
        cluster_points = coords2d[mask]
        centroid = model.cluster_centers_[cluster_id][:2]
        label_counter = Counter(true_labels[mask])
        top_label, top_count = label_counter.most_common(1)[0]
        frac = top_count / mask.sum()
        if cluster_points.shape[0] > 1:
            covariance = np.cov(cluster_points, rowvar=False)
        else:
            covariance = np.eye(2) * 1e-6
        details.append(
            {
                "cluster_id": cluster_id,
                "centroid": centroid,
                "size": int(mask.sum()),
                "top_label": top_label,
                "top_fraction": frac,
                "covariance": covariance,
            }
        )

    return {
        "coords2d": coords2d,
        "assignments": assignments,
        "centroids": model.cluster_centers_[:, :2],
        "ari": ari,
        "nmi": nmi,
        "details": details,
    }


def render_cluster_panel(
    ax,
    result,
    label_title,
    n_labels,
    ellipse_scale=2.5,
    levels=18,
):
    """Render a single cluster panel with density contours and centroid annotations."""
    plt = _get_pyplot()
    coords2d = result["coords2d"]
    assignments = result["assignments"]

    x_min, x_max = coords2d[:, 0].min(), coords2d[:, 0].max()
    y_min, y_max = coords2d[:, 1].min(), coords2d[:, 1].max()
    x_pad = max(1e-6, 0.08 * (x_max - x_min))
    y_pad = max(1e-6, 0.08 * (y_max - y_min))
    x_grid = np.linspace(x_min - x_pad, x_max + x_pad, 160)
    y_grid = np.linspace(y_min - y_pad, y_max + y_pad, 160)
    xx, yy = np.meshgrid(x_grid, y_grid)
    grid_stack = np.vstack([xx.ravel(), yy.ravel()])

    try:
        kde = gaussian_kde(coords2d.T)
        density = kde(grid_stack).reshape(xx.shape)
        ax.contourf(
            xx,
            yy,
            density,
            levels=levels,
            cmap="Blues",
            alpha=0.55,
        )
    except np.linalg.LinAlgError:
        pass

    cmap = plt.get_cmap("tab20", n_labels)
    ax.scatter(
        coords2d[:, 0],
        coords2d[:, 1],
        c=assignments,
        cmap=cmap,
        s=12,
        alpha=0.6,
        linewidths=0,
    )

    offset_x = 0.015 * (x_max - x_min + 1e-6)
    offset_y = 0.015 * (y_max - y_min + 1e-6)
    for detail in result["details"]:
        centroid = detail["centroid"]
        cluster_id = detail["cluster_id"]
        color = cmap(cluster_id)
        ax.scatter(
            centroid[0],
            centroid[1],
            s=160,
            marker="X",
            c=[color],
            edgecolor="black",
            linewidth=0.6,
            zorder=5,
        )

        cov = detail["covariance"]
        try:
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = np.maximum(eigvals, 1e-6)
            order = np.argsort(eigvals)[::-1]
            eigvals = eigvals[order]
            eigvecs = eigvecs[:, order]
            angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
            width = 2 * ellipse_scale * np.sqrt(eigvals[0])
            height = 2 * ellipse_scale * np.sqrt(eigvals[1])
            ellipse = Ellipse(
                centroid,
                width=width,
                height=height,
                angle=angle,
                edgecolor=color,
                facecolor="none",
                linewidth=1.4,
                linestyle="--",
                alpha=0.9,
            )
            ax.add_patch(ellipse)
        except np.linalg.LinAlgError:
            pass

        ax.text(
            centroid[0] + offset_x,
            centroid[1] + offset_y,
            f"{detail['top_label']}\n{detail['top_fraction']*100:.0f}%",
            fontsize=8,
            ha="left",
            va="bottom",
            color="black",
            bbox=dict(
                boxstyle="round,pad=0.25",
                fc="white",
                ec=color,
                lw=0.6,
                alpha=0.85,
            ),
            zorder=6,
        )

    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_aspect("equal")
    ax.text(
        0.02,
        0.98,
        f"ARI {result['ari']:.2f}\nNMI {result['nmi']:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.85),
    )


def read_pancreas(path: Path):
    """Load the h5ad file while patching NumPy 2.0 string removal."""
    import anndata as ad  # Imported lazily to keep CLI fast.

    if not hasattr(np, "string_"):
        np.string_ = np.bytes_  # type: ignore[attr-defined]

    return ad.read_h5ad(path)


def main():
    data_path = Path("data/human_pancreas_norm_complexBatch.h5ad")
    if not data_path.exists():
        raise FileNotFoundError(
            "Expected dataset at data/human_pancreas_norm_complexBatch.h5ad. "
            "Download the SCIB pancreas benchmark before running this script."
        )

    rng = np.random.default_rng(42)
    adata = read_pancreas(data_path)

    X = adata.X.astype(np.float32, copy=False)
    tech = adata.obs["tech"].to_numpy().astype(str)
    celltype = adata.obs["celltype"].to_numpy().astype(str)
    print(f"Loaded matrix: {X.shape[0]} cells × {X.shape[1]} genes")

    target_cells = 1200
    if X.shape[0] > target_cells:
        subset_idx = rng.choice(X.shape[0], target_cells, replace=False)
        X = X[subset_idx]
        tech = tech[subset_idx]
        celltype = celltype[subset_idx]
    print(f"Subset to {X.shape[0]} cells for benchmarking")

    target_genes = 800
    if X.shape[1] > target_genes:
        variances = X.var(axis=0)
        top_idx = np.argpartition(variances, -target_genes)[-target_genes:]
        X = X[:, top_idx]
    print(f"Selected {X.shape[1]} high-variance genes")

    scaler = StandardScaler(with_mean=True, with_std=False)
    X_centered = scaler.fit_transform(X)
    print("Centered expression matrix")

    metric_dims = 4
    n_components = 6
    lambda_grid = [0.0, 0.4, 0.8]

    pca = PCA(n_components=n_components, random_state=42)
    pca_coords = pca.fit_transform(X_centered)
    pca_times = []
    repetitions = 1
    for _ in range(repetitions):
        start = time.perf_counter()
        pca.fit_transform(X_centered)
        pca_times.append(time.perf_counter() - start)
    print("Computed PCA baseline")

    # Quantile batch normalization baseline: map each batch to a shared quantile distribution.
    X_quantile = X_centered.copy()
    unique_batches = np.unique(tech)
    for level in unique_batches:
        mask = tech == level
        if mask.sum() < 2:
            continue
        X_quantile[mask] = quantile_transform(
            X_quantile[mask],
            axis=0,
            n_quantiles=min(1000, mask.sum()),
            output_distribution="normal",
            copy=True,
            random_state=42,
        )
    qn_pca = PCA(n_components=n_components, random_state=42)
    qn_pca_coords = qn_pca.fit_transform(X_quantile)
    print("Computed quantile-normalized PCA baseline")

    tech_encoder = LabelEncoder()
    tech_encoded = tech_encoder.fit_transform(tech)

    acpca_results = []
    best = None
    best_coords = None

    print("Scanning ACPCA lambdas:", lambda_grid)
    for lam in lambda_grid:
        acpca = ACPCA(
            n_components=n_components,
            L=lam,
            preprocess=False,
            align_orientation=True,
        )
        start = time.perf_counter()
        coords = acpca.fit_transform(X_centered, tech_encoded)
        runtime = time.perf_counter() - start

        tech_sil = silhouette_score(coords[:, :metric_dims], tech)
        cell_sil = silhouette_score(coords[:, :metric_dims], celltype)
        score = cell_sil - abs(tech_sil)

        acpca_results.append(
            {
                "lambda": lam,
                "runtime_seconds": runtime,
                "batch_silhouette": tech_sil,
                "celltype_silhouette": cell_sil,
                "score": score,
            }
        )

        if best is None or score > best["score"]:
            best = acpca_results[-1]
            best_coords = coords

    if best is None or best_coords is None:
        raise RuntimeError("ACPCA evaluation failed; no results collected.")

    assets_dir = Path("assets")
    assets_dir.mkdir(exist_ok=True)
    metrics_path = assets_dir / "pancreas_acpca_metrics.csv"
    metrics_sorted = sorted(acpca_results, key=lambda item: item["lambda"])
    fieldnames = [
        "lambda",
        "runtime_seconds",
        "batch_silhouette",
        "celltype_silhouette",
        "score",
    ]
    with metrics_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics_sorted:
            writer.writerow(row)

    pca_mean = np.mean(pca_times)
    pca_std = np.std(pca_times)

    skip_figure = os.getenv("ACPCA_SKIP_FIGURE") == "1"
    fig_path = assets_dir / "pancreas_acpca_pca_comparison.png"
    if not skip_figure:
        plt = _get_pyplot()

        # 2×2 panel: PCA vs ACPCA (best lambda), colored by batch and cell type.
        cmap_batches = plt.get_cmap("tab20", len(np.unique(tech)))
        cmap_celltypes = plt.get_cmap("tab20b", len(np.unique(celltype)))

        fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex="col", sharey="row")
        panels = [
            ("PCA", pca_coords, tech, cmap_batches, np.unique(tech), "Batch"),
            (
                f"ACPCA (λ={best['lambda']:.2f})",
                best_coords,
                tech,
                cmap_batches,
                np.unique(tech),
                "Batch",
            ),
            (
                "PCA",
                pca_coords,
                celltype,
                cmap_celltypes,
                np.unique(celltype),
                "Cell type",
            ),
            (
                f"ACPCA (λ={best['lambda']:.2f})",
                best_coords,
                celltype,
                cmap_celltypes,
                np.unique(celltype),
                "Cell type",
            ),
        ]

        for ax, (title, coords, labels, cmap, categories, legend_title) in zip(
            axes.flat, panels
        ):
            for cat_idx, category in enumerate(categories):
                mask = labels == category
                ax.scatter(
                    coords[mask, 0],
                    coords[mask, 1],
                    s=16,
                    alpha=0.7,
                    linewidth=0.2,
                    edgecolor="white",
                    color=cmap(cat_idx),
                    label=category,
                )
            ax.set_title(title)
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")

            # Add one legend per row so both batch and cell-type mappings are visible.
            if legend_title == "Batch" and ax is axes[0, 0]:
                legend = ax.legend(
                    title=legend_title,
                    bbox_to_anchor=(1.02, 1),
                    loc="upper left",
                    borderaxespad=0.0,
                    fontsize=8,
                )
                legend.get_title().set_fontsize(9)
            elif legend_title == "Cell type" and ax is axes[1, 0]:
                legend = ax.legend(
                    title=legend_title,
                    bbox_to_anchor=(1.02, 1),
                    loc="upper left",
                    borderaxespad=0.0,
                    fontsize=8,
                )
                legend.get_title().set_fontsize(9)

        fig.suptitle(
            "Human pancreas benchmark: PCA vs ACPCA on ~1.2k cells (variance-selected genes)"
        )
        fig.tight_layout(rect=[0, 0, 0.78, 0.96])
        fig.savefig(fig_path, dpi=200)
        plt.close(fig)

    print("=== Summary ===")
    print(f"Subset matrix: {X_centered.shape[0]} cells × {X_centered.shape[1]} genes")
    print(
        f"PCA runtime per fit: {pca_mean*1e3:.2f} ± {pca_std*1e3:.2f} ms "
        f"(n={repetitions})"
    )
    print("ACPCA grid search results (sorted by lambda):")
    for row in metrics_sorted:
        print(
            "  "
            f"λ={row['lambda']:.2f} | "
            f"runtime={row['runtime_seconds']*1e3:.1f} ms | "
            f"batch_silhouette={row['batch_silhouette']:.3f} | "
            f"celltype_silhouette={row['celltype_silhouette']:.3f} | "
            f"score={row['score']:.3f}"
        )
    print(f"Metric CSV saved to {metrics_path}")
    if skip_figure:
        print("Figure generation skipped (set ACPCA_SKIP_FIGURE=0 to enable).")
    else:
        print(f"Comparison figure saved to {fig_path}")

    # Clustering evaluation comparing embeddings to true labels.
    cluster_dims = min(n_components, 6)
    n_celltypes = len(np.unique(celltype))
    n_batches = len(np.unique(tech))

    methods = [
        ("PCA", pca_coords),
        ("QuantileNorm_PCA", qn_pca_coords),
        (f"ACPCA_lambda_{best['lambda']:.2f}", best_coords),
    ]
    cluster_results_celltype = {
        name: cluster_analysis(embedding, celltype, n_celltypes, cluster_dims)
        for name, embedding in methods
    }
    cluster_results_batch = {
        name: cluster_analysis(embedding, tech, n_batches, cluster_dims)
        for name, embedding in methods
    }

    cluster_rows_raw = []
    for name, _ in methods:
        cell_res = cluster_results_celltype[name]
        batch_res = cluster_results_batch[name]
        cluster_rows_raw.append(
            {
                "method": name,
                "embedding_dims": cluster_dims,
                "celltype_ari": cell_res["ari"],
                "celltype_nmi": cell_res["nmi"],
                "batch_ari": batch_res["ari"],
                "batch_nmi": batch_res["nmi"],
            }
        )

    cluster_path = assets_dir / "pancreas_cluster_quality.csv"
    cluster_fields = [
        "method",
        "embedding_dims",
        "celltype_ari",
        "celltype_nmi",
        "batch_ari",
        "batch_nmi",
    ]
    cluster_rows_pretty = []
    for row in cluster_rows_raw:
        out_row = row.copy()
        out_row["method"] = pretty_method(row["method"])
        cluster_rows_pretty.append(out_row)

    with cluster_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=cluster_fields)
        writer.writeheader()
        for row in cluster_rows_pretty:
            writer.writerow(row)

    skip_cluster_fig = os.getenv("ACPCA_SKIP_CLUSTER_FIG") == "1"
    cluster_fig_path = assets_dir / "pancreas_cluster_density.png"
    if not skip_cluster_fig:
        plt = _get_pyplot()
        n_methods = len(methods)
        fig2, axes2 = plt.subplots(2, n_methods, figsize=(6 * n_methods, 10))
        if n_methods == 1:
            axes2 = np.array([[axes2[0]], [axes2[1]]])

        for col, (name, embedding) in enumerate(methods):
            cell_res = cluster_results_celltype[name]
            batch_res = cluster_results_batch[name]

            display_name = pretty_method(name)

            batch_ax = axes2[0, col]
            batch_ax.set_title(f"{display_name} – clusters vs. batch labels")
            render_cluster_panel(batch_ax, batch_res, "Batch", n_batches)

            cell_ax = axes2[1, col]
            cell_ax.set_title(f"{display_name} – clusters vs. cell types")
            render_cluster_panel(cell_ax, cell_res, "Cell type", n_celltypes)

        axes2[0, 0].set_ylabel("Component 2")
        axes2[1, 0].set_ylabel("Component 2")

        fig2.suptitle(
            "KMeans clustering on PCA, batch-normalized PCA, and ACPCA embeddings\n"
            "Row 1: clusters aligned to batch labels (k = number of batches)\n"
            "Row 2: clusters aligned to cell-type labels (k = number of cell types)"
        )
        fig2.tight_layout(rect=[0, 0, 1, 0.92])
        fig2.savefig(cluster_fig_path, dpi=300)
        plt.close(fig2)

    print("Cluster agreement with true labels (ARI/NMI):")
    for row in cluster_rows_pretty:
        print(
            f"  {row['method']}: "
            f"celltype ARI={row['celltype_ari']:.3f}, NMI={row['celltype_nmi']:.3f} | "
            f"batch ARI={row['batch_ari']:.3f}, NMI={row['batch_nmi']:.3f}"
        )
    print(f"Cluster metrics saved to {cluster_path}")
    if skip_cluster_fig:
        print("Cluster quality figure skipped (set ACPCA_SKIP_CLUSTER_FIG=0 to enable).")
    else:
        print(f"Cluster quality figure saved to {cluster_fig_path}")

    sweep_fig_path = assets_dir / "pancreas_acpca_lambda_sweep.png"
    skip_sweep_fig = os.getenv("ACPCA_SKIP_LAMBDA_SWEEP_FIG") == "1"
    lambda_values = np.round(np.arange(0.0, 1.01, 0.2), 2)
    lambda_records = []

    if not skip_sweep_fig:
        plt = _get_pyplot()
        ncols = 3
        nrows = int(np.ceil(len(lambda_values) / ncols))
        height_ratios = [1.0] * nrows + [0.55]
        fig3 = plt.figure(figsize=(6 * ncols, 4.8 * (nrows + 0.55)))
        gs = fig3.add_gridspec(nrows + 1, ncols, height_ratios=height_ratios)

        sweep_axes = []
        for row in range(nrows):
            for col in range(ncols):
                idx = row * ncols + col
                if idx >= len(lambda_values):
                    ax = fig3.add_subplot(gs[row, col])
                    ax.axis("off")
                    continue
                lam = float(lambda_values[idx])
                acpca_sweep = ACPCA(
                    n_components=n_components,
                    L=lam,
                    preprocess=False,
                    align_orientation=True,
                )
                coords = acpca_sweep.fit_transform(X_centered, tech_encoded)
                cell_res = cluster_analysis(
                    coords, celltype, n_celltypes, cluster_dims
                )
                batch_res = cluster_analysis(
                    coords, tech, n_batches, cluster_dims
                )
                lambda_records.append(
                    {
                        "lambda": lam,
                        "celltype_ari": cell_res["ari"],
                        "celltype_nmi": cell_res["nmi"],
                        "batch_ari": batch_res["ari"],
                        "batch_nmi": batch_res["nmi"],
                    }
                )
                ax = fig3.add_subplot(gs[row, col])
                render_cluster_panel(
                    ax,
                    cell_res,
                    "Cell type",
                    n_celltypes,
                )
                ax.set_title(f"ACPCA λ = {lam:.1f} – cell-type clusters")
                sweep_axes.append(ax)

        lambda_records.sort(key=lambda item: item["lambda"])
        line_ax = fig3.add_subplot(gs[-1, :])
        lam_arr = np.array([rec["lambda"] for rec in lambda_records])
        batch_ari_vals = np.array([rec["batch_ari"] for rec in lambda_records])
        batch_nmi_vals = np.array([rec["batch_nmi"] for rec in lambda_records])

        line_ax.plot(
            lam_arr,
            batch_ari_vals,
            marker="o",
            linewidth=2,
            markersize=7,
            label="Batch ARI",
            color="#1f77b4",
        )
        line_ax.plot(
            lam_arr,
            batch_nmi_vals,
            marker="s",
            linewidth=2,
            markersize=7,
            label="Batch NMI",
            color="#d62728",
        )
        line_ax.set_xlabel("ACPCA λ")
        line_ax.set_ylabel("Clustering score vs. batch")
        line_ax.set_ylim(0.0, max(0.05 + batch_nmi_vals.max(), 1.0))
        line_ax.grid(True, alpha=0.3, linewidth=0.8)
        line_ax.legend(frameon=False, loc="upper right")
        line_ax.set_title("Batch label agreement across λ sweep (KMeans, k = #batches)")

        fig3.suptitle(
            "ACPCA λ sensitivity: cell-type clustering panels and batch agreement metrics",
            y=0.99,
            fontsize=16,
        )
        fig3.tight_layout(rect=[0, 0, 1, 0.94])
        fig3.savefig(sweep_fig_path, dpi=300)
        plt.close(fig3)

    if lambda_records:
        print("ACPCA λ sweep batch metrics:")
        for rec in lambda_records:
            print(
                f"  λ={rec['lambda']:.1f}: batch ARI={rec['batch_ari']:.3f}, "
                f"batch NMI={rec['batch_nmi']:.3f}, "
                f"celltype ARI={rec['celltype_ari']:.3f}, "
                f"celltype NMI={rec['celltype_nmi']:.3f}"
            )
        if skip_sweep_fig:
            print("Lambda sweep figure skipped (set ACPCA_SKIP_LAMBDA_SWEEP_FIG=0 to enable).")
        else:
            print(f"Lambda sweep figure saved to {sweep_fig_path}")

if __name__ == "__main__":
    main()
