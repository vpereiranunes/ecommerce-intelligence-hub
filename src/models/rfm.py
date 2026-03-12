"""
rfm.py - Segmentação de clientes via K-Means sobre features RFM.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from loguru import logger

PROCESSED_DIR = Path("data/processed")
FIGURES_DIR = Path("dashboard/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_features() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / "rfm_features.parquet")


def find_optimal_k(X_scaled: np.ndarray, k_range: range = range(2, 10)) -> int:
    """Método do cotovelo + Silhouette para escolher k."""
    inertias, silhouettes = [], []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(k_range, inertias, "bo-")
    ax1.set(title="Elbow Method", xlabel="Número de clusters (k)", ylabel="Inertia")
    ax2.plot(k_range, silhouettes, "ro-")
    ax2.set(title="Silhouette Score", xlabel="k", ylabel="Score")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "optimal_k.png", dpi=150, bbox_inches="tight")
    plt.close()

    best_k = k_range[np.argmax(silhouettes)]
    logger.info(f"K ótimo escolhido: {best_k} (Silhouette: {max(silhouettes):.3f})")
    return best_k


def fit_kmeans(X_scaled: np.ndarray, k: int) -> KMeans:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    return km


def label_segments(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Nomeia segmentos com base no perfil médio de cada cluster.
    Os rótulos são atribuídos dinamicamente após análise do centroide.
    """
    segment_profiles = rfm.groupby("segment")[["recency", "frequency", "monetary"]].mean()

    # Rank: menor recency = mais recente (melhor), maior freq e monetary = melhor
    segment_profiles["score"] = (
        -segment_profiles["recency"].rank()
        + segment_profiles["frequency"].rank()
        + segment_profiles["monetary"].rank()
    )

    score_rank = segment_profiles["score"].rank(ascending=False)

    labels_map = {1: "Champions", 2: "Leais", 3: "Em risco", 4: "Perdidos"}
    # Mapeia pelo rank (funciona para k=4; ajuste para outros k)
    segment_profiles["label"] = score_rank.map(
        {i: labels_map.get(i, f"Cluster {i}") for i in score_rank}
    )

    rfm = rfm.merge(
        segment_profiles[["label"]].reset_index(),
        on="segment"
    )
    return rfm


def plot_segments(rfm: pd.DataFrame):
    """Scatter plot 3D simplificado em 2D (Recency vs Monetary, colorido por segmento)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    palette = {"Champions": "#2ecc71", "Leais": "#3498db", "Em risco": "#e67e22", "Perdidos": "#e74c3c"}

    # Plot 1: Recency vs Monetary
    for label, group in rfm.groupby("label"):
        axes[0].scatter(group["recency"], group["monetary"],
                        label=label, alpha=0.6, s=20, color=palette.get(label, "gray"))
    axes[0].set(title="Recency vs Monetary", xlabel="Recency (dias)", ylabel="Monetary (£)")
    axes[0].legend()

    # Plot 2: Tamanho dos segmentos
    counts = rfm["label"].value_counts()
    colors = [palette.get(l, "gray") for l in counts.index]
    axes[1].bar(counts.index, counts.values, color=colors)
    axes[1].set(title="Clientes por Segmento", xlabel="Segmento", ylabel="Quantidade")
    for i, v in enumerate(counts.values):
        axes[1].text(i, v + 10, f"{v:,}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "segments.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.success("Gráficos de segmentação salvos.")


def run():
    logger.info("=== MODELO: SEGMENTAÇÃO RFM ===")
    rfm = load_features()

    features = ["recency", "frequency", "monetary"]
    X = rfm[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    best_k = find_optimal_k(X_scaled)
    km = fit_kmeans(X_scaled, best_k)

    rfm["segment"] = km.labels_
    rfm = label_segments(rfm)

    # Resumo por segmento
    summary = rfm.groupby("label")[features + ["churn"]].agg(["mean", "count"]).round(1)
    logger.info(f"\n{summary}")

    plot_segments(rfm)

    out_path = PROCESSED_DIR / "rfm_segmented.parquet"
    rfm.to_parquet(out_path, index=False)
    logger.success(f"Segmentação salva em {out_path}")
    return rfm


if __name__ == "__main__":
    run()
