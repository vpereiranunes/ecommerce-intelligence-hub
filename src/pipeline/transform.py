"""
transform.py - Feature engineering para RFM e churn.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger

PROCESSED_DIR = Path("data/processed")
SNAPSHOT_DATE = None  # será definido como max(invoice_date) + 1 dia


def load_clean_data() -> pd.DataFrame:
    path = PROCESSED_DIR / "transactions_clean.parquet"
    df = pd.read_parquet(path)
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    df["revenue"] = df["quantity"] * df["price"]
    return df


def build_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula métricas RFM por cliente."""
    global SNAPSHOT_DATE
    SNAPSHOT_DATE = df["invoice_date"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("customer_id").agg(
        recency=("invoice_date", lambda x: (SNAPSHOT_DATE - x.max()).days),
        frequency=("invoice", "nunique"),
        monetary=("revenue", "sum"),
        first_purchase=("invoice_date", "min"),
        last_purchase=("invoice_date", "max"),
        n_products=("stock_code", "nunique"),
        avg_basket=("revenue", lambda x: x.sum() / x.count()),
        country=("country", lambda x: x.mode()[0]),
    ).reset_index()

    logger.info(f"RFM calculado para {len(rfm):,} clientes")
    return rfm


def build_churn_label(rfm: pd.DataFrame, threshold_days: int = 90) -> pd.DataFrame:
    """
    Define churn: cliente sem compra nos últimos `threshold_days` dias.
    """
    rfm["churn"] = (rfm["recency"] > threshold_days).astype(int)
    churn_rate = rfm["churn"].mean()
    logger.info(f"Churn rate: {churn_rate:.1%} (threshold: {threshold_days} dias)")
    return rfm


def add_behavioral_features(rfm: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Features comportamentais extras."""
    # Intervalo médio entre compras
    purchase_gaps = (
        df.sort_values("invoice_date")
        .groupby("customer_id")["invoice_date"]
        .apply(lambda x: x.diff().dt.days.mean())
        .rename("avg_days_between_purchases")
        .reset_index()
    )

    # Mês mais frequente de compra
    df["month"] = df["invoice_date"].dt.month
    fav_month = (
        df.groupby("customer_id")["month"]
        .agg(lambda x: x.mode()[0])
        .rename("favorite_month")
        .reset_index()
    )

    rfm = rfm.merge(purchase_gaps, on="customer_id", how="left")
    rfm = rfm.merge(fav_month, on="customer_id", how="left")
    rfm["avg_days_between_purchases"] = rfm["avg_days_between_purchases"].fillna(rfm["recency"])

    return rfm


def scale_rfm_scores(rfm: pd.DataFrame) -> pd.DataFrame:
    """Adiciona scores de 1-5 para R, F, M (para visualização)."""
    rfm["r_score"] = pd.qcut(rfm["recency"], q=5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm["f_score"] = pd.qcut(rfm["frequency"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["m_score"] = pd.qcut(rfm["monetary"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["rfm_score"] = rfm["r_score"] + rfm["f_score"] + rfm["m_score"]
    return rfm


def run():
    logger.info("=== ETAPA 2: TRANSFORMAÇÃO ===")
    df = load_clean_data()
    rfm = build_rfm(df)
    rfm = build_churn_label(rfm)
    rfm = add_behavioral_features(rfm, df)
    rfm = scale_rfm_scores(rfm)

    out_path = PROCESSED_DIR / "rfm_features.parquet"
    rfm.to_parquet(out_path, index=False)
    logger.success(f"Features salvas em {out_path}")
    return rfm


if __name__ == "__main__":
    run()
