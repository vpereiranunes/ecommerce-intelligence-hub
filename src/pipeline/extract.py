"""
extract.py - Ingestão e validação dos dados brutos.
Dataset: Online Retail II (UCI ML Repository)
"""
import os
import requests
import zipfile
import pandas as pd
from pathlib import Path
from loguru import logger

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
DATASET_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"


def download_dataset() -> Path:
    """Baixa o dataset do UCI ML Repository."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / "online_retail_ii.zip"

    if zip_path.exists():
        logger.info("Dataset já baixado, pulando download.")
        return zip_path

    logger.info(f"Baixando dataset de {DATASET_URL}...")
    response = requests.get(DATASET_URL, stream=True)
    response.raise_for_status()

    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    logger.success(f"Download concluído: {zip_path}")
    return zip_path


def extract_zip(zip_path: Path) -> Path:
    """Extrai o arquivo zip."""
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(RAW_DIR)
    logger.success(f"Arquivos extraídos em {RAW_DIR}")
    return RAW_DIR


def load_raw_data() -> pd.DataFrame:
    """Carrega e combina os dois anos do dataset."""
    files = list(RAW_DIR.glob("*.xlsx"))
    if not files:
        raise FileNotFoundError("Nenhum arquivo .xlsx encontrado em data/raw/")

    dfs = []
    for f in files:
        logger.info(f"Carregando {f.name}...")
        df = pd.read_excel(f, dtype={"Customer ID": str})
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Total de registros brutos: {len(combined):,}")
    return combined


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validações básicas de qualidade."""
    initial_len = len(df)

    # Padroniza nomes de colunas
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    # Remove registros inválidos
    df = df.dropna(subset=["customer_id"])
    df = df[df["quantity"] > 0]
    df = df[df["price"] > 0]

    # Remove cancelamentos (InvoiceNo começando com 'C')
    df = df[~df["invoice"].astype(str).str.startswith("C")]

    removed = initial_len - len(df)
    logger.info(f"Registros removidos na validação: {removed:,} ({removed/initial_len:.1%})")
    logger.success(f"Registros válidos: {len(df):,}")
    return df


def save_clean_data(df: pd.DataFrame) -> Path:
    """Salva dados limpos em parquet (eficiente para próximas etapas)."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "transactions_clean.parquet"
    df.to_parquet(out_path, index=False)
    logger.success(f"Dados limpos salvos em {out_path}")
    return out_path


def run():
    logger.info("=== ETAPA 1: EXTRAÇÃO ===")
    zip_path = download_dataset()
    extract_zip(zip_path)
    df = load_raw_data()
    df = validate_data(df)
    save_clean_data(df)
    logger.success("Extração concluída com sucesso!")
    return df


if __name__ == "__main__":
    run()
