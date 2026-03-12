"""
generator.py - Gera insights executivos usando Claude API.
Recebe métricas agregadas e retorna narrativas + recomendações em JSON.
"""
import json
import os
import pandas as pd
import anthropic
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv
from .prompts import build_system_prompt, build_user_prompt

load_dotenv()

PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR = Path("data/processed")


def compute_segment_metrics(df: pd.DataFrame) -> dict:
    """Agrega métricas por segmento para enviar ao LLM."""
    overall = {
        "total_customers": len(df),
        "churn_rate": round(df["churn"].mean(), 3),
        "avg_revenue_per_customer": round(df["monetary"].mean(), 2),
        "total_revenue": round(df["monetary"].sum(), 2),
        "high_churn_risk_customers": int((df["churn_probability"] > 0.7).sum()),
    }

    by_segment = (
        df.groupby("label").agg(
            customers=("customer_id", "count"),
            avg_recency=("recency", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean"),
            churn_rate=("churn", "mean"),
            avg_churn_probability=("churn_probability", "mean"),
        )
        .round(2)
        .reset_index()
        .to_dict(orient="records")
    )

    return {"overall": overall, "by_segment": by_segment}


def generate_insights(metrics: dict) -> dict:
    """
    Envia métricas ao Claude e retorna insights estruturados.

    Retorno esperado:
    {
        "executive_summary": str,
        "segment_insights": [{"segment": str, "insight": str, "action": str}],
        "top_recommendations": [str],
        "churn_alert": str
    }
    """
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    message = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1500,
        system=build_system_prompt(),
        messages=[
            {"role": "user", "content": build_user_prompt(metrics)}
        ]
    )

    raw_text = message.content[0].text

    # Remove markdown se o modelo incluir ```json
    clean = raw_text.strip().removeprefix("```json").removesuffix("```").strip()

    try:
        insights = json.loads(clean)
    except json.JSONDecodeError:
        logger.warning("Resposta não era JSON válido, retornando como texto.")
        insights = {"executive_summary": raw_text}

    return insights


def run():
    logger.info("=== IA INSIGHTS ===")
    df = pd.read_parquet(PROCESSED_DIR / "final_dataset.parquet")

    metrics = compute_segment_metrics(df)
    logger.info(f"Métricas calculadas:\n{json.dumps(metrics['overall'], indent=2)}")

    insights = generate_insights(metrics)

    # Salva JSON
    out_path = OUTPUT_DIR / "ai_insights.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(insights, f, ensure_ascii=False, indent=2)

    logger.success(f"Insights salvos em {out_path}")

    # Preview no terminal
    if "executive_summary" in insights:
        logger.info(f"\n📊 RESUMO EXECUTIVO:\n{insights['executive_summary']}")

    if "top_recommendations" in insights:
        logger.info("\n🎯 TOP RECOMENDAÇÕES:")
        for i, rec in enumerate(insights["top_recommendations"], 1):
            logger.info(f"  {i}. {rec}")

    return insights


if __name__ == "__main__":
    run()
