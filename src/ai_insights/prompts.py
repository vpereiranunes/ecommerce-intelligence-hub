"""
prompts.py - Templates de prompts para geração de insights via LLM.
"""
import json


def build_system_prompt() -> str:
    return """Você é um analista sênior de dados especializado em e-commerce e CRM.
Sua função é interpretar métricas de clientes (RFM + churn) e gerar insights
executivos claros, diretos e acionáveis para o time de negócios.

Responda SOMENTE com um JSON válido, sem markdown, sem texto adicional.
O JSON deve ter exatamente esta estrutura:
{
  "executive_summary": "Parágrafo de 2-3 frases com o panorama geral do negócio",
  "segment_insights": [
    {
      "segment": "nome do segmento",
      "insight": "O que este segmento representa e seu comportamento",
      "action": "Ação concreta de marketing/CRM recomendada"
    }
  ],
  "top_recommendations": [
    "Recomendação prioritária 1",
    "Recomendação prioritária 2",
    "Recomendação prioritária 3"
  ],
  "churn_alert": "Alerta específico sobre risco de churn e impacto estimado em receita"
}"""


def build_user_prompt(metrics: dict) -> str:
    return f"""Analise as métricas de clientes abaixo e gere insights executivos.

## VISÃO GERAL
{json.dumps(metrics['overall'], indent=2, ensure_ascii=False)}

## BREAKDOWN POR SEGMENTO
{json.dumps(metrics['by_segment'], indent=2, ensure_ascii=False)}

Contexto:
- Churn é definido como ausência de compra nos últimos 90 dias
- Churn probability > 0.7 indica alto risco
- Monetary está em libras esterlinas (£)
- O negócio é um e-commerce britânico B2C de produtos de varejo

Gere o JSON de insights agora."""
