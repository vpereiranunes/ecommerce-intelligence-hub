# 🛒 E-commerce Intelligence Hub

> Pipeline completo de dados com Machine Learning e IA Generativa para análise de clientes em e-commerce.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4-orange?logo=scikit-learn)
![SQL](https://img.shields.io/badge/SQL-PostgreSQL-336791?logo=postgresql)
![Claude AI](https://img.shields.io/badge/AI-Claude_API-blueviolet)
![Status](https://img.shields.io/badge/Status-Em%20desenvolvimento-yellow)

---

## 📌 Visão Geral

Este projeto simula um ambiente real de **analytics em e-commerce**, cobrindo desde a ingestão de dados brutos até a geração de insights executivos com IA. O objetivo é demonstrar domínio do stack completo de dados moderno.

### Problema de negócio
Uma empresa de e-commerce enfrenta aumento de churn, dificuldade em personalizar campanhas e falta de visibilidade sobre o valor real de cada cliente.

### Solução
Pipeline automatizado que:
1. **Processa** dados transacionais históricos
2. **Segmenta** clientes via RFM + K-Means
3. **Prediz** churn com modelos de ML
4. **Gera** insights e recomendações em linguagem natural via LLM
5. **Visualiza** KPIs em dashboard executivo

---

## 🏗️ Arquitetura

```
Raw Data (CSV/DB)
      │
      ▼
┌─────────────────┐
│  ETL Pipeline   │  ← Python + SQL
│  (src/pipeline) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Feature Store  │  ← Pandas + PostgreSQL
│  (data/processed)│
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐ ┌──────────┐
│  RFM  │ │  Churn   │  ← Scikit-learn
│K-Means│ │Prediction│
└───┬───┘ └────┬─────┘
    └────┬─────┘
         ▼
┌─────────────────┐
│   AI Insights   │  ← Claude API
│  (src/ai_insights)│
└────────┬────────┘
         ▼
┌─────────────────┐
│    Dashboard    │  ← Power BI
└─────────────────┘
```

---

## 📂 Estrutura do Projeto

```
ecommerce-intelligence-hub/
├── data/
│   ├── raw/                    # Dados brutos (não commitados)
│   └── processed/              # Dados tratados
├── notebooks/
│   ├── 01_eda.ipynb            # Análise exploratória
│   ├── 02_rfm_segmentation.ipynb
│   ├── 03_churn_modeling.ipynb
│   └── 04_ai_insights.ipynb
├── src/
│   ├── pipeline/
│   │   ├── extract.py          # Ingestão de dados
│   │   ├── transform.py        # Transformações
│   │   └── load.py             # Carga no banco
│   ├── models/
│   │   ├── rfm.py              # Segmentação RFM
│   │   ├── churn.py            # Modelo de churn
│   │   └── evaluate.py         # Métricas e validação
│   └── ai_insights/
│       ├── generator.py        # Geração de insights via LLM
│       └── prompts.py          # Templates de prompts
├── sql/
│   ├── schema.sql              # DDL do banco
│   ├── rfm_query.sql           # Query de features RFM
│   └── kpis.sql                # KPIs do negócio
├── dashboard/                  # Prints e arquivo .pbix
├── tests/
│   ├── test_pipeline.py
│   └── test_models.py
├── .env.example
├── requirements.txt
└── README.md
```

---

## 🚀 Como Executar

### 1. Clone o repositório
```bash
git clone https://github.com/seu-usuario/ecommerce-intelligence-hub.git
cd ecommerce-intelligence-hub
```

### 2. Configure o ambiente
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

pip install -r requirements.txt
cp .env.example .env      # Configure suas credenciais
```

### 3. Execute o pipeline
```bash
python src/pipeline/extract.py      # Baixa e valida dados
python src/pipeline/transform.py    # Processa features
python src/models/rfm.py            # Segmentação
python src/models/churn.py          # Modelo preditivo
python src/ai_insights/generator.py # Gera insights com IA
```

---

## 📊 Dataset

Utiliza o dataset público **[Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii)** (UCI ML Repository), com ~1 milhão de transações de um e-commerce britânico entre 2009-2011.

| Campo | Descrição |
|-------|-----------|
| InvoiceNo | ID da transação |
| StockCode | Código do produto |
| Description | Descrição do produto |
| Quantity | Quantidade |
| InvoiceDate | Data da compra |
| UnitPrice | Preço unitário |
| CustomerID | ID do cliente |
| Country | País |

---

## 🤖 Modelos

### Segmentação RFM + K-Means
- **Recency**: dias desde a última compra
- **Frequency**: número de compras
- **Monetary**: valor total gasto
- Clusterização com K-Means (k otimizado via Elbow + Silhouette)

### Predição de Churn
- Features: RFM + métricas comportamentais
- Modelos comparados: Random Forest, XGBoost, LightGBM
- Otimização: Optuna (hyperparameter tuning)
- Threshold ajustado para maximizar recall de churn

### AI Insights (Claude API)
- Recebe métricas agregadas por segmento
- Gera narrativas executivas e recomendações acionáveis
- Output estruturado em JSON para integração com dashboard

---

## 📈 Resultados

| Métrica | Valor |
|---------|-------|
| AUC-ROC (Churn) | *a preencher* |
| Silhouette Score (RFM) | *a preencher* |
| Clientes segmentados | *a preencher* |
| Churn identificado | *a preencher* |

---

## 🛠️ Tech Stack

| Camada | Tecnologia |
|--------|-----------|
| Linguagem | Python 3.11 |
| Processamento | Pandas, NumPy |
| ML | Scikit-learn, XGBoost, LightGBM, Optuna |
| Banco de dados | PostgreSQL / SQLite |
| IA Generativa | Anthropic Claude API |
| Visualização | Power BI / Matplotlib / Seaborn |
| Versionamento | Git + DVC |

---

## 📬 Contato

Desenvolvido por **[Seu Nome]** — [[LinkedIn]](https://linkedin.com/in/seu-perfil) · [[Portfolio]](https://seu-site.com)
