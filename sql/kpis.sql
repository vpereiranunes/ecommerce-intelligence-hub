-- ============================================
-- kpis.sql - KPIs executivos para dashboard
-- ============================================

-- 1. Visão geral de clientes
SELECT
    COUNT(*)                                    AS total_customers,
    ROUND(AVG(monetary), 2)                     AS avg_clv,
    ROUND(SUM(monetary), 2)                     AS total_revenue,
    ROUND(AVG(churn_probability) * 100, 1)      AS avg_churn_prob_pct,
    SUM(CASE WHEN churn_predicted = 1 THEN 1 ELSE 0 END) AS predicted_churners,
    ROUND(
        SUM(CASE WHEN churn_predicted = 1 THEN monetary ELSE 0 END), 2
    )                                           AS revenue_at_risk
FROM customers;

-- 2. Breakdown por segmento
SELECT
    label                                       AS segment,
    COUNT(*)                                    AS customers,
    ROUND(AVG(recency), 0)                      AS avg_recency_days,
    ROUND(AVG(frequency), 1)                    AS avg_frequency,
    ROUND(AVG(monetary), 2)                     AS avg_monetary,
    ROUND(AVG(churn_probability) * 100, 1)      AS avg_churn_risk_pct,
    ROUND(SUM(monetary), 2)                     AS total_segment_revenue
FROM customers
GROUP BY label
ORDER BY avg_monetary DESC;

-- 3. Top 20 clientes com maior risco e maior valor (priorizar retenção)
SELECT
    customer_id,
    label,
    monetary,
    churn_probability,
    recency,
    frequency,
    country
FROM customers
WHERE churn_probability > 0.6
ORDER BY monetary DESC
LIMIT 20;

-- 4. Revenue at risk por país
SELECT
    country,
    COUNT(*) FILTER (WHERE churn_predicted = 1) AS churners,
    ROUND(SUM(monetary) FILTER (WHERE churn_predicted = 1), 2) AS revenue_at_risk
FROM customers
GROUP BY country
HAVING COUNT(*) FILTER (WHERE churn_predicted = 1) > 5
ORDER BY revenue_at_risk DESC;

-- 5. Distribuição de RFM Score
SELECT
    rfm_score,
    COUNT(*) AS customers,
    ROUND(AVG(monetary), 2) AS avg_monetary
FROM customers
GROUP BY rfm_score
ORDER BY rfm_score;
