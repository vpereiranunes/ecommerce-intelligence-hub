-- ============================================
-- schema.sql - DDL do banco de dados
-- ============================================

CREATE TABLE IF NOT EXISTS transactions (
    invoice         VARCHAR(20),
    stock_code      VARCHAR(20),
    description     TEXT,
    quantity        INTEGER,
    invoice_date    TIMESTAMP,
    price           NUMERIC(10, 2),
    customer_id     VARCHAR(20),
    country         VARCHAR(50),
    revenue         NUMERIC(10, 2),
    PRIMARY KEY (invoice, stock_code)
);

CREATE TABLE IF NOT EXISTS customers (
    customer_id                  VARCHAR(20) PRIMARY KEY,
    recency                      INTEGER,
    frequency                    INTEGER,
    monetary                     NUMERIC(12, 2),
    r_score                      SMALLINT,
    f_score                      SMALLINT,
    m_score                      SMALLINT,
    rfm_score                    SMALLINT,
    segment                      SMALLINT,
    label                        VARCHAR(30),
    churn                        SMALLINT,
    churn_probability            NUMERIC(5, 4),
    churn_predicted              SMALLINT,
    avg_days_between_purchases   NUMERIC(8, 2),
    country                      VARCHAR(50),
    last_purchase                TIMESTAMP,
    first_purchase               TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_customers_label ON customers(label);
CREATE INDEX IF NOT EXISTS idx_customers_churn ON customers(churn_predicted);
