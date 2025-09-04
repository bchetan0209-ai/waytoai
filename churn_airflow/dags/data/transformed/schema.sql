
    CREATE TABLE transformed_customers (
        customer_id VARCHAR(50) PRIMARY KEY,
        monthly_spend FLOAT,
        invoices INT,
        last_payment_date FLOAT,
        sign_up_date FLOAT,
        last_seen_at FLOAT,
        sessions_30d INT,
        pages_viewed_30d INT,
        time_on_site_sec_30d FLOAT,
        avg_monthly_spend_per_invoice FLOAT,
        engagement_score FLOAT,
        tenure_proxy FLOAT,
        avg_monthly_spend_per_invoice_scaled FLOAT,
        engagement_score_scaled FLOAT,
        tenure_proxy_scaled FLOAT,
        churn INT
    );
    