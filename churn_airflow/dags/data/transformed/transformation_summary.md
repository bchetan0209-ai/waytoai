# Transformation Summary

## Feature Engineering

### Derived / Aggregated Features
- **avg_monthly_spend_per_invoice** = monthly_spend / invoices
- **engagement_score** = sessions_30d + pages_viewed_30d + (time_on_site_sec_30d / 60)
- **tenure_proxy** = last_seen_at - sign_up_date

### Scaling / Normalization
- Applied **StandardScaler** (mean=0, std=1) on:
  - avg_monthly_spend_per_invoice
  - engagement_score
  - tenure_proxy

### Target
- Data is stored locally as **transformed.csv** inside `/usr/local/airflow/dags/data/transformed`.

---

## Deliverables
- **Transformed dataset**: `transformed.csv`
- **SQL Schema**: `schema.sql`
- **Sample Queries**: `sample_queries.sql`
- **Summary**: `transformation_summary.md`
