SELECT customer_id, monthly_spend, churn FROM transformed_customers WHERE churn=1;
SELECT AVG(engagement_score) as avg_engagement FROM transformed_customers GROUP BY churn;
SELECT customer_id, tenure_proxy FROM transformed_customers ORDER BY tenure_proxy DESC LIMIT 10;