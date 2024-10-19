import pandas as pd
import matplotlib.pyplot as plt

# Load your data
data = pd.read_excel("data/E_Commerce_Dataset.xlsx", sheet_name='E Comm')

# 1. Bar Chart: Distribution of Purchase Frequency (OrderCount)
plt.figure(figsize=(8, 6))
plt.hist(data['OrderCount'].dropna(), bins=10, color='skyblue', edgecolor='black')
plt.title('Distribution of Purchase Frequency (OrderCount)')
plt.xlabel('Number of Purchases')
plt.ylabel('Number of Customers')
plt.grid(True)
plt.savefig("end_to_end_deployment/churn-report/purchase_frequency.png")

# 2. Pie Chart: Churn vs. Non-Churn Customers
churn_counts = data['Churn'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(churn_counts, labels=['Non-Churn', 'Churn'], autopct='%1.1f%%', colors=['green', 'red'], startangle=90)
plt.title('Churn vs. Non-Churn Customers')
plt.savefig("end_to_end_deployment/churn-report/churn_distribution.png")

# Generate the HTML file
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Customer Churn Prediction</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            background-color: #f4f4f9;
            color: #333;
            margin: 20px;
        }}
        h1 {{
            text-align: center;
            color: #4CAF50;
        }}
        .content {{
            max-width: 900px;
            margin: auto;
        }}
        img {{
            width: 100%;
            max-width: 600px;
            display: block;
            margin: 20px auto;
        }}
        .summary {{
            text-align: center;
            margin: 20px 0;
        }}
        .summary p {{
            font-size: 18px;
        }}
    </style>
</head>
<body>
    <div class="content">
        <h1>Customer Churn Prediction Report</h1>
        <div class="summary">
            <p>This report provides insights into customer churn based on purchase frequency and churn status in the dataset.</p>
        </div>

        <h2>Distribution of Purchase Frequency</h2>
        <img src="purchase_frequency.png" alt="Distribution of Purchase Frequency">

        <h2>Churn vs. Non-Churn Customers</h2>
        <img src="churn_distribution.png" alt="Churn vs. Non-Churn Customers">

        <div class="summary">
            <p><b>Total Customers:</b> {total_customers}</p>
            <p><b>Churned Customers:</b> {churned_customers}</p>
            <p><b>Non-Churned Customers:</b> {non_churned_customers}</p>
        </div>
    </div>
</body>
</html>
"""

# Calculate summary statistics
total_customers = len(data)
churned_customers = churn_counts[1]
non_churned_customers = churn_counts[0]

# Save the HTML content
html_content = html_content.format(
    total_customers=total_customers,
    churned_customers=churned_customers,
    non_churned_customers=non_churned_customers
)

with open("end_to_end_deployment/churn-report/customer_churn_report.html", "w") as f:
    f.write(html_content)
