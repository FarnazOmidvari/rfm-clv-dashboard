<<<<<<< HEAD
# rfm-clv-dashboard
=======
# 📊 Customer Segmentation & CLV Analysis

## 🚀 Project Highlights
- 📈 **RFM Analysis** – Segmentation of customers based on **Recency**, **Frequency**, and **Monetary value**.  
- 💰 **Customer Lifetime Value (CLV) Estimation** – Measuring the potential value of each customer over time.  
- 🖥️ **Interactive Dashboard (Power BI)**  
  - **📌 Treemap** → Number of clusters & customer counts (quarterly).
  - **📊 Clustered Column Chart** → RFM metrics & CLV per cluster  
  - **🔎 Column Chart** → Total monthly sales trends.
  - **Pie Chart** → Sales contribution by cluster.
  - **Slicers** → Filter by *Cluster* and *Country*.


---

## 📊 Dashboard Preview

![Dashboard Overview](assets/dashboard_main_page.png)
![Customer Details](assets/dashboard_customer_detail.png)

---

## 📌 Cluster Insightsٰ

| Cluster | Emoji | Recency | Frequency | Monetary | CLV | Interpretation |
|---------|-------|---------|-----------|----------|-----|----------------|
| 0 | 🟢 | Low (recent purchase) | Above avg | Above avg | 0.86 | **Active Loyal** – valuable repeat customers; maintain engagement |
| 1 | 🟡 | Medium | Low | Low | -0.14 | **Low-Value Active** – many customers, but low spending; target for upsell/cross-sell |
| 2 | 👑 | Very Low (very recent) | Very High | Very High | 21.2 | **VIP Champions** – highest value, very few customers; must be retained at all costs |
| 3 | 🔴 | High (long since last purchase) | Low | Low | -0.20 | **At-Risk / Hibernating** – likely churned; need reactivation campaigns |
| 4 | 🟣 | Low | High | Medium-High | 5.7 | **High-Value Potential** – on the verge of becoming VIP; nurture with marketing campaigns |

> 💡 **Note:** The table provides a quick visual overview of cluster characteristics and their CLV.

---

## 📝 Managerial Summary
## 📝 Managerial Summary (Updated)

- **🎯 Customer Segmentation:** Differentiates **At-Risk / Hibernating (Cluster 3 🔴)**, **Low-Value Active (Cluster 1 🟡)**, **Active Loyal (Cluster 0 🟢)**, **High-Value Potential (Cluster 4 🟣)**, and **VIP Champions (Cluster 2 👑)** customers.

- **💡 Retention Focus:** **At-Risk / Hibernating customers (Cluster 3 🔴)** require targeted reactivation campaigns.

- **📈 Growth Opportunities:** **High-Value Potential (Cluster 4 🟣)** and **Low-Value Active (Cluster 1 🟡)** customers can be nurtured toward **VIP status**.

- **💎 High-Value Customers:** **VIP Champions (Cluster 2 👑)** are the primary source of revenue and long-term business value.

- **💚 Loyal Customers:** **Active Loyal (Cluster 0 🟢)** maintain consistent engagement and provide stable revenue streams.

- **📊 Data-Driven Decisions:** Power BI dashboard enables **trend monitoring**, **cluster comparisons**, and **targeted strategy planning**.

Decisions:** Power BI dashboard enables trend monitoring and detailed cluster analysis  

---


## 📂 Project Structure

```text
rfm-clv-dashboard/
├── notebooks/       # Jupyter Notebooks (RFM, CLV calculations, data cleaning)
├── data/            # OnlineRetail.xlsx
├── assets/          # Dashboard screenshots
├── powerbi/         # Power BI (RFM_CLV_dashboard.pbix) file
├── exports/         # excel and csv exported files
├── README.md        # Project documentation
├── .gitignore       # Ignore unnecessary files
└── LICENSE          # MIT License
```

---

## 📂 Dataset
- **Source:** Internal sales and transaction data  
- **Preprocessing:** Data cleaning and RFM & CLV calculation performed before normalization and clustering  

---

## 🛠️ Tech Stack
- **Python**: Pandas, NumPy, Matplotlib  
- **Power BI**: Interactive dashboarding & visualization  
- **Git/GitHub**: Version control & project sharing  

---

## ⚙️ How to Use
1. Clone the repository:
```bash
git clone  https://github.com/FarnazOmidvari/rfm-clv-dashboard
```

2. Open the Power BI dashboard (RFM_CLV_dashboard.pbix file) in Power BI Desktop to explore the dashboard interactively.
3. Open the Jupyter Notebooks in notebooks/ for data processing.
4. Replace data/OnlineRetail.xlsx with your own dataset (if available).


---

🚧 Future Improvements

🔹 Deploy dashboard online (Power BI Service / Streamlit).

🔹 Add predictive models for churn & CLV forecasting.

🔹 Automate ETL pipeline with Airflow.

---

📜 License

This project is licensed under the MIT License – see the LICENSE
 file for details.

👤 Author

Farnaz Omidvari

📧 farnaz.omidvari1983@gmail.com

💼 https://www.linkedin.com/in/farnazomidvari/

>>>>>>> 7ea8886 (Initial commit - RFM CLV Dashboard project)
