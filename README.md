
# 📦 Vendor Invoice Intelligence Portal
### AI-Driven Freight Cost Prediction & Invoice Risk Flagging

Organizations process thousands of invoices every day, and manual verification is slow, error-prone, and costly. This project is a Machine Learning-powered internal analytics portal designed for Finance and Supply Chain teams. It automates the verification of vendor invoices and forecasts logistics costs, effectively reducing manual workload and financial leakage.

## 🚀 Project Overview & Business Impact
This system helps finance teams focus only on suspicious transactions by providing instant, AI-backed analysis.
* 📉 **Improved Cost Forecasting**: Accurately predicts expected freight costs to support budgeting and vendor negotiations.
* 🧾 **Reduced Invoice Fraud**: Detects abnormal vendor behavior and discrepancies using 3-way matching logic.
* ⚙️ **Faster Finance Operations**: Streamlines the approval pipeline by auto-approving safe invoices and flagging risky ones.

---

## 📸 Dashboard Preview
The portal features a clean, intuitive interface built with Streamlit, providing real-time AI inference.

### 1. Freight Cost Prediction
Estimates the logistical cost based on invoice dollar amounts using a trained Random Forest Regression model.
![Freight Prediction](Freight%20Prediction.png)

### 2. Invoice Risk Analysis (SAFE)
When the vendor invoice aligns with warehouse receiving data, the system marks it for auto-approval.
![Safe Invoice](Safe%20Invoice.png)

### 3. Invoice Risk Analysis (FLAGGED)
If a significant discrepancy is detected between billed amounts and received goods, the system triggers a high-risk warning for manual review.
![Flagged Invoice](Flag%20Invoice.png)

---

## 🧠 Machine Learning Architecture

### 🚚 Module 1: Freight Cost Prediction
* **Goal**: Forecast the expected freight cost of a vendor invoice using order value.
* **Models Evaluated**: Linear Regression, Decision Tree, Random Forest.
* **Metric**: Optimized for lowest Mean Absolute Error (MAE).

### 🚨 Module 2: Invoice Risk Flagging
* **Goal**: Classify invoices as `Safe` (0) or `Requires Manual Approval` (1).
* **Logic**: Analyzes abnormal patterns between Invoice Quantities/Dollars vs. Actually Received Quantities/Dollars.
* **Models Evaluated**: Logistic Regression, Decision Tree, and Random Forest.
* **Best Performing Model**: **Random Forest**
* **Model Tuning**: Hyperparameters were systematically optimized using `GridSearchCV`.
* **Evaluation Metrics**: Performance was validated using Accuracy, Precision, Recall, F1 Score, and Confusion Matrix to account for class imbalances.

---

## 📊 Dataset Features
The system was trained on historical vendor invoice data using the following engineered features:

| Feature | Description |
| :--- | :--- |
| `invoice_quantity` | Quantity claimed in the vendor invoice |
| `invoice_dollars` | Total dollar amount claimed in the invoice |
| `Freight` | Associated freight/shipping cost |
| `total_item_quantity` | Actual quantity of items physically received |
| `total_item_dollars` | Actual value of items physically received |
| `days_po_to_invoice` | Days elapsed between Purchase Order and Invoice |
| `days_to_pay` | Historical payment delay |
| `avg_receiving_delay`| Average warehouse receiving delay |

**Target Variable**: `flag_invoice` (0 = Safe, 1 = Flagged)

---

## 🛠️ Tech Stack
* **Programming**: Python 3.10+
* **Framework**: [Streamlit](https://streamlit.io/) (Frontend Dashboard)
* **Machine Learning**: Scikit-Learn, SciPy
* **Data Processing & Viz**: Pandas, NumPy, Matplotlib, Seaborn
* **Database & Storage**: SQLite, Joblib

---

## 📁 Project Structure
```text
In Project/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Project dependencies
├── data/
│   └── inventory.db            # Source database (SQLite)
├── models/                     # Saved ML models (.pkl)
│   ├── predict_freight_model.pkl
│   ├── invoice_risk_model.pkl
│   └── invoice_scaler.pkl
├── freight_cost_prediction/    # Freight Prediction Module
│   ├── train.py                
│   └── inference/              
│       └── predict_freight.py
└── invoice_flagging/           # Invoice Risk Module
    ├── train.py                
    └── inference/              
        └── predict_invoice_risk.py

```

---

# ⚙️ Installation & Usage
## 1. Clone the repository

```bash
git clone [https://github.com/ashishjaiswar21/Vendor-Invoice-Intelligence-Portal.git](https://github.com/ashishjaiswar21/Vendor-Invoice-Intelligence-Portal.git)
cd Vendor-Invoice-Intelligence-Portal
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Run the Streamlit App

```bash
streamlit run app.py
```
---

## ⭐ Future Scope
### Add authentication / login system for finance managers.

### Deploy backend API on AWS or Render.

### Integrate deep learning anomaly detection models.

### Connect to a live, real-time invoice SQL database.

## 🧑‍💻 Author
Ashish Kumar Jaiswar B.Tech IT Student at Netaji Subhas University of Technology (NSUT) | Machine Learning & Android Developer Skills: Python, Machine Learning, Data Analysis, Kotlin, Jetpack Compose, FastAPI, SQL.

## 📜 License
This project is licensed under the MIT License.

