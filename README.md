# 🧠 BigMart Sales Prediction — Advanced Regression Pipeline

## 📋 Overview
This project builds a full **end-to-end regression workflow** to predict product sales across retail outlets. It combines **feature engineering**, **data cleaning**, and **model optimization** to demonstrate practical machine learning design for structured business data. The notebook follows a clear pedagogical structure, ideal for learners, educators, and applied data scientists.

---

## 🚀 Features
- **Comprehensive preprocessing:** missing-value imputation, categorical encoding, visibility ratio, and outlet age.
- **Target transformations:** log, **Yeo-Johnson**, and **Box-Cox** compared for variance stabilization.
- **Model suite:** Linear + Ridge, Decision Tree, Random Forest (baseline → tuned → interactions), Gradient Boosting, and **XGBoost**.
- **Cross-validation framework:** reusable 5-fold RMSE evaluator.
- **Hyperparameter optimization:** RandomizedSearchCV → GridSearchCV with visual comparison.
- **Performance visualization:** feature importances, convergence curves, and before/after RMSE charts.
- **Final ensemble summary:** automated model ranking and CSV export guardrails.

---

## 🧩 Technologies
`Python 3.10`, `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `xgboost`, `scipy`  
Notebook: Google Colab / Jupyter compatible

---

## 📊 Results
- **Baseline RMSE:** ≈ 1706  
- **Best model:** Random Forest (+ engineered interactions) — RMSE ≈ 1080  
- **Key insight:** Feature combinations such as `Item_MRP × Outlet_Years` and transformed targets significantly stabilized variance and improved generalization.

---

## 🧱 Repository Structure
```
📁 BigMart-Sales-Prediction
 ├── BigMart_Sales_Pipeline.ipynb     # main Colab notebook
 ├── data/                            # train/test/sample_submission CSVs
 ├── results/                         # figures, grid/random search tables
 ├── README.md                        # this file
 ├── LICENSE                          # MIT License
 └── requirements.txt (optional)
```

---

## 🧪 How to Run
```bash
# Clone repository
git clone https://github.com/<your-username>/BigMart-Sales-Prediction.git
cd BigMart-Sales-Prediction

# Install dependencies
pip install -r requirements.txt

# Open notebook
jupyter notebook BigMart_Sales_Pipeline.ipynb
```

Or open directly in **Google Colab** → upload your dataset CSVs and run all cells sequentially.

---

## 🧾 License
Released under the **MIT License** – free for educational and research use with attribution.
