# MLOps Pipeline: Housing Price Prediction

---

## 📌 Project Overview
This project implements a robust **Machine Learning Operations (MLOps) pipeline** to predict California Housing prices using a Random Forest Regressor. 

The pipeline focuses on reproducibility, automation, and experiment tracking. It creates a seamless workflow from raw data ingestion to model training and evaluation, orchestrating distinct steps into a cohesive lifecycle.

## 🛠️ Tech Stack & Architecture

This project adapts industry-standard tools to create a lightweight, reproducible environment on macOS:

| Component | Tool Used | Purpose |
| :--- | :--- | :--- |
| **Orchestration** | **MLflow Projects** | Replaces Kubeflow to manage the pipeline workflow and parameter logging locally. |
| **Data Versioning** | **DVC (Data Version Control)** | Tracks large datasets and ensures data lineage (simulated S3 bucket locally). |
| **Model Training** | **Scikit-Learn** | Random Forest Regressor for tabular data prediction. |
| **CI/CD** | **GitHub Actions** | Replaces Jenkins to automatically test code syntax and dependencies on every push. |
| **Language** | **Python 3.9+** | Core scripting language. |

---

## 📂 Project Structure

```text
├── .github/workflows
│   └── ci.yml             # CI/CD pipeline configuration (GitHub Actions)
├── data/
│   ├── raw_data.csv       # Versioned input dataset
│   └── processed_data.csv # Generated output from ETL step
├── src/
│   ├── etl.py             # Script: Data extraction, cleaning, and splitting
│   ├── train.py           # Script: Model training and RMSE calculation
│   └── main.py            # Driver script connecting ETL and Training steps
├── MLproject              # MLflow definition file (Entry points & Parameters)
├── requirements.txt       # Project dependencies
├── Dockerfile             # Container configuration
└── README.md              # Project documentation
