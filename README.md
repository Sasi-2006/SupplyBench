# SupplyBench

## AI-Powered Benchmarking Framework for Supply Chain Demand Forecasting

### Overview

SupplyBench is a benchmarking framework designed to evaluate and compare multiple time-series forecasting models for supply chain demand prediction.

The framework automatically trains, evaluates, and ranks forecasting algorithms using rolling time-series cross-validation and standard forecasting metrics.

This project was developed as a hackathon prototype to help organizations identify the most accurate forecasting model for their sales and demand data.

---

## Problem Statement

Demand forecasting is a critical task in supply chain management.

Inaccurate forecasts can lead to:

* Inventory shortages
* Overstocking
* Increased storage costs
* Delayed deliveries
* Revenue loss

Organizations often struggle to determine which forecasting model performs best on their historical sales data.

SupplyBench solves this problem by providing an automated benchmarking framework that compares multiple forecasting algorithms under the same evaluation conditions.

---

## Key Features

### Data Processing

* Automated dataset loading
* Date parsing and preprocessing
* Daily sales aggregation
* Time-series preparation

### Forecasting Models

The framework currently supports:

* ARIMA
* ETS (Exponential Smoothing)
* XGBoost
* Prophet
* LSTM Neural Network

### Evaluation Metrics

Models are evaluated using:

* MAE (Mean Absolute Error)
* RMSE (Root Mean Squared Error)
* SMAPE (Symmetric Mean Absolute Percentage Error)
* R² Score

### Validation Strategy

* Rolling Time-Series Cross Validation
* 5 Validation Folds
* Confidence Interval Estimation

### Reporting

* Benchmark Summary
* Best Model Selection
* CSV Result Export
* Performance Visualization

---

## Project Structure

```text
SupplyBench/
│
├── data/
│   ├── dataco.csv
│   └── rossmann.csv
│
├── models/
│   ├── base_model.py
│   ├── arima_model.py
│   ├── ets_model.py
│   ├── xgboost_model.py
│   ├── prophet_model.py
│   └── lstm_model.py
│
├── benchmark.py
├── data_loader.py
├── metrics.py
├── benchmark_results.csv
├── benchmark_comparison.png
├── requirements.txt
└── README.md
```

---

## Datasets Used

### 1. DataCo Supply Chain Dataset

The dataset contains:

* Customer Orders
* Product Information
* Shipping Information
* Sales Records

Used for:

* Supply Chain Forecasting
* Demand Prediction
* Inventory Planning

---

### 2. Rossmann Store Sales Dataset

The dataset contains:

* Daily Store Sales
* Promotion Information
* Historical Retail Data

Used for:

* Retail Forecasting
* Benchmark Validation

---

## Installation

### Clone Repository

```bash
git clone https://github.com/Sasi-2006/SupplyBench.git

cd SupplyBench
```

---

### Create Virtual Environment

```bash
python -m venv venv
```

---

### Activate Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / Mac

```bash
source venv/bin/activate
```

---

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Required Libraries

```text
pandas
numpy
scikit-learn
statsmodels
xgboost
prophet
tensorflow
matplotlib
```

---

## Running the Benchmark

Execute:

```bash
py -3.11 benchmark.py
```

or

```bash
python benchmark.py
```

---

## Benchmark Pipeline

### Step 1

Load dataset.

### Step 2

Convert dates into time-series format.

### Step 3

Aggregate daily sales.

### Step 4

Perform rolling cross-validation.

### Step 5

Train forecasting models.

### Step 6

Generate future predictions.

### Step 7

Calculate evaluation metrics.

### Step 8

Compute confidence intervals.

### Step 9

Export benchmark results.

### Step 10

Identify best-performing model.

---

## Evaluation Metrics

### MAE

Mean Absolute Error

Measures average forecasting error.

Lower is better.

---

### RMSE

Root Mean Squared Error

Penalizes larger forecasting errors.

Lower is better.

---

### SMAPE

Symmetric Mean Absolute Percentage Error

Measures percentage forecasting error.

Lower is better.

---

### R² Score

Measures goodness of fit.

Higher is better.

---

## Example Benchmark Output

```text
DATASET: DataCo

Running Model: ARIMA
Fold 1: MAE = 2684

Running Model: Prophet
Fold 1: MAE = 2588

Running Model: XGBoost
Fold 1: MAE = 2735
```

---

## Output Files

### benchmark_results.csv

Contains:

* Dataset Name
* Model Name
* Mean MAE
* Mean RMSE
* Mean SMAPE
* Mean R²
* Confidence Intervals

---

### benchmark_comparison.png

Visual comparison of forecasting model performance.

---

## Sample Results

| Model   | Mean MAE |
| ------- | -------- |
| ARIMA   | 5375     |
| ETS     | 5364     |
| XGBoost | 5171     |
| Prophet | 5274     |

Best Model:

**XGBoost**

---

## Applications

SupplyBench can be used for:

* Supply Chain Optimization
* Inventory Forecasting
* Retail Analytics
* Manufacturing Planning
* Logistics Forecasting
* Demand Prediction

---

## Future Enhancements

* Streamlit Dashboard
* Real-Time Forecasting
* Hyperparameter Tuning
* AutoML Integration
* Transformer-Based Forecasting Models
* Multi-Dataset Benchmarking
* Cloud Deployment

---

## Authors

Developed as a Hackathon Prototype Project.

---

## License

This project is intended for academic, educational, and research purposes.
