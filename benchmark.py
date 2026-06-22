import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from metrics import evaluate
from models.arima_model import ARIMAModel
from models.ets_model import ETSModel
from models.xgboost_model import XGBoostModel
from models.prophet_model import ProphetModel
from models.lstm_model import LSTMModel
from data_loader import (
    load_dataco,
    load_rossmann
)

# ==========================================
# LOAD DATASETS
# ==========================================

datasets = {
    "DataCo": load_dataco(),

    # Uncomment after Rossmann is fixed
    "Rossmann": load_rossmann()
}

# ==========================================
# CROSS VALIDATION
# ==========================================

tscv = TimeSeriesSplit(
    n_splits=5
)

# ==========================================
# MODELS
# ==========================================

models = {
    "ARIMA": ARIMAModel(),
    "ETS": ETSModel(),
    "XGBoost": XGBoostModel(),
    "Prophet": ProphetModel(),
    "LSTM": LSTMModel()
}

# ==========================================
# STORE RESULTS
# ==========================================

all_results = {}

# ==========================================
# BENCHMARK LOOP
# ==========================================

for dataset_name, series in datasets.items():

    print("\n")
    print("=" * 60)

    print(
        f"DATASET: {dataset_name}"
    )

    print("=" * 60)

    print(
        f"Number of Records: {len(series)}"
    )

    for model_name, model in models.items():

        print("\n")
        print("-" * 40)

        print(
            f"Running Model: {model_name}"
        )

        print("-" * 40)

        results = []

        for fold, (
            train_idx,
            test_idx
        ) in enumerate(
            tscv.split(series),
            start=1
        ):

            train = series.iloc[
                train_idx
            ]

            test = series.iloc[
                test_idx
            ]

            model.fit(train)

            predictions = model.predict(
                len(test)
            )

            metrics = evaluate(
                test,
                predictions
            )

            results.append(
                metrics
            )

            print(
                f"Fold {fold}: {metrics}"
            )

        all_results[
            (dataset_name, model_name)
        ] = results

# ==========================================
# SUMMARY + CONFIDENCE INTERVALS
# ==========================================

print("\n")
print("=" * 60)
print("BENCHMARK SUMMARY")
print("=" * 60)

summary_rows = []

for (
    dataset_name,
    model_name
), results in all_results.items():

    mae_scores = [
        r["MAE"]
        for r in results
    ]

    rmse_scores = [
        r["RMSE"]
        for r in results
    ]

    smape_scores = [
        r["SMAPE"]
        for r in results
    ]

    r2_scores = [
        r["R2"]
        for r in results
    ]

    mean_mae = np.mean(
        mae_scores
    )

    mean_rmse = np.mean(
        rmse_scores
    )

    mean_smape = np.mean(
        smape_scores
    )

    mean_r2 = np.mean(
        r2_scores
    )

    std_mae = np.std(
        mae_scores
    )

    ci = (
        1.96 *
        std_mae /
        np.sqrt(
            len(mae_scores)
        )
    )

    print("\n")

    print(
        f"Dataset: {dataset_name}"
    )

    print(
        f"Model: {model_name}"
    )

    print(
        f"Mean MAE: {mean_mae:.2f}"
    )

    print(
        f"Mean RMSE: {mean_rmse:.2f}"
    )

    print(
        f"Mean SMAPE: {mean_smape:.2f}"
    )

    print(
        f"Mean R2: {mean_r2:.2f}"
    )

    print(
        f"95% MAE CI: "
        f"[{mean_mae-ci:.2f}, {mean_mae+ci:.2f}]"
    )

    summary_rows.append({

        "Dataset": dataset_name,

        "Model": model_name,

        "Mean_MAE": round(
            mean_mae,
            2
        ),

        "Mean_RMSE": round(
            mean_rmse,
            2
        ),

        "Mean_SMAPE": round(
            mean_smape,
            2
        ),

        "Mean_R2": round(
            mean_r2,
            2
        ),

        "CI_Lower": round(
            mean_mae - ci,
            2
        ),

        "CI_Upper": round(
            mean_mae + ci,
            2
        )
    })

# ==========================================
# EXPORT RESULTS
# ==========================================

summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    "benchmark_results.csv",
    index=False
)

print("\n")
print("=" * 60)

print(
    "Results exported successfully!"
)

print(
    "File: benchmark_results.csv"
)

print("=" * 60)

# ==========================================
# BEST MODEL
# ==========================================

best_model = summary_df.loc[
    summary_df["Mean_MAE"].idxmin()
]

print("\nBEST MODEL")

print(best_model)
# ==========================================
# VISUALIZATION
# ==========================================

plt.figure(figsize=(8, 5))

summary_df.plot(
    x="Model",
    y="Mean_MAE",
    kind="bar",
    legend=False
)

plt.title(
    "Forecasting Model Comparison"
)

plt.ylabel("Mean MAE")

plt.tight_layout()

plt.savefig(
    "benchmark_comparison.png"
)

print(
    "Chart saved as benchmark_comparison.png"
)