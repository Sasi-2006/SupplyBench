import pandas as pd


def load_dataco():

    df = pd.read_csv(
        "data/dataco.csv",
        encoding="latin1"
    )

    date_column = "order date (DateOrders)"
    target_column = "Sales"

    df[date_column] = pd.to_datetime(
        df[date_column]
    )

    df["Date"] = df[
        date_column
    ].dt.date

    daily_sales = (
        df.groupby("Date")[target_column]
          .sum()
          .reset_index()
    )

    daily_sales["Date"] = pd.to_datetime(
        daily_sales["Date"]
    )

    daily_sales = daily_sales.sort_values(
        "Date"
    )

    return daily_sales["Sales"]


def load_rossmann():

    df = pd.read_csv(
        "data/rossmann.csv",
        low_memory = False
    )

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    daily_sales = (
        df.groupby("Date")["Sales"]
          .sum()
          .reset_index()
    )

    daily_sales = daily_sales.sort_values(
        "Date"
    )

    return daily_sales["Sales"]