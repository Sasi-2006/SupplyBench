from prophet import Prophet
from models.base_model import BaseModel

import pandas as pd


class ProphetModel(BaseModel):

    def fit(self, train):

        self.train_length = len(train)

        df = pd.DataFrame({
            "ds": pd.date_range(
                start="2020-01-01",
                periods=len(train),
                freq="D"
            ),
            "y": train.values
        })

        self.model = Prophet()

        self.model.fit(df)

    def predict(self, steps):

        future = self.model.make_future_dataframe(
            periods=steps
        )

        forecast = self.model.predict(
            future
        )

        return (
            forecast["yhat"]
            .tail(steps)
            .values
        )