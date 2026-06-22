import pandas as pd

from xgboost import XGBRegressor

from models.base_model import BaseModel


class XGBoostModel(BaseModel):

    def create_features(self, series):

        df = pd.DataFrame()

        df["y"] = series

        df["lag1"] = df["y"].shift(1)
        df["lag2"] = df["y"].shift(2)
        df["lag3"] = df["y"].shift(3)

        df = df.dropna()

        return df

    def fit(self, train):

        df = self.create_features(train)

        X = df[
            ["lag1", "lag2", "lag3"]
        ]

        y = df["y"]

        self.model = XGBRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        )

        self.model.fit(X, y)

        self.history = list(train)

    def predict(self, steps):

        predictions = []

        history = self.history.copy()

        for _ in range(steps):

            X = [[
                history[-1],
                history[-2],
                history[-3]
            ]]

            pred = self.model.predict(X)[0]

            predictions.append(pred)

            history.append(pred)

        return predictions