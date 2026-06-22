import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Input
from models.base_model import BaseModel


class LSTMModel(BaseModel):

    def fit(self, train):

        self.window = 7

        # Scale data
        self.scaler = MinMaxScaler()

        values = train.values.reshape(-1, 1)

        values = self.scaler.fit_transform(
            values
        )

        values = values.flatten()

        X = []
        y = []

        for i in range(
            self.window,
            len(values)
        ):

            X.append(
                values[
                    i-self.window:i
                ]
            )

            y.append(
                values[i]
            )

        X = np.array(X)
        y = np.array(y)

        X = X.reshape(
            (
                X.shape[0],
                X.shape[1],
                1
            )
        )
        self.model = Sequential([
            Input(shape = (self.window,1)),
            LSTM(32),
            Dense(1)
        ])


        self.model.add(
            Dense(1)
        )

        self.model.compile(
            optimizer="adam",
            loss="mse"
        )

        self.model.fit(
            X,
            y,
            epochs=30,
            verbose=0
        )

        self.history = list(values)

    def predict(self, steps):

        predictions = []

        history = self.history.copy()

        for _ in range(steps):

            x = np.array(
                history[-self.window:]
            )

            x = x.reshape(
                (
                    1,
                    self.window,
                    1
                )
            )

            pred = self.model.predict(
                x,
                verbose=0
            )[0][0]

            predictions.append(pred)

            history.append(pred)

        predictions = np.array(
            predictions
        ).reshape(-1, 1)

        predictions = (
            self.scaler.inverse_transform(
                predictions
            )
        )

        return predictions.flatten()