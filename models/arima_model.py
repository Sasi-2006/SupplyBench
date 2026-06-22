from statsmodels.tsa.arima.model import ARIMA
from models.base_model import BaseModel
class ARIMAModel(BaseModel):
    def fit(self, train):
        self.model = ARIMA(
            train,
            order=(5, 1, 0)
        )
        self.fitted_model = self.model.fit()
    def predict(self, steps):
        predictions = self.fitted_model.forecast(
            steps=steps
        )
        return predictions