from statsmodels.tsa.holtwinters import ExponentialSmoothing
from models.base_model import BaseModel
class ETSModel(BaseModel):
    def fit(self, train):
        self.model = ExponentialSmoothing(
            train,
            trend="add"
        )
        self.fitted_model = self.model.fit()
    def predict(self, steps):
        return self.fitted_model.forecast(
            steps
        )