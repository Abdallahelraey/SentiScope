from sklearn.linear_model import LogisticRegression
from SentiScope.components.baseline_modeling.base_model import BaseModel
from sklearn.svm import SVC


class SklearnModelWrapper(BaseModel):
    """Wrapper for scikit-learn models to conform to our interface"""
    
    def __init__(self, model):
        self.model = model
    
    def train(self, X_train, y_train):
        return self.model.fit(X_train, y_train)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def get_params(self):
        return self.model.get_params()
    
    def set_params(self, **params):
        return self.model.set_params(**params)