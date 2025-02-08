from SentiScope.components.baseline_modeling.sklearn_wrapper import SklearnModelWrapper
from sklearn.linear_model import LogisticRegression

class LogisticRegressionModel(SklearnModelWrapper):
    """Logistic Regression implementation"""
    
    def __init__(self):
        super().__init__(LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000))