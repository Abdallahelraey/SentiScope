from sklearn.svm import SVC
from SentiScope.components.baseline_modeling.sklearn_wrapper import SklearnModelWrapper

class SVMModel(SklearnModelWrapper):
    """Support Vector Machine implementation"""
    
    def __init__(self):
        super().__init__(SVC(kernel='linear', probability=True))