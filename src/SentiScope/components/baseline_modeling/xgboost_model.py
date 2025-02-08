from SentiScope.components.baseline_modeling.sklearn_wrapper import SklearnModelWrapper
import xgboost as xgb


class XGBoostModel(SklearnModelWrapper):
    """XGBoost implementation using sklearn API"""
    
    def __init__(self, num_classes=3, params=None):
        default_params = {
            'objective': 'multi:softmax',
            'num_class': num_classes,
            'eta': 0.3,
            'max_depth': 6,
            'eval_metric': 'merror'

        }
        if params:
            default_params.update(params)
        super().__init__(xgb.XGBClassifier(**default_params))
        
# Example Usage (In the pipeline):   
# pipeline.models['xgboost'] = XGBoostModel(num_classes=3, params=xgb_params)
# results = pipeline.train_models()