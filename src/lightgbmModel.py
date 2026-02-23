import lightgbm as lgb
import pandas as pd
import numpy as np
from typing import Dict, Optional
import time


class LightGBMForecaster:
    #LightGBM sales forecaster
    
    DEFAULT_PARAMS = {
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 42
    }
    
    def __init__(self, params: Optional[Dict] = None):
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.model_ = None
        self.featureImportance_ = None
        self.trainingTime_ = None
    
    def fit(self, xTrain: pd.DataFrame, yTrain: pd.Series, xValid: pd.DataFrame = None, 
            yValid: pd.Series = None, numBoostRound: int = 500, earlyStoppingRounds: int = 50,
            verboseEval: int = 100) -> 'LightGBMForecaster':
        
        trainSet = lgb.Dataset(xTrain, yTrain)
        validSets = [trainSet]
        
        if xValid is not None:
            validSet = lgb.Dataset(xValid, yValid, reference=trainSet)
            validSets.append(validSet)
        
        start = time.time()
        self.model_ = lgb.train(
            self.params, trainSet, num_boost_round=numBoostRound,
            valid_sets=validSets,
            callbacks=[lgb.early_stopping(earlyStoppingRounds, verbose=False), lgb.log_evaluation(verboseEval)]
        )
        self.trainingTime_ = time.time() - start
        
        self.featureImportance_ = pd.DataFrame({
            'feature': xTrain.columns.tolist(),
            'importance': self.model_.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(X)
    
    def getFeatureImportance(self, topN: int = 15) -> pd.DataFrame:
        return self.featureImportance_.head(topN)