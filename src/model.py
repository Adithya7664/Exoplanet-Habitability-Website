import numpy as np
import pandas as pd
import joblib
import shap
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from .features import FEATURE_COLS

MODEL_PATH = Path("models/ensemble.joblib")

class HabitabilityEnsemble:
    def __init__(self):
        self.rf = RandomForestRegressor(
              n_estimators=300, max_depth=12, min_samples_leaf=3,
              random_state=42, n_jobs=-1
          )
        self.xgb = XGBRegressor(
            n_estimators=300, max_depth=7, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, verbosity=0
        )
        self.scaler = StandardScaler()
        self.mlp = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32), max_iter=600,
            learning_rate_init=0.001, random_state=42
        )
        
    def _X(self, df: pd.DataFrame) -> np.ndarray:
        return df[FEATURE_COLS].fillna(0).values
    
    def fit(self, df: pd.DataFrame, y: pd.Series) -> "HabitabilityEnsemble":
          X = self._X(df)
          self.rf.fit(X, y.values)
          self.xgb.fit(X, y.values)
          self.mlp.fit(self.scaler.fit_transform(X), y.values)
          return self
      
    def predict(self, df: pd.DataFrame) -> np.ndarray:
          X = self._X(df)
          p_rf = self.rf.predict(X)
          p_xgb = self.xgb.predict(X)
          p_mlp = self.mlp.predict(self.scaler.transform(X))
          return (0.4 * p_rf + 0.4 * p_xgb + 0.2 * p_mlp).clip(0, 1)
      
    def predict_breakdown(self, df: pd.DataFrame)->dict:
        X=self._X(df)
        return {
            "Random Forest": self.rf.predict(X).clip(0, 1).tolist(),
            "XGBoost": self.xgb.predict(X).clip(0, 1).tolist(),
            "Neural Network": self.mlp.predict(self.scaler.transform(X)).clip(0, 1).tolist(),
        }
        
    
def shap_for_planet(model: HabitabilityEnsemble, row: pd.DataFrame):
    X_arr=row[FEATURE_COLS].fillna(0)
    rf_exp = shap.TreeExplainer(model.rf)
    xgb_exp = shap.TreeExplainer(model.xgb)
    rf_vals = rf_exp.shap_values(X_arr)
    xgb_vals = xgb_exp.shap_values(X_arr)
    avg_vals = 0.5 * rf_vals + 0.5 * xgb_vals
    avg_expected = float(0.5 * rf_exp.expected_value + 0.5 * xgb_exp.expected_value)
    return avg_vals[0], avg_expected

def train(df: pd.DataFrame)->HabitabilityEnsemble:
    mask=df["esi"].notna() & (df["esi"] >= 0) & (df["esi"] <= 1)
    X=df[mask]
    y=df.loc[mask, "esi"]
    model=HabitabilityEnsemble()
    model.fit(X, y)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    return model
    
def load_model() -> HabitabilityEnsemble:
    return joblib.load(MODEL_PATH)