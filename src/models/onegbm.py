import xgboost
import pandas as pd

from sklearn.pipeline import Pipeline
from models.common import FeatureRemover
from models.common import SomeLinearWrapper


class OneGbm:
    
    def __init__(self):
        self.name = "onegbm"
        self.m = Pipeline([
        ("drop", FeatureRemover(["UPBS", "UPBE", "SCMN", "earthmars_km", "OCC_MARS_200KM_START_", "sa_monthly"])),
        ("gbm", xgboost.XGBRegressor(max_depth=7, n_estimators=1000, learning_rate=0.05, silent=1, seed=42))
        ])
    
    def reformat(self, x, y):
        xs = []
        for col in self.cols_to_predict:
            df = x.copy()
            df['powerline'] = int(col[-4:])
            xs.append(df)
        xs = pd.concat(xs)
         
        if y is not None:
            ys = []
            for col in self.cols_to_predict:
                ys.append(y[col])
            ys = pd.concat(ys)
        else:
            ys = y
        
        return xs, ys
    
    def fit(self, x, y):
        self.cols_to_predict = y.columns
        x, y = self.reformat(x, y)
        self.m.fit(x, y)
        return self
        
    def predict(self, x, y = None):
        x, y = self.reformat(x, y)
        df = x[['powerline']].copy()
        df['pred'] = self.m.predict(x)
        df['powerline'] = "NPWD" + df['powerline'].astype(str)
        df = df.pivot(columns='powerline', values='pred')
        
        return df
        
        
onegbm = SomeLinearWrapper(OneGbm())
