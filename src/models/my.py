import sklearn
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
import xgboost
from tqdm import tqdm
import pandas as pd
from models.common import SomeLinearWrapper
from models.common import FeatureRemover

        
#http://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn
gbm = Pipeline([
    ("drop", FeatureRemover(["UPBS", "UPBE", "SCMN", "earthmars_km", "sunmars_km", "OCC_MARS_200KM_START_", "sa_monthly"])),
    ("scale", preprocessing.StandardScaler()),
    ("norm", preprocessing.Normalizer()),
    ("gbm", xgboost.XGBRegressor(max_depth=5, n_estimators=250, learning_rate=0.05, silent=1, seed=42))
    ])



NPWD2372 = Pipeline([
    ("drop", FeatureRemover(["UPBS", "UPBE", "SCMN", "earthmars_km", "sunmars_km", "OCC_MARS_200KM_START_",
                             "sa_monthly", "sx_monthly", "sy_monthly", "sz_monthly", "AOS", "LOS",
                             "ATTA", "ATTB"])),
    ("scale", preprocessing.StandardScaler()),
    ("norm", preprocessing.Normalizer()),
    ("gbm", xgboost.XGBRegressor(max_depth=5, n_estimators=250, learning_rate=0.05, silent=1, seed=42))
    ])

NPWD2401 = Pipeline([
    ("drop", FeatureRemover(["UPBS", "UPBE", "SCMN", "earthmars_km", "sunmars_km", "OCC_MARS_200KM_START_",
                             "sa_monthly", "sx_monthly", "sz_monthly", "AOS", "LOS", "ATTB"])),
    ("scale", preprocessing.StandardScaler()),
    ("norm", preprocessing.Normalizer()),
    ("gbm", xgboost.XGBRegressor(max_depth=5, n_estimators=250, learning_rate=0.05, silent=1, seed=42))
    ])

NPWD2402 = Pipeline([
    ("drop", FeatureRemover(["UPBS", "UPBE", "SCMN", "earthmars_km", "sunmars_km", "OCC_MARS_200KM_START_",
                             "sa_monthly", "sx_monthly", "sy_monthly", "AOS"])),
    ("scale", preprocessing.StandardScaler()),
    ("norm", preprocessing.Normalizer()),
    ("gbm", xgboost.XGBRegressor(max_depth=5, n_estimators=250, learning_rate=0.05, silent=1, seed=42))
    ])


NPWD2451 = Pipeline([
    ("drop", FeatureRemover(["earthmars_km", "MAR_PENUMBRA_END", "sunmars_km",
                             "sa_monthly", "sx_monthly", "sy_monthly", "sz_monthly",
                             "AOS", "LOS", "occultationduration_min_monthly", "ATTA", "ATTB"])),
    ("scale", preprocessing.StandardScaler()),
    ("norm", preprocessing.Normalizer()),
    ("gbm", xgboost.XGBRegressor(max_depth=5, n_estimators=500, learning_rate=0.05, silent=1, seed=42))
    ])

    
NPWD2481 = Pipeline([
    ("drop", FeatureRemover(["UPBS", "UPBE", "SCMN", "earthmars_km", "sunmars_km", "OCC_MARS_200KM_START_",
                             "sa_monthly", "sy_monthly", "sz_monthly", "LOS",
                             "occultationduration_min_monthly"])),
    ("scale", preprocessing.StandardScaler()),
    ("norm", preprocessing.Normalizer()),
    ("gbm", xgboost.XGBRegressor(max_depth=5, n_estimators=250, learning_rate=0.05, silent=1, seed=42))
    ])
    
NPWD2482 = Pipeline([
    ("drop", FeatureRemover(["UPBS", "UPBE", "SCMN", "earthmars_km", "sunmars_km", "OCC_MARS_200KM_START_",
                             "sa_monthly", "sx_monthly", "AOS", "LOS"])),
    ("scale", preprocessing.StandardScaler()),
    ("norm", preprocessing.Normalizer()),
    ("gbm", xgboost.XGBRegressor(max_depth=5, n_estimators=250, learning_rate=0.05, silent=1, seed=42))
    ])
#TODO
NPWD2491 = Pipeline([
    ("drop", FeatureRemover(["UPBS", "UPBE", "SCMN", "earthmars_km", "sunmars_km", "OCC_MARS_200KM_START_",
                             "sa_monthly", "sz_monthly", "AOS", "occultationduration_min_monthly",
                             "ATTA", "ATTB"])),
    ("scale", preprocessing.StandardScaler()),
    ("norm", preprocessing.Normalizer()),
    ("gbm", xgboost.XGBRegressor(max_depth=5, n_estimators=250, learning_rate=0.05, silent=1, seed=42))
    ])
#TODO
NPWD2501 = Pipeline([
    ("drop", FeatureRemover(["UPBS", "UPBE", "SCMN", "earthmars_km", "sunmars_km", "OCC_MARS_200KM_START_",
                             "sa_monthly", "sz_monthly", "AOS", "LOS", "ATTA", "ATTB"])),
    ("scale", preprocessing.StandardScaler()),
    ("norm", preprocessing.Normalizer()),
    ("gbm", xgboost.XGBRegressor(max_depth=5, n_estimators=250, learning_rate=0.05, silent=1, seed=42))
    ])


NPWD2531 = Pipeline([
    ("drop", FeatureRemover(["earthmars_km", "sy-2", "2000_KM_DESCEND",
                             "sa_monthly", "sx_monthly", "sy_monthly", "AOS", "LOS",
                             "occultationduration_min_monthly", "ATTA"])),
    ("scale", preprocessing.StandardScaler()),
    ("norm", preprocessing.Normalizer()),
    ("gbm", xgboost.XGBRegressor(max_depth=5, n_estimators=500, learning_rate=0.05, silent=1, seed=42))
     ])
     
#TODO
NPWD2532 = Pipeline([
    ("drop", FeatureRemover(["UPBS", "UPBE", "SCMN", "earthmars_km", "sunmars_km", "OCC_MARS_200KM_START_",
                             "sa_monthly", "AOS", "LOS", "occultationduration_min_monthly", "ATTA",
                             "ATTB"])),
    ("scale", preprocessing.StandardScaler()),
    ("norm", preprocessing.Normalizer()),
    ("gbm", xgboost.XGBRegressor(max_depth=5, n_estimators=900, learning_rate=0.05, silent=1, seed=42))
     ])


# TODO
NPWD2551 = Pipeline([
    ("drop", FeatureRemover(["earthmars_km", "BLK_AOS_00_", "BLK_AOS_10_", "SCMN", "sunmars_km",
                             "MSL_LOS_03", "sa_monthly", "sx_monthly", "sy_monthly", "sz_monthly",
                             "AOS", "LOS", "occultationduration_min_monthly", "ATTA"])),
    ("scale", preprocessing.StandardScaler()),
    ("norm", preprocessing.Normalizer()),
    ("gbm", xgboost.XGBRegressor(max_depth=5, n_estimators=800, learning_rate=0.05, silent=1, seed=42))
    ])

  
    

#TODO    
NPWD2561 = Pipeline([
    ("drop", FeatureRemover(["UPBS", "UPBE", "SCMN", "earthmars_km", "sunmars_km", "OCC_MARS_200KM_START_",
                             "sa_monthly", "sz_monthly", "AOS", "occultationduration_min_monthly", "ATTA"])),
    ("scale", preprocessing.StandardScaler()),
    ("norm", preprocessing.Normalizer()),
    ("gbm", xgboost.XGBRegressor(max_depth=5, n_estimators=1000, learning_rate=0.02, silent=1, seed=42))
    ])
#TODO    
NPWD2562 = Pipeline([
    ("drop", FeatureRemover(["UPBS", "UPBE", "SCMN", "earthmars_km", "sunmars_km", "OCC_MARS_200KM_START_",
                             "AOS", "LOS", "occultationduration_min_monthly", "ATTB"])),
    ("scale", preprocessing.StandardScaler()),
    ("norm", preprocessing.Normalizer()),
    ("gbm", xgboost.XGBRegressor(max_depth=5, n_estimators=250, learning_rate=0.05, silent=1, seed=42))
    ])
    
#TODO
NPWD2721 = Pipeline([
    ("drop", FeatureRemover(["UPBS", "UPBE", "SCMN", "earthmars_km", "sunmars_km", "OCC_MARS_200KM_START_",
                             "sa_monthly", "sx_monthly", "sy_monthly", "sz_monthly", "AOS",
                             "occultationduration_min_monthly", "ATTA", "ATTB"])),
    ("scale", preprocessing.StandardScaler()),
    ("norm", preprocessing.Normalizer()),
    ("gbm", xgboost.XGBRegressor(max_depth=5, n_estimators=600, learning_rate=0.05, silent=1, seed=42))
    ])
#TODO
NPWD2722 = Pipeline([
    ("drop", FeatureRemover(["UPBS", "UPBE", "SCMN", "earthmars_km", "sunmars_km", "OCC_MARS_200KM_START_",
                             "sa_monthly", "sy_monthly", "sz_monthly", "AOS", "LOS",
                             "occultationduration_min_monthly", "ATTA", "ATTB"])),
    ("scale", preprocessing.StandardScaler()),
    ("norm", preprocessing.Normalizer()),
    ("gbm", xgboost.XGBRegressor(max_depth=5, n_estimators=110, learning_rate=0.05, silent=1, seed=42))
    ])
    
#TODO
NPWD2742 = Pipeline([
    ("drop", FeatureRemover(["UPBS", "UPBE", "SCMN", "earthmars_km", "sunmars_km", "OCC_MARS_200KM_START_",
                             "sa_monthly", "sx_monthly", "sy_monthly", "sz_monthly", "AOS", "LOS",
                             "occultationduration_min_monthly", "ATTA", "ATTB"])),
    ("scale", preprocessing.StandardScaler()),
    ("norm", preprocessing.Normalizer()),
    ("gbm", xgboost.XGBRegressor(max_depth=5, n_estimators=250, learning_rate=0.05, silent=1, seed=42))
    ])
#TODO
NPWD2771 = Pipeline([
    ("drop", FeatureRemover(["UPBS", "UPBE", "SCMN", "earthmars_km", "sunmars_km", "OCC_MARS_200KM_START_",
                             "sa_monthly", "sx_monthly", "AOS", "occultationduration_min_monthly", "ATTB"])),
    ("scale", preprocessing.StandardScaler()),
    ("norm", preprocessing.Normalizer()),
    ("gbm", xgboost.XGBRegressor(max_depth=5, n_estimators=250, learning_rate=0.05, silent=1, seed=42))
    ])
    
#TODO
NPWD2791 = Pipeline([
    ("drop", FeatureRemover(["UPBS", "UPBE", "SCMN", "earthmars_km", "sunmars_km", "OCC_MARS_200KM_START_",
                             "sy_monthly", "sz_monthly", "AOS", "occultationduration_min_monthly"])),
    ("scale", preprocessing.StandardScaler()),
    ("norm", preprocessing.Normalizer()),
    ("gbm", xgboost.XGBRegressor(max_depth=5, n_estimators=550, learning_rate=0.05, silent=1, seed=42))
    ])
    
# TODO
NPWD2792 = Pipeline([
    ("drop", FeatureRemover(["UPBS", "UPBE", "SCMN", "earthmars_km", "sunmars_km", "OCC_MARS_200KM_START_",
                             "sa_monthly", "sx_monthly", "sz_monthly", "occultationduration_min_monthly",
                             "ATTA", "ATTB"])),
    ("scale", preprocessing.StandardScaler()),
    ("norm", preprocessing.Normalizer()),
    ("gbm", xgboost.XGBRegressor(max_depth=5, n_estimators=250, learning_rate=0.05, silent=1, seed=42))
    ])
        
#TODO
NPWD2802 = Pipeline([
    ("drop", FeatureRemover(["UPBS", "UPBE", "SCMN", "earthmars_km", "sunmars_km", "OCC_MARS_200KM_START_",
                             "sx_monthly", "sy_monthly", "sz_monthly", "AOS", "LOS",
                             "occultationduration_min_monthly", "ATTA", "ATTB"])),
    ("scale", preprocessing.StandardScaler()),
    ("norm", preprocessing.Normalizer()),
    ("gbm", xgboost.XGBRegressor(max_depth=5, n_estimators=250, learning_rate=0.05, silent=1, seed=42))
    ])
#TODO
NPWD2821 = Pipeline([
    ("drop", FeatureRemover(["UPBS", "UPBE", "SCMN", "earthmars_km", "sunmars_km", "OCC_MARS_200KM_START_",
                             "sa_monthly", "sz_monthly", "occultationduration_min_monthly", "ATTB"])),
    ("scale", preprocessing.StandardScaler()),
    ("norm", preprocessing.Normalizer()),
    ("gbm", xgboost.XGBRegressor(max_depth=5, n_estimators=250, learning_rate=0.05, silent=1, seed=42))
    ])
    
NPWD2851 = Pipeline([
    ("drop", FeatureRemover(["UPBS", "UPBE", "SCMN", "earthmars_km", "sunmars_km", "OCC_MARS_200KM_START_",
                             "sy_monthly", "AOS", "occultationduration_min_monthly", "ATTB"])),
    ("scale", preprocessing.StandardScaler()),
    ("norm", preprocessing.Normalizer()),
    ("gbm", xgboost.XGBRegressor(max_depth=5, n_estimators=750, learning_rate=0.05, silent=1, seed=42))
    ])
    
#TODO
NPWD2881 = Pipeline([
    ("drop", FeatureRemover(["UPBS", "UPBE", "SCMN", "earthmars_km", "sunmars_km", "OCC_MARS_200KM_START_",
                             "sa_monthly", "sy_monthly", "LOS", "occultationduration_min_monthly", "ATTA"])),
    ("scale", preprocessing.StandardScaler()),
    ("norm", preprocessing.Normalizer()),
    ("gbm", xgboost.XGBRegressor(max_depth=5, n_estimators=500, learning_rate=0.01, silent=1, seed=42))
    ])
    

class MyNodel(sklearn.base.BaseEstimator):
    def __init__(self):
        self.default_estimator = gbm
        self.seed = 42
        self.name = "my"
        self.estimators = {
                           "NPWD2372": NPWD2372,
                           "NPWD2401": NPWD2401,
                           "NPWD2402": NPWD2402,
                           "NPWD2451": NPWD2451,
                           "NPWD2481": NPWD2481,
                           "NPWD2482": NPWD2482,
                           "NPWD2491": NPWD2491,
                           "NPWD2501": NPWD2501,
                           "NPWD2531": NPWD2531,
                           "NPWD2532": NPWD2532,
                           "NPWD2551": NPWD2551,
                           "NPWD2561": NPWD2561,
                           "NPWD2562": NPWD2562,
                           "NPWD2721": NPWD2721,
                           "NPWD2722": NPWD2722,
                           "NPWD2742": NPWD2742,
                           "NPWD2771": NPWD2771,
                           "NPWD2791": NPWD2791,
                           "NPWD2792": NPWD2792,
                           "NPWD2802": NPWD2802,
                           "NPWD2821": NPWD2821,
                           "NPWD2851": NPWD2851,
                           "NPWD2881": NPWD2881,
                           }
        self.increase = 100.0

    def fit(self, x, y):
        self.cols_to_predict = y.columns
        iterator = tqdm(self.cols_to_predict) if len(self.cols_to_predict) > 1 else self.cols_to_predict
        for col in iterator:
            # TODO try removing
            y_col = y[col] * self.increase
            if col in self.estimators:
                m = self.estimators[col]
            else:
                m = sklearn.base.clone(self.default_estimator)
                
            m.fit(x, y_col)
            self.estimators[col] = m
        return self

    def predict(self, x):
        # Join regressors' predictions
        preds = []
        for col in self.cols_to_predict:
            preds.append(pd.Series(self.estimators[col].predict(x)) / self.increase)
        pred = pd.concat(preds, axis=1)
        pred.index = x.index
        pred.columns = self.cols_to_predict
        return pred


my_model = SomeLinearWrapper(MyNodel())
