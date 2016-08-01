from sklearn.pipeline import Pipeline
from models.common import FeatureRemover
from models.common import SomeLinearWrapper
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesRegressor


et = ExtraTreesRegressor(n_estimators=400, min_samples_leaf=3)
et.name = "et"
et = SomeLinearWrapper(et)

