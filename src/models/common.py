import sklearn
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn import linear_model
import xgboost
from xgboost import DMatrix
from xgboost import train
import pandas as pd


class Pipeline(sklearn.pipeline.Pipeline):

    def fit_transform_only(self, x, y=None):
        xt = x
        for name, transform in self.steps[:-1]:
            xt = transform.fit(xt).transform(xt)
        return xt


# for some powerlines, we just want to use a simple linear model. do it here, so
# we don't have to implement this for all of our models
class SomeLinearWrapper():

    def __init__(self, m):
        self.m = m
        self.name = m.name
        self.linear_powerlines = {"NPWD2691", "NPWD2692", "NPWD2871", "NPWD2471",
                                  "NPWD2872", "NPWD2472", "NPWD2801", "NPWD2882"}
        self.linear_model = sklearn.base.clone(model_ridge)

    def fit(self, x, y):
        self.cols_to_predict = list(set(y.columns) - self.linear_powerlines)
        self.linear_cols = list(set(y.columns) & self.linear_powerlines)
        
        if len(self.linear_cols) > 0:
            self.linear_model.fit(x, y[self.linear_cols])
        self.m.fit(x, y[self.cols_to_predict])
        return self

    def predict(self, x):
        preds = pd.DataFrame(self.m.predict(x), index=x.index, columns=self.cols_to_predict)
        if len(self.linear_cols) > 0:
            linear_preds = pd.DataFrame(self.linear_model.predict(x), index=x.index, columns=self.linear_cols)
            preds = pd.concat([linear_preds, preds], axis=1)
        return preds


class VectorRegression(sklearn.base.BaseEstimator):
    def __init__(self, estimator):
        self.estimator = estimator
        self._estimators = {}
        self.increase = 100

    def fit(self, x, y):
        # Fit a separate regressor for each column of y
        iterator = tqdm(y.columns) if len(y.columns) > 1 else y.columns
        for col in iterator:
            y_col = y[col] * self.increase
            m = sklearn.base.clone(self.estimator)
            m.fit(x, y_col)
            self._estimators[col] = m
        return self

    def predict(self, x):
        # Join regressors' predictions
        preds = []
        for col in cols_to_predict:
            preds.append(pd.Series(self._estimators[col].predict(x)) / self.increase)
        pred = pd.concat(preds, axis=1)
        pred.index = x.index
        pred.columns = cols_to_predict
        return pred


class ScaleAndNorm:

    def __init__(self):
        self.scaler = preprocessing.StandardScaler()
        self.normer = preprocessing.Normalizer()

    def transform(self, x):
        return pd.DataFrame(self.normer.transform(self.scaler.transform(x)), index=x.index, columns=x.columns)

    def fit(self, x, y=None):
        self.normer.fit(self.scaler.fit_transform(x))
        return self


class FeatureRemover:

    def __init__(self, cols):
        self.cols = cols

    def transform(self, x):
        x = x.copy()
        for col in self.cols:
            if col in x.columns:
                del x[col]
        return x

    def fit(self, x, y=None):
        return self


model_ridge = Pipeline([
      ("drop", FeatureRemover(["ATTB"])),
      ("dropna", preprocessing.Imputer()),
      ("scale", preprocessing.StandardScaler()),
      #("norm", preprocessing.Normalizer()),
      ("ridge", linear_model.Ridge(normalize=True, fit_intercept=True, alpha = 0.4)),
      ])

model_ridge.name = "ridge"


# subclassed to play with eta decay and dart booster
class MyXGBRegressor(xgboost.XGBRegressor):


    # overwriting to get desired behaviour
    def fit(self, X, y, eval_set=None, eval_metric=None,
            early_stopping_rounds=None, verbose=True):
        # pylint: disable=missing-docstring,invalid-name,attribute-defined-outside-init
        """
        Fit the gradient boosting model

        Parameters
        ----------
        X : array_like
            Feature matrix
        y : array_like
            Labels
        eval_set : list, optional
            A list of (X, y) tuple pairs to use as a validation set for
            early-stopping
        eval_metric : str, callable, optional
            If a str, should be a built-in evaluation metric to use. See
            doc/parameter.md. If callable, a custom evaluation metric. The call
            signature is func(y_predicted, y_true) where y_true will be a
            DMatrix object such that you may need to call the get_label
            method. It must return a str, value pair where the str is a name
            for the evaluation and value is the value of the evaluation
            function. This objective is always minimized.
        early_stopping_rounds : int
            Activates early stopping. Validation error needs to decrease at
            least every <early_stopping_rounds> round(s) to continue training.
            Requires at least one item in evals.  If there's more than one,
            will use the last. Returns the model from the last iteration
            (not the best one). If early stopping occurs, the model will
            have two additional fields: bst.best_score and bst.best_iteration.
        verbose : bool
            If `verbose` and an evaluation set is used, writes the evaluation
            metric measured on the validation set to stderr.
        """
        trainDmatrix = DMatrix(X, label=y, missing=self.missing)

        evals_result = {}
        if eval_set is not None:
            evals = list(DMatrix(x[0], label=x[1]) for x in eval_set)
            evals = list(zip(evals, ["validation_{}".format(i) for i in
                                     range(len(evals))]))
        else:
            evals = ()

        params = self.get_xgb_params()

        feval = eval_metric if callable(eval_metric) else None
        if eval_metric is not None:
            if callable(eval_metric):
                eval_metric = None
            else:
                params.update({'eval_metric': eval_metric})

        params.update({'booster': 'dart', 'rate_drop': 0.3, 'skip_drop': 0.7, 'sample_type': 'weighted',})

        #learning_rates = [0.2]*50 + [0.1]*50 +[0.05]*200 + [0.03]*200 + [0.01]*100 + [0.005] * 2000
        #learning_rates = learning_rates[:self.n_estimators]
        learning_rates = None

        self._Booster = train(params, trainDmatrix,
                              self.n_estimators, evals=evals,
                              early_stopping_rounds=early_stopping_rounds,
                              evals_result=evals_result, feval=feval,
                              verbose_eval=verbose, learning_rates=learning_rates)

        if evals_result:
            for val in evals_result.items():
                evals_result_key = list(val[1].keys())[0]
                evals_result[val[0]][evals_result_key] = val[1][evals_result_key]
            self.evals_result_ = evals_result

        if early_stopping_rounds is not None:
            self.best_score = self._Booster.best_score
            self.best_iteration = self._Booster.best_iteration
        return self
