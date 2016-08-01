import pandas as pd
import numpy as np
import os
import common
from scipy.optimize import minimize


def average(models, oob=True):
    dfs = [common.load_preds(m, oob) for m in models]
    df = pd.concat(dfs)
    df = df.groupby(df.index).mean()
    common.save_preds("ensemble_average", df)
    print("ensemble", common.rmse(common.y_all, df))
    return df


def weighted_average(models=weights, oob=True):
    dfs = []
    for m, perc in models:
        preds = common.load_preds(m, oob)
        print(m, common.rmse(preds, common.y_all))
        df = preds * perc
        dfs.append(df)
    df = pd.concat(dfs)
    df = df.groupby(df.index).sum()
    print("ensemble", common.rmse(common.y_all, df))

    common.save_preds("ensemble_weighted", df, oob=oob)
    return df


def best_weights(models):

    predictions = []
    for m in models:
        preds = common.load_preds(m)
        predictions.append(preds)
        print(m, common.rmse(common.y_all, preds))

    def objective_fun(weights):
        final_prediction = 0
        for weight, prediction in zip(weights, predictions):
            final_prediction += weight * prediction
        return common.rmse(common.y_all, final_prediction)
    
    for i in range(5):
        starting_values = np.random.random(len(models))
        starting_values /= starting_values.sum()

        #adding constraints  and a different solver as suggested by user 16universe
        #https://kaggle2.blob.core.windows.net/forum-message-attachments/75655/2393/otto%20model%20weights.pdf?sv=2012-02-12&se=2015-05-03T21%3A22%3A17Z&sr=b&sp=r&sig=rkeA7EJC%2BiQ%2FJ%2BcMpcA4lYQLFh6ubNqs2XAkGtFsAv0%3D
        cons = ({'type': 'eq','fun': lambda w: 1-sum(w)})
        #our weights are bound between 0 and 1
        bounds = [(0, 1)] * len(predictions)

        res = minimize(objective_fun, starting_values, method='SLSQP', bounds=bounds, constraints=cons)

        print('score: {best_score}'.format(best_score=res['fun']))
        print('weights: {weights}'.format(weights=res['x']))
        

if __name__ == "__main__":
    best_weights(["my", "onegbm", "nn", "lstm", "et", "ridge"])
    weighted_average([("lstm", 0.1), ("nn", 0.12), ("onegbm", 0.3), ("my", 0.45), ("ridge", 0.03)],)
