import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from data import *
from tqdm import tqdm


# adjust pandas and numpy settings for better readable output
pd.set_option('max_columns', 50)
pd.set_option('display.width', 240)
np.set_printoptions(linewidth=240)
matplotlib.style.use('ggplot')



def rmse(val, pred):
    diff = (val - pred) ** 2
    return np.mean(diff.values) ** 0.5


def predict(m, x, cols=cols_to_predict):
    preds = pd.DataFrame(m.predict(x), index=x.index, columns=cols).clip_lower(0)
    return preds


def test_train_splits():
    """
    Year 1: 2008-08-22 to 2010-07-10
    Year 2: 2010-07-10 to 2012-05-27
    Year 3: 2012-05-27 to 2014-04-14
    """
    x1 = x_all[(x_all.index >= pd.Timestamp("2008-08-22")) & (x_all.index < pd.Timestamp("2010-07-10"))]
    x2 = x_all[(x_all.index >= pd.Timestamp("2010-07-10")) & (x_all.index < pd.Timestamp("2012-05-27"))]
    x3 = x_all[(x_all.index >= pd.Timestamp("2012-05-27")) & (x_all.index < pd.Timestamp("2014-04-14"))]
    
    y1 = y_all[(y_all.index >= pd.Timestamp("2008-08-22")) & (y_all.index < pd.Timestamp("2010-07-10"))]
    y2 = y_all[(y_all.index >= pd.Timestamp("2010-07-10")) & (y_all.index < pd.Timestamp("2012-05-27"))]
    y3 = y_all[(y_all.index >= pd.Timestamp("2012-05-27")) & (y_all.index < pd.Timestamp("2014-04-14"))]
    res = [(pd.concat([x1, x2]), pd.concat([y1, y2]), x3, y3),
           (pd.concat([x1, x3]), pd.concat([y1, y3]), x2, y2),
           (pd.concat([x2, x3]), pd.concat([y2, y3]), x1, y1)]
    res = [(x_train, y_train, x_test, y_test) for x_train, y_train, x_test, y_test in res if len(x_train) > 0 and len(x_test) > 0]
    res = [(x_train, y_train.loc[x_train.index], x_test, y_test.loc[x_test.index]) for x_train, y_train, x_test, y_test in res]
    return res


def fitandpredict(m, cols=cols_to_predict):
    splits = test_train_splits()
    preds = []
    for x_train, y_train, x_test, y_test in splits:
        y_train = y_train[cols]
        y_test = y_test[cols]
        pred = predict(m.fit(x_train, y_train), x_test, cols)
        print(rmse(y_test, pred))
        preds.append(pred)
    preds = pd.concat(preds)
    save_preds(m.name, preds, cols=cols)
    return preds
    
    
def validate_model(m, cols=cols_to_predict):
    preds = fitandpredict(m, cols)
    score = rmse(y_all[cols], preds)
    print(m.name, score)
    return score


def get_folder(oob=True):
    return "results_cv" if oob else "results"


def save_preds(name, preds, oob=True, cols=cols_to_predict):
    for col in cols:
        preds[[col]].to_csv("{}/{}_{}.csv".format(get_folder(oob), name, col))
    

def load_preds(model, oob=True):
    raw = []
    folder = get_folder(oob)
    for filename in os.listdir(folder):
        if filename.split("_NPWD")[0] == model:
            df = pd.read_csv(os.path.join(folder, filename))
            df['ut_ms'] = pd.to_datetime(df['ut_ms'])
            df = df.set_index("ut_ms")
            raw.append(df)
    
    return pd.concat(raw, axis=1)    
