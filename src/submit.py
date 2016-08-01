from common import *
from data import *
import ensemble
import os
import shutil
import main

now = datetime.datetime.now().isoformat()
target_path = os.path.join("submissions", now)


def copy_files():
    os.mkdir(target_path)
    for filename in os.listdir("."):
        if filename.endswith(".py"):
            shutil.copyfile(filename, os.path.join(target_path, filename))


def submit_ensemble(ensemble_models, fit=True):
    copy_files()

    if fit:
        for m_name, perc in ensemble_models:
            print("fitting", m_name)
            m = main.models[m_name]
            m.fit(x_all, y_all)
            df = predict(m, x_competition, cols_to_predict)
            save_preds(m_name, df, oob=False)

    df = ensemble.weighted_average(ensemble_models, oob=False)
    # convert back to integer dates
    df['ut_ms'] = (df.index.astype(np.int64) * 1e-6).astype(int)
    # save csv
    path = os.path.join(target_path, "submission.csv")
    df[['ut_ms'] + cols_to_predict].to_csv(path, index=False)


def submit(m):
    copy_files()

    print("fitting", m.name)
    m.fit(x_all, y_all)
    df = predict(m, x_competition, cols_to_predict)
    save_preds(m.name, df, oob=False)

    # convert back to integer dates
    df['ut_ms'] = (df.index.astype(np.int64) * 1e-6).astype(int)
    # save csv
    path = os.path.join(target_path, "submission.csv")
    df[['ut_ms'] + cols_to_predict].to_csv(path, index=False)


submit_ensemble([("lstm", 0.1), ("nn", 0.12), ("onegbm", 0.3), ("my", 0.45), ("ridge", 0.03)], fit=False)
