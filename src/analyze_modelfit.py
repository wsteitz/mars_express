import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from common import *
import argparse
 

def error_contribution(preds, name):
    res = []
    for col in tqdm(cols_to_predict):
        res.append((col, rmse(y_all[col], preds[col])))
    res.append(("total", rmse(y_all, preds)))
    df = pd.DataFrame(res, columns=["powerline", "rmse"])
    df = df.sort_values("rmse", ascending=False)
    today = datetime.date.today().isoformat()
    df.to_csv("logs/error_by_powerline_%s_%s.csv" % (name, today), index=False)
    

def plot_model_fit(preds, name):
    df = y_all.join(preds, rsuffix="_est")
    print("plotting")
    for col in tqdm(cols_to_predict):
        df[[col, col + "_est"]].plot(figsize=(30, 10))
        plt.savefig(os.path.join("plots", "fits", name, "model_fit_%s.png" % col))
        plt.close()
    
    df = preds.copy()    
    df['power'] = y_all[cols_to_predict].sum(axis=1)
    df['estimate'] = preds[cols_to_predict].sum(axis=1)
    for col in cols_to_predict:
        del df[col]

    df.plot(figsize=(30, 10))
    plt.savefig(os.path.join("plots", "fits", name, "model_fit.png"))
    plt.close()
    
    
def analyze_model_fit(preds, name):
    print(name, rmse(y_all, preds))
    error_contribution(preds, name)
    plot_model_fit(preds, name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument('-m','--model', help='model', required=True)
    args = parser.parse_args()
    preds = load_preds(args.model)

    analyze_model_fit(preds, args.model)
