from common import *
from pandas.tools.plotting import scatter_matrix



def correlations():
    df = x_all

    df['power'] = y_all[cols_to_predict].sum(axis=1)

    #######################################
    # correlations between power - feature
    #######################################
    df.corr()['power'].to_csv("correlations.csv")

    #######################################
    # scatter plots feature vs power
    #######################################
    for col in df.columns:
        df.plot.scatter(x='power', y=col)
        plt.savefig(os.path.join("plots", "power_vs_" + col + "_scatter.png"))
        plt.close()


def feature_timelines():
    #######################################
    # feature timeline
    #######################################
    df = pd.concat([x_all, x_competition]).fillna(0)
    #df = df[(df.index >= pd.Timestamp("2013-09-14")) & (df.index < pd.Timestamp("2013-09-15"))]
    df['power'] = y_all[cols_to_predict].sum(axis=1)

    for col in df.columns:
        plt.figure()
        df.power.plot(figsize=(18,4))
        df[col].plot(secondary_y=True, style='g')
        plt.savefig(os.path.join("plots", "power vs features", "power_vs_" + col + "_timeline.png"))
        plt.close()


def power_timeline():
    ###########################
    # power timelines
    ############################
    plt.figure()
    y_all.plot(figsize=(12,4))
    plt.savefig(os.path.join("plots", "power_timeline.png"))
    plt.close()
    for col in cols_to_predict:
        plt.figure()
        y_all[col].plot(figsize=(12,4))
        plt.savefig(os.path.join("plots", col + "_timeline.png"))
        plt.close()


feature_timelines()
