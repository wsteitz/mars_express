import pandas as pd
import numpy as np
import datetime
import os
import feather


DATA_DIR = "../data"


def read(path, content):
    raw = []
    for filename in os.listdir(os.path.join(DATA_DIR, path)):
        if content in filename:
            compression = "gzip" if filename.endswith(".gz") else ""
            df = pd.read_csv(os.path.join(DATA_DIR, path, filename), compression=compression)
            raw.append(df)
    df = pd.concat(raw)
    if 'ut_ms' in df.columns:
        df['ut_ms'] = pd.to_datetime(df['ut_ms'], unit='ms')
        df.sort_values("ut_ms", inplace=True)
        df = df.set_index('ut_ms')
    return df


def parse_power(path):
    df = read(path, "power")
    df = df.resample("1h").mean()
    return df


def parse_context_ltdata(path):
    df = read(path, "ltdata")
    return df


def parse_context_saaf(path):
    df = read(path, "saaf")
    # interpolate to fill the gaps
    hourly = df.resample("1h").mean().interpolate()
    daily = df.resample("1D").mean().reindex(hourly.index, method='nearest')
    return hourly


def parse_context_dmop(path):
    df = read(path, "dmop")

    # ATTT-A and ATTT-B are different
    attt = df[df['subsystem'].str.startswith("ATTT")]
    attt['subsystem'] = attt['subsystem'].str[:3] + attt['subsystem'].str[-1]
    
    df = pd.concat([attt, df])

    # take the first 4 chars
    df['subsystem'] = df['subsystem'].str[:4]

    # convert to 1 / 0
    df = pd.get_dummies(df.subsystem)
    df = df.resample("1h").sum().fillna(0.0)
    
    df['sum_dmop'] = df.sum(axis=1)

    return df
    

def parse_context_evtf(path):
    df = read(path, "evtf")
    df['event'] = df.description.str.split("/", 1).apply(lambda x: x[0])
    del df['description']
    # all PASSAGE events are numbered - take out the number
    df['event'] = df.event.str.split("AGE_", 1).apply(lambda x: x[0])
    
    # FIXME what happens when we change the 100
    dummies = df.groupby("event").filter(lambda x: len(x) > 100)
    dummies = pd.get_dummies(dummies.event)
    dummies = dummies.resample("1h").sum().fillna(0.0)
    
    return dummies


def event_to_min_per_hour(df, event):
    def hourly(start, end):
        ret = [(start.floor("1h"), 60 - start.minute)]
        t = start.ceil("1h")
        while t <= end:
            ret.append((t, 60))
            t += pd.Timedelta("1h")
        ret.append((end.floor("1h"), end.minute - 60))
        return ret

    df = df[df.event.str.contains(event)]

    res = []
    for i, (start, end, _) in df.iterrows():
        res += hourly(start, end)

    df = pd.DataFrame(res)
    df.columns = ['ut_ms', event + "_mins"]

    df = df.set_index('ut_ms')
    df = df.resample("1h").sum().fillna(0.0)
    return df


def parse_context_ftl(path):
    raw = read(path, "ftl")

    df = raw.copy()
    df['ut_ms'] = pd.to_datetime(raw['utb_ms'], unit='ms')
    df.sort_values("ut_ms", inplace=True)
    # dummies
    df = df.set_index('ut_ms')
    dummies = pd.get_dummies(df.type).join(df['flagcomms'], how="outer")
    dummies = dummies.resample("1h").sum().fillna(0.0)

    df = raw.copy()
    df['event'] = df.type + df.flagcomms.astype("str")
    del df['type'], df['flagcomms']
    df['ute_ms'] = pd.to_datetime(df['ute_ms'], unit='ms')
    df['utb_ms'] = pd.to_datetime(df['utb_ms'], unit='ms')
    durations = [event_to_min_per_hour(df, event) for event in df.event.unique()]
    durations = pd.concat(durations, axis=1).fillna(0)

    return dummies.join(durations, how="outer")


def parse(path):
    power = parse_power(path)
    ltdata = parse_context_ltdata(path)
    saaf = parse_context_saaf(path)
    dmop = parse_context_dmop(path)
    evtf = parse_context_evtf(path)
    ftl = parse_context_ftl(path)
    ltdata = ltdata.reindex(power.index, method='ffill')
    saaf = saaf.reindex(power.index, method='nearest')
    # if the first event happens after the first entry, it will be NA
    dmop = dmop.reindex(power.index, method='ffill').fillna(method='bfill').fillna(0)
    evtf = evtf.reindex(power.index, method='ffill').fillna(method='bfill').fillna(0)
    ftl = ftl.reindex(power.index, method='ffill').fillna(method='bfill').fillna(0)
    df = power.join(ltdata)
    df = df.join(saaf)
    df = df.join(dmop)
    df = df.join(evtf)
    df = df.join(ftl)
    return df


def maybe_parse(path):
    feather_file = path + ".feather"
    if os.path.exists(feather_file):
        print("loading %s from cache" % path)
        df = feather.read_dataframe(feather_file)
        df = df.set_index("ut_ms")
        return df
    else:
        print("parsing %s" % path)
        df = parse(path)
        feather.write_dataframe(df.reset_index(), feather_file)
        return df


def compute_features(df):
    # well just guessing
    first_mars_day = datetime.date(2003, 1, 1)
    df['date_diffs'] = df.index.date - first_mars_day
    df['days_in_orbit'] = df['date_diffs'] / np.timedelta64(1, 'D')
    df['months_in_orbit'] = (df['date_diffs'] / np.timedelta64(1, 'M')).astype(int)
    del df['date_diffs']

    # shifted values
    df['sz-1'] = df.sz.shift(1).fillna(df["sz"])
    df['sz-2'] = df.sz.shift(2).fillna(df["sz-1"])
    df['sz-3'] = df.sz.shift(3).fillna(df["sz-2"])
    df['sz-4'] = df.sz.shift(4).fillna(df["sz-3"])
    df['sx-1'] = df.sx.shift(1).fillna(df["sx"])
    df['sx-2'] = df.sx.shift(2).fillna(df["sx-1"])
    df['sx-3'] = df.sx.shift(3).fillna(df["sx-2"])
    df['sx-4'] = df.sx.shift(4).fillna(df["sx-3"])
    df['sy-1'] = df.sy.shift(1).fillna(df["sy"])
    df['sy-2'] = df.sy.shift(2).fillna(df["sy-1"])
    df['sy-3'] = df.sy.shift(3).fillna(df["sy-2"])
    df['sy-4'] = df.sy.shift(4).fillna(df["sy-3"])
    df['sa-1'] = df.sa.shift(1).fillna(df["sa"])
    df['sa-2'] = df.sa.shift(2).fillna(df["sa-1"])
    df['sa-3'] = df.sa.shift(3).fillna(df["sa-2"])
    df['sa-4'] = df.sa.shift(4).fillna(df["sa-3"])

    df['ASXX-1'] = df['ASXX'].shift(1).fillna(df['ASXX'])

    df['sa_monthly'] = df.sa.resample("1M").mean().reindex(df.index, method='bfill').fillna(0.0)
    df['sx_monthly'] = df.sx.resample("1M").mean().reindex(df.index, method='bfill').fillna(0.0)
    df['sy_monthly'] = df.sy.resample("1M").mean().reindex(df.index, method='bfill').fillna(0.0)
    df['sz_monthly'] = df.sz.resample("1M").mean().reindex(df.index, method='bfill').fillna(0.0)

    df['occultationduration_min_monthly'] = df.occultationduration_min.resample("1M").mean().reindex(df.index, method='bfill').fillna(0.0)

    # doesnt' help much
    df['AOS'] = df[[c for c in df.columns if "_AOS_" in c]].sum(axis=1)
    df['LOS'] = df[[c for c in df.columns if "_LOS_" in c]].sum(axis=1)

    df["eclipseduration_min_sqrt"] = np.sqrt(df["eclipseduration_min"])
    del df['eclipseduration_min']
    df["sunmarsearthangle_deg_log"] = np.log(df["sunmarsearthangle_deg"])
    df["sunmarsearthangle_deg_sqrt"] = np.sqrt(df["sunmarsearthangle_deg"])
    del df['sunmarsearthangle_deg']
    df["EARTHTrue_mins_sqrt"] = np.sqrt(df["EARTHTrue_mins"])
    del df['EARTHTrue_mins']
    
    # they cómpletely changed the way they operate the aircraft. add a flag that indicates this
    df['new_operation_mode'] = df.index >= "2012-10-01"

    # counts for every ascend / descend
    df['ACSEND_OR_DESCEND'] = df[[col for col in df.columns if ("ACSEND" in col) or ("DESCEND" in col)]].sum(axis=1)

    return df


def remove_features(df):
    # FIXME might be helpful
    del df["ATTP"]
    del df["ATTR"]
    del df["SCMN"]
    
    return df


x_competition = maybe_parse("test_set")
x_competition = compute_features(x_competition)
cols_to_predict = [c for c in x_competition.columns if "NPWD" in c]
for col in cols_to_predict:
    del x_competition[col]


# load data
x_all = maybe_parse("train_set").dropna()
x_all = compute_features(x_all)
x_all = remove_features(x_all)
y_all = x_all[cols_to_predict]
for col in cols_to_predict:
    del x_all[col]

# need to filter out cols that we have in training but not in the competition data - and vice versa
for col in x_all.columns:
    if col not in x_competition.columns:
        del x_all[col]
for col in x_competition.columns:
    if col not in x_all.columns:
        del x_competition[col]

# sort by column name to keep results stable
x_all = x_all.reindex_axis(sorted(x_all.columns), axis=1)
x_competition = x_competition.reindex_axis(sorted(x_competition.columns), axis=1)


print("got data!")
