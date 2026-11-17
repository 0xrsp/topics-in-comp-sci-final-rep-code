import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
SEED = 42
N_PCA = 10
N_FORECAST_TRAIN_DATES = 5

ENV_GROUPS = ["air_temp", "rh", "light", "co2", "soil_moist", "soil_temp"]


def load_data():
    t = pd.read_csv(OUT / "model_table.csv", parse_dates=["date"])
    img = pd.read_csv(OUT / "image_features.csv", parse_dates=["date"])
    img["plant"] = img["plant"].astype(str)
    t["plant"] = t["plant"].astype(str)
    d = t.merge(img, on=["year", "plant", "date"], how="inner")
    d = d.sort_values(["year", "plant", "date"]).reset_index(drop=True)
    d["cultivar_jinzhu"] = (d["cultivar"] == 2).astype(int)
    for tr in ["T1", "T2", "T3", "T4", "T5", "T6", "CK"]:
        d[f"treat_{tr}"] = (d["treatment"] == tr).astype(int)
    d["date_idx"] = d.groupby("year")["date"].rank(method="dense").astype(int)
    return d


def column_sets(d):
    time_cols = ["dat", "gdd"]
    mgmt_cols = ["cultivar_jinzhu"] + [c for c in d.columns if c.startswith("treat_")]
    env_cols = [c for c in d.columns if c.startswith("env7_")]
    hand_cols = [c for c in d.columns if c.startswith("img_")]
    resnet_cols = [c for c in d.columns if c.startswith("resnet_")]
    return time_cols, mgmt_cols, env_cols, hand_cols, resnet_cols


def logistic(t, A, k, t0):
    return A / (1.0 + np.exp(-k * (t - t0)))


def gompertz(t, A, k, t0):
    return A * np.exp(-np.exp(-k * (t - t0)))


CURVES = {"Logistic": logistic, "Gompertz": gompertz}


class GrowthCurve:

    def __init__(self, kind="Logistic", time_col="gdd"):
        self.f = CURVES[kind]
        self.kind, self.time_col = kind, time_col
        self.params = {}

    def fit(self, df):
        for cv, g in df.groupby("cultivar"):
            t, y = g[self.time_col].values, g["height"].values
            span = t.max() - t.min()
            p0 = [y.max() * 1.5, 4.0 / span, np.median(t)]
            bounds = ([y.max() * 0.8, 1e-5, t.min() - 3 * span],
                      [1000.0, 1.0, t.max() + 3 * span])
            p, _ = curve_fit(self.f, t, y, p0=p0, bounds=bounds, maxfev=20000)
            self.params[cv] = p
        return self

    def predict(self, df):
        out = np.zeros(len(df))
        for cv in df["cultivar"].unique():
            m = (df["cultivar"] == cv).values
            out[m] = self.f(df.loc[m, self.time_col].values, *self.params[cv])
        return out


def make_learner(name):
    if name == "RF":
        return RandomForestRegressor(n_estimators=500, min_samples_leaf=3,
                                     max_features=0.5, n_jobs=-1, random_state=SEED)
    return XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=4,
                        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                        n_jobs=-1, random_state=SEED)


class FeatureBuilder:

    def __init__(self, d, feature_set):
        self.time, self.mgmt, self.env, self.hand, self.resnet = column_sets(d)
        self.fs = feature_set

    def fit(self, tr):
        if self.fs in ("Img", "All"):
            self.scaler = StandardScaler().fit(tr[self.resnet].values)
            self.pca = PCA(n_components=N_PCA, random_state=SEED).fit(
                self.scaler.transform(tr[self.resnet].values))
        return self

    def transform(self, df):
        cols = self.time + self.mgmt
        if self.fs in ("Env", "All"):
            cols = cols + self.env
        X = df[cols].copy()
        if self.fs in ("Img", "All"):
            z = self.pca.transform(self.scaler.transform(df[self.resnet].values))
            for i in range(N_PCA):
                X[f"resnet_pc{i + 1}"] = z[:, i]
            for c in self.hand:
                X[c] = df[c].values
        return X


def parse_model(name):
    spec = dict(curve=None, time="gdd", cal=False, learner=None, fs=None)
    hybrid = name.startswith("Hybrid ")
    body = name[len("Hybrid "):] if hybrid else name
    if hybrid:
        c, ml = body.split("+")
        spec["cal"] = c.endswith("-cal")
        spec["curve"] = c.replace("-cal", "")
        spec["learner"], spec["fs"] = ml.split("-")
    elif body.split("-")[0] in CURVES:
        parts = body.split("-")
        spec["curve"], spec["time"] = parts[0], parts[1].lower()
        spec["cal"] = len(parts) > 2 and parts[2] == "cal"
    else:
        parts = body.split("-")
        spec["learner"], spec["fs"] = parts[0], parts[1]
        spec["cal"] = len(parts) > 2 and parts[2] == "cal"
    return spec


def plant_factors(curve, tr, te):
    c = curve.predict(tr)
    r = (pd.DataFrame({"k": tr["plant_key"].values, "h": tr["height"].values, "c": c})
         .groupby("k").sum())
    ratio = (r["h"] / r["c"]).to_dict()
    return (np.array([ratio.get(k, 1.0) for k in tr["plant_key"]]),
            np.array([ratio.get(k, 1.0) for k in te["plant_key"]]))


def fit_predict(model_name, tr, te, d):
    s = parse_model(model_name)
    if s["curve"] is not None:
        curve = GrowthCurve(s["curve"], s["time"]).fit(tr)
        r_tr, r_te = plant_factors(curve, tr, te) if s["cal"] else (1.0, 1.0)
        base_tr, base_te = r_tr * curve.predict(tr), r_te * curve.predict(te)
        if s["learner"] is None:
            return base_te, curve
        fb = FeatureBuilder(d, s["fs"]).fit(tr)
        m = make_learner(s["learner"]).fit(fb.transform(tr), tr["height"].values - base_tr)
        return base_te + m.predict(fb.transform(te)), (curve, fb, m)
    fb = FeatureBuilder(d, s["fs"]).fit(tr)
    Xtr, Xte = fb.transform(tr), fb.transform(te)
    if s["cal"]:
        ref = GrowthCurve("Logistic", "gdd").fit(tr)
        Xtr["plant_ratio"], Xte["plant_ratio"] = plant_factors(ref, tr, te)
    m = make_learner(s["learner"]).fit(Xtr, tr["height"].values)
    return m.predict(Xte), (fb, m)


MODELS = (
    ["Logistic-DAT", "Gompertz-DAT", "Logistic-GDD", "Gompertz-GDD"]
    + [f"{l}-{fs}" for l in ["RF", "XGB"] for fs in ["Env", "Img", "All"]]
    + [f"Hybrid {c}+{l}-{fs}" for c in ["Logistic", "Gompertz"]
       for l in ["RF", "XGB"] for fs in ["Env", "Img", "All"]]
)
CAL_MODELS = (["Logistic-GDD-cal", "Gompertz-GDD-cal", "RF-All-cal", "XGB-All-cal"]
              + [f"Hybrid {c}-cal+{l}-{fs}" for c in ["Logistic", "Gompertz"]
                 for l in ["RF", "XGB"] for fs in ["Env", "Img", "All"]])

def metrics(y, p):
    e = y - p
    mse = float(np.mean(e ** 2))
    return dict(MAE=float(np.mean(np.abs(e))), MSE=mse, RMSE=float(np.sqrt(mse)),
                MAPE=float(np.mean(np.abs(e) / y) * 100),
                R2=float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)))


def splits(d, protocol):
    if protocol == "LOSO":
        for y in sorted(d["year"].unique()):
            yield str(y), d.index[d["year"] != y], d.index[d["year"] == y]
    elif protocol == "Forecast":
        for y in sorted(d["year"].unique()):
            early = (d["year"] == y) & (d["date_idx"] <= N_FORECAST_TRAIN_DATES)
            late = (d["year"] == y) & (d["date_idx"] > N_FORECAST_TRAIN_DATES)
            yield str(y), d.index[(d["year"] != y) | early], d.index[late]
    elif protocol == "PlantCV":
        gkf = GroupKFold(n_splits=5)
        for i, (a, b) in enumerate(gkf.split(d, groups=d["plant_key"])):
            yield f"fold{i + 1}", d.index[a], d.index[b]


def run_protocol(d, protocol):
    models = MODELS + (CAL_MODELS if protocol == "Forecast" else [])
    preds = {m: np.full(len(d), np.nan) for m in models}
    fold_rows = []
    for fold, tr_idx, te_idx in splits(d, protocol):
        tr, te = d.loc[tr_idx], d.loc[te_idx]
        for m in models:
            p, _ = fit_predict(m, tr, te, d)
            preds[m][te_idx] = p
            fold_rows.append(dict(protocol=protocol, fold=fold, model=m,
                                  **metrics(te["height"].values, p)))
        print(f"  {protocol} {fold} done", flush=True)
    mask = ~np.isnan(preds[MODELS[0]])
    y = d["height"].values[mask]
    pooled = [dict(protocol=protocol, model=m, **metrics(y, preds[m][mask])) for m in models]
    grp = d.loc[mask].groupby(["year", "date", "box", "cultivar", "treatment"])["height"]
    pooled.append(dict(protocol=protocol, model="Reference: group mean (floor)",
                       **metrics(y, grp.transform("mean").values)))
    return pd.DataFrame(pooled), pd.DataFrame(fold_rows), preds


def grouped_permutation_importance(d, model_name="XGB-Env", n_repeats=10):
    rng = np.random.default_rng(SEED)
    rows = []
    for fold, tr_idx, te_idx in splits(d, "LOSO"):
        tr, te = d.loc[tr_idx], d.loc[te_idx]
        p, obj = fit_predict(model_name, tr, te, d)
        hybrid = model_name.startswith("Hybrid")
        if hybrid:
            curve, fb, m = obj
            base_curve = curve.predict(te)
        else:
            fb, m = obj
            base_curve = 0.0
        X = fb.transform(te)
        y = te["height"].values
        base = metrics(y, base_curve + m.predict(X))["RMSE"]
        for g in ENV_GROUPS + ["time", "management"]:
            if g == "time":
                cols = ["dat", "gdd"]
            elif g == "management":
                cols = [c for c in X.columns if c.startswith(("treat_", "cultivar"))]
            else:
                cols = [c for c in X.columns if c.startswith(f"env7_{g}_")]
            inc = []
            for _ in range(n_repeats):
                Xp = X.copy()
                perm = rng.permutation(len(Xp))
                Xp[cols] = Xp[cols].values[perm]
                inc.append(metrics(y, base_curve + m.predict(Xp))["RMSE"] - base)
            rows.append(dict(model=model_name, fold=fold, group=g,
                             rmse_increase=float(np.mean(inc))))
    return pd.DataFrame(rows)


def curve_parameters(d):
    rows = []
    for kind in CURVES:
        for tcol in ["dat", "gdd"]:
            c = GrowthCurve(kind, tcol).fit(d)
            for cv, (A, k, t0) in c.params.items():
                if kind == "Logistic":
                    t_infl, rate = t0, A * k / 4
                else:
                    t_infl, rate = t0, A * k / np.e
                rows.append(dict(curve=kind, time=tcol.upper(),
                                 cultivar="Guanghui 201" if cv == 1 else "Jinzhu",
                                 A=A, k=k, t_inflection=t_infl, max_rate=rate))
    return pd.DataFrame(rows)


def main():
    d = load_data()
    print("Modelling rows:", len(d), d.groupby("year").size().to_dict())
    d.to_csv(OUT / "analysis_table.csv", index=False)

    pooled_all, folds_all, pred_store = [], [], {}
    for protocol in ["LOSO", "Forecast", "PlantCV"]:
        pooled, folds, preds = run_protocol(d, protocol)
        pooled_all.append(pooled); folds_all.append(folds)
        pred_store[protocol] = preds
    pooled = pd.concat(pooled_all); folds = pd.concat(folds_all)
    pooled.to_csv(OUT / "results_pooled.csv", index=False)
    folds.to_csv(OUT / "results_by_fold.csv", index=False)

    for protocol in ["LOSO", "Forecast"]:
        p = d[["year", "plant", "date", "dat", "gdd", "cultivar", "treatment", "height"]].copy()
        for m, v in pred_store[protocol].items():
            p[m] = v
        p.to_csv(OUT / f"predictions_{protocol}.csv", index=False)

    imp = pd.concat([grouped_permutation_importance(d, "XGB-Env"),
                     grouped_permutation_importance(d, "Hybrid Gompertz+XGB-Env"),
                     grouped_permutation_importance(d, "Hybrid Logistic+XGB-Env")])
    imp.to_csv(OUT / "env_importance.csv", index=False)
    curve_parameters(d).to_csv(OUT / "curve_parameters.csv", index=False)

    pd.set_option("display.width", 200)
    for protocol in ["LOSO", "Forecast", "PlantCV"]:
        print(f"\n=== {protocol} ===")
        print(pooled[pooled.protocol == protocol].drop(columns="protocol")
              .round(3).to_string(index=False))
    print("\n=== LOSO per-season RMSE ===")
    print(folds[folds.protocol == "LOSO"].pivot(index="model", columns="fold", values="RMSE").round(2))
    print("\n=== Forecast per-season RMSE ===")
    print(folds[folds.protocol == "Forecast"].pivot(index="model", columns="fold", values="RMSE").round(2))
    print("\n=== Env importance ===")
    print(imp.groupby(["model", "group"])["rmse_increase"].mean().unstack(0).round(2))
    print("\n=== Curve parameters ===")
    print(curve_parameters(d).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
