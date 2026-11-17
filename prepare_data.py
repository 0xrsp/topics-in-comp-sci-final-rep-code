import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "2023-2025 Tomato dataset"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

PLANTING = {
    2023: pd.Timestamp("2023-05-01"),
    2024: pd.Timestamp("2024-08-22"),
    2025: pd.Timestamp("2025-03-22"),
}
T_BASE = 10.0

ENV_VARS = ["Air Temperature", "Relative Humidity", "Light Intensity",
            "CO2", "Soil Moisture", "Soil Temperature"]
ENV_SHORT = {"Air Temperature": "air_temp", "Relative Humidity": "rh",
             "Light Intensity": "light", "CO2": "co2",
             "Soil Moisture": "soil_moist", "Soil Temperature": "soil_temp"}


def parse_plant_id(pid: str, year: int):
    pid = str(pid).strip().upper()
    pid = re.sub(r"^[LR]", "", pid)
    if pid.startswith("CK"):
        digits = pid[2:]
        return "CK", int(digits[0])
    treat, cult = int(pid[0]), int(pid[1])
    if year == 2023 and treat == 7:
        return "CK", cult
    return f"T{treat}", cult


def load_measurements() -> pd.DataFrame:
    rows = []
    f23 = next((DATA / "2023/202305/raw_data/grow_index").glob("*.csv"))
    d = pd.read_csv(f23, encoding="gbk")
    d = d.rename(columns={d.columns[5]: "height", d.columns[3]: "leaf_area",
                          d.columns[14]: "date"})
    for _, r in d.iterrows():
        pid = str(int(r["Pid"]))
        treat, cult = parse_plant_id(pid, 2023)
        rows.append(dict(year=2023, plant=pid, date=pd.Timestamp(r["date"]),
                         treatment=treat, cultivar=cult, height=r["height"],
                         ndvi=r["NDVI"], lai=r["LAI"]))

    for year, sub in [(2024, "2024/2024"), (2025, "2025/2025")]:
        files = sorted((DATA / sub).glob("*/raw_data/grow*_index/*.csv"))
        for f in files:
            date = pd.Timestamp(f.stem)
            try:
                d = pd.read_csv(f, encoding="utf-8-sig")
            except UnicodeDecodeError:
                d = pd.read_csv(f, encoding="gbk")
            d.columns = ["camera", "plant", "NDVI", "RVI", "LNC", "LNA", "LAI",
                         "LDW", "height", "stem", "leaf_w", "leaf_l"]
            for _, r in d.iterrows():
                treat, cult = parse_plant_id(r["plant"], year)
                rows.append(dict(year=year, plant=str(r["plant"]).strip(), date=date,
                                 treatment=treat, cultivar=cult,
                                 height=r["height"], ndvi=r["NDVI"], lai=r["LAI"]))
    m = pd.DataFrame(rows)
    m["height"] = pd.to_numeric(m["height"], errors="coerce")
    m = m.dropna(subset=["height"])
    m = m[m["height"] > 0]
    m["plant_key"] = m["year"].astype(str) + "_" + m["plant"]
    m["dat"] = [(d - PLANTING[y]).days for d, y in zip(m["date"], m["year"])]
    return m.sort_values(["year", "plant", "date"]).reset_index(drop=True)


def load_daily_env() -> pd.DataFrame:
    files = sorted(DATA.glob("**/processed_data/sensor_data/sensor_data.xlsx"))
    frames = []
    for f in files:
        sheets = pd.read_excel(f, sheet_name=None)
        for name, s in sheets.items():
            if "Time" not in s.columns:
                continue
            s = s.copy()
            s["Time"] = pd.to_datetime(s["Time"], errors="coerce")
            s = s.dropna(subset=["Time"])
            out = pd.DataFrame({"Time": s["Time"]})
            for var in ENV_VARS:
                cols = [c for c in s.columns if str(c).startswith(var)]
                vals = s[cols].apply(pd.to_numeric, errors="coerce")
                if var != "Light Intensity":
                    vals = vals.mask(vals <= 0)
                if var == "Relative Humidity":
                    vals = vals.mask(vals > 100)
                out[ENV_SHORT[var]] = vals.mean(axis=1)
            frames.append(out)
    hh = pd.concat(frames).drop_duplicates("Time").sort_values("Time")
    hh["day"] = hh["Time"].dt.normalize()
    agg = hh.groupby("day")[list(ENV_SHORT.values())].agg(["mean", "min", "max"])
    agg.columns = [f"{v}_{s}" for v, s in agg.columns]
    agg = agg.reset_index().rename(columns={"day": "date"})
    agg["year"] = agg["date"].dt.year
    return agg


def add_time_features(m: pd.DataFrame, env: pd.DataFrame) -> pd.DataFrame:
    out = []
    for year, g in m.groupby("year"):
        e = env[env["year"] == year].set_index("date").sort_index()
        days = pd.date_range(PLANTING[year], g["date"].max(), freq="D")
        e = e.reindex(days)
        e = e.interpolate(limit_direction="both")
        tmean = (e["air_temp_max"] + e["air_temp_min"]) / 2
        gdd_daily = (tmean - T_BASE).clip(lower=0)
        gdd_cum = gdd_daily.cumsum()
        roll = e.drop(columns=["year"]).rolling(7, min_periods=1).mean()
        g = g.copy()
        g["gdd"] = gdd_cum.reindex(g["date"]).values
        for c in roll.columns:
            g[f"env7_{c}"] = roll[c].reindex(g["date"]).values
        out.append(g)
    return pd.concat(out).reset_index(drop=True)


def sensor_coverage(env: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for y, g in env.groupby("year"):
        rows.append(dict(year=y, first=g["date"].min().date(), last=g["date"].max().date(),
                         days=len(g), planting=PLANTING[y].date()))
    return pd.DataFrame(rows)


def main():
    m = load_measurements()
    env = load_daily_env()
    m.to_csv(OUT / "measurements.csv", index=False)
    env.to_csv(OUT / "daily_env.csv", index=False)
    table = add_time_features(m, env)
    table.to_csv(OUT / "model_table.csv", index=False)

    print("Measurements per season:")
    print(m.groupby("year").agg(n=("height", "size"), plants=("plant", "nunique"),
                                dates=("date", "nunique"),
                                h_min=("height", "min"), h_max=("height", "max")))
    print("\nSensor coverage:")
    print(sensor_coverage(env).to_string(index=False))
    print("\nMean height by season / cultivar / date:")
    print(m.pivot_table(index=["year", "date"], columns="cultivar",
                        values="height", aggfunc="mean").round(1))


if __name__ == "__main__":
    sys.exit(main())
