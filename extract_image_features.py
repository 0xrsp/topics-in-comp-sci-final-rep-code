import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "2023-2025 Tomato dataset"
OUT = ROOT / "outputs"
YEAR_DIR = {2023: DATA / "2023", 2024: DATA / "2024/2024", 2025: DATA / "2025/2025"}


MAX_GAP_DAYS = 3


def plant_box_map(year):
    mp = {}
    for f in YEAR_DIR[year].glob("*/processed_data/growth_index(final)/*.csv"):
        p = pd.read_csv(f)
        for n, ph in zip(p["Number"].astype(str), p["Photo Path"]):
            if isinstance(ph, str) and ph.strip():
                box = ph.split(";")[0].replace("\\", "/").split("/")[1]
                mp[n.strip().upper()] = re.split(r"[,，]", box)[0]
    return mp


def photo_date(year, path: Path):
    stem = path.stem
    if year == 2023:
        m, d = stem.split("_")[:2]
        return pd.Timestamp(year=2023, month=int(m), day=int(d))
    ts = int(stem.split("_")[1])
    return (pd.Timestamp(ts, unit="s") + pd.Timedelta(hours=8)).normalize()


def photo_index(year):
    rows = []
    for d in YEAR_DIR[year].glob("*/images/*"):
        box = re.split(r"[,，]", d.name)[0]
        for f in d.glob("*.jpg"):
            try:
                rows.append(dict(box=box, date=photo_date(year, f), file=f))
            except (ValueError, IndexError):
                pass
    return pd.DataFrame(rows)


def photo_lists(meas: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, g in meas.groupby("year"):
        boxes = plant_box_map(year)
        idx = photo_index(year)
        days_by_box = {b: np.array(sorted(x["date"].unique()), dtype="datetime64[D]") for b, x in idx.groupby("box")}
        for _, r in g.iterrows():
            box = boxes.get(str(r["plant"]).strip().upper())
            if box is None or box not in days_by_box:
                continue
            days = days_by_box[box]
            gap = np.abs((days - np.datetime64(r["date"].date(), "D")).astype(int))
            if gap.min() > MAX_GAP_DAYS:
                continue
            day = pd.Timestamp(days[gap.argmin()])
            files = idx[(idx["box"] == box) & (idx["date"] == day)]["file"].tolist()
            rows.append(dict(year=year, plant=r["plant"], date=r["date"], box=box,
                             photo_gap=float(gap.min()), files=files))
    return pd.DataFrame(rows)


def load_image(path, size=448):
    im = Image.open(path)
    im.draft("RGB", (size * 2, size * 2))
    return im.convert("RGB")


def canopy_features(im: Image.Image):
    a = np.asarray(im.resize((512, 384)), dtype=np.float32) / 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    s = r + g + b + 1e-6
    exg = 2 * g / s - r / s - b / s
    mask = exg > 0.10
    cover = mask.mean()
    if mask.any():
        cr, cg, cb = r[mask].mean(), g[mask].mean(), b[mask].mean()
    else:
        cr = cg = cb = 0.0
    return dict(img_green_cover=cover, img_exg_mean=float(exg.mean()),
                img_canopy_r=cr, img_canopy_g=cg, img_canopy_b=cb,
                img_brightness=float(a.mean()))


def main():
    meas = pd.read_csv(OUT / "measurements.csv", parse_dates=["date"])
    pl = photo_lists(meas)
    print(f"{len(pl)} plant-dates with photos out of {len(meas)}")
    uniq = sorted({str(f) for fs in pl["files"] for f in fs})
    print(f"{len(uniq)} unique images to process")

    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    model = torchvision.models.resnet50(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    tf = weights.transforms()
    torch.set_num_threads(max(1, torch.get_num_threads()))

    emb, hand = {}, {}
    batch, names = [], []

    def flush():
        with torch.no_grad():
            z = model(torch.stack(batch)).numpy()
        for n, v in zip(names, z):
            emb[n] = v
        batch.clear(); names.clear()

    for i, f in enumerate(uniq):
        try:
            im = load_image(f)
        except Exception as e:
            print("skip", f, e)
            continue
        hand[f] = canopy_features(im)
        batch.append(tf(im)); names.append(f)
        if len(batch) == 32:
            flush()
        if i % 200 == 0:
            print(f"  {i}/{len(uniq)}", flush=True)
    if batch:
        flush()

    rows = []
    for _, r in pl.iterrows():
        fs = [str(f) for f in r["files"] if str(f) in emb]
        if not fs:
            continue
        e = np.mean([emb[f] for f in fs], axis=0)
        h = pd.DataFrame([hand[f] for f in fs]).mean().to_dict()
        row = dict(year=r["year"], plant=r["plant"], date=r["date"], box=r["box"],
                   n_images=len(fs), **h)
        row.update({f"resnet_{k}": float(v) for k, v in enumerate(e)})
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "image_features.csv", index=False)
    print("saved", out.shape)


if __name__ == "__main__":
    main()
