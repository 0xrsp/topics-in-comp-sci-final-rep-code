from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from run_experiments import GrowthCurve, load_data

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True, parents=True)

plt.rcParams.update({"font.family": "Arial", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 200,
                     "savefig.bbox": "tight"})
SEASON_COL = {2023: "#1f5f8b", 2024: "#c0392b", 2025: "#2e7d32"}
CULT = {1: "Guanghui 201", 2: "Jinzhu"}
FAMILY_COL = {"Growth curve": "#7f8c8d", "ML": "#1f5f8b", "Hybrid": "#c0392b"}


def family(m):
    if m.startswith("Hybrid"):
        return "Hybrid"
    if m.split("-")[0] in ("Logistic", "Gompertz"):
        return "Growth curve"
    return "ML"


def fig_growth_curves(d):
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.2), sharey=True)
    for j, tcol in enumerate(["dat", "gdd"]):
        for i, cv in enumerate([1, 2]):
            ax = axes[i, j]
            g = d[d.cultivar == cv]
            for y, gy in g.groupby("year"):
                s = gy.groupby("date").agg(t=(tcol, "mean"), h=("height", "mean"),
                                           sd=("height", "std"))
                ax.errorbar(s.t, s.h, yerr=s.sd, fmt="o", ms=4, capsize=2,
                            color=SEASON_COL[y], label=str(y), alpha=0.9)
            tt = np.linspace(0, g[tcol].max() * 1.05, 200)
            for kind, ls in [("Logistic", "-"), ("Gompertz", "--")]:
                c = GrowthCurve(kind, tcol).fit(g)
                ax.plot(tt, c.f(tt, *c.params[cv]), ls, color="black", lw=1.3,
                        label=f"{kind} (all seasons)")
            ax.set_title(f"{CULT[cv]}", fontsize=10)
            ax.set_xlabel("Days after transplanting" if tcol == "dat" else "Growing degree days (°C·d, base 10 °C)")
            if j == 0:
                ax.set_ylabel("Plant height (cm)")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "fig1_growth_curves.png")
    plt.close(fig)


def fig_model_comparison(pooled):
    show = ["Logistic-DAT", "Gompertz-DAT", "Logistic-GDD", "Gompertz-GDD",
            "RF-Env", "XGB-Env", "XGB-Img", "XGB-All",
            "Hybrid Logistic+XGB-Env", "Hybrid Logistic+XGB-Img", "Hybrid Logistic+XGB-All", "Hybrid Gompertz+XGB-Env",
            "Hybrid Gompertz+XGB-Img", "Hybrid Gompertz+XGB-All"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.9), sharey=True)
    for ax, prot, title in zip(axes, ["LOSO", "Forecast"],
                               ["Leave-one-season-out", "Mid-season forecast"]):
        p = pooled[pooled.protocol == prot].set_index("model").loc[show]
        cols = [FAMILY_COL[family(m)] for m in show]
        ax.barh(range(len(show)), p["RMSE"], color=cols)
        for k, v in enumerate(p["RMSE"]):
            ax.text(v + 0.5, k, f"{v:.1f}", va="center", fontsize=8)
        ax.set_yticks(range(len(show)))
        ax.set_yticklabels([m.replace("Hybrid ", "H: ") for m in show], fontsize=8.5)
        ax.invert_yaxis()
        floor = pooled[(pooled.protocol == prot) & pooled.model.str.startswith("Reference")]["RMSE"].iloc[0]
        ax.axvline(floor, color="black", ls=":", lw=1)
        ax.text(floor - 0.5, len(show) - 0.3, f"noise floor {floor:.1f}", fontsize=7.5, ha="right", va="bottom")
        ax.set_xlabel("RMSE (cm)")
        ax.set_title(title, fontsize=10, pad=14)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FAMILY_COL.values()]
    fig.legend(handles, FAMILY_COL.keys(), frameon=False, fontsize=9, ncol=3,
               loc="upper center", bbox_to_anchor=(0.6, 1.04))
    fig.tight_layout()
    fig.savefig(FIG / "fig2_model_comparison.png")
    plt.close(fig)


def fig_pred_vs_obs(best_curve, best_ml, best_hybrid):
    p = pd.read_csv(OUT / "predictions_LOSO.csv")
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), sharex=True, sharey=True)
    lim = [0, max(p.height.max(), 320)]
    for ax, m in zip(axes, [best_curve, best_ml, best_hybrid]):
        for y, g in p.groupby("year"):
            ax.scatter(g.height, g[m], s=6, alpha=0.5, color=SEASON_COL[y], label=f"{y} held out")
        ax.plot(lim, lim, color="black", lw=0.8)
        e = p.height - p[m]
        ax.set_title(f"{m}\nRMSE {np.sqrt((e**2).mean()):.1f} cm", fontsize=9)
        ax.set_xlabel("Observed height (cm)")
    axes[0].set_ylabel("Predicted height (cm)")
    axes[0].legend(frameon=False, fontsize=7, markerscale=2)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_pred_vs_obs_loso.png")
    plt.close(fig)


def fig_importance():
    imp = pd.read_csv(OUT / "env_importance.csv")
    names = {"air_temp": "Air temperature", "rh": "Relative humidity", "light": "Light intensity",
             "co2": "CO2", "soil_moist": "Soil moisture", "soil_temp": "Soil temperature",
             "time": "Time (DAT, GDD)", "management": "Cultivar & treatment"}
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for ax, m, title in zip(axes, ["XGB-Env", "Hybrid Gompertz+XGB-Env"],
                            ["XGB-Env (direct)", "Hybrid Gompertz+XGB-Env (residual model)"]):
        s = imp[imp.model == m].groupby("group")["rmse_increase"].agg(["mean", "std"])
        s = s.sort_values("mean")
        cols = ["#7f8c8d" if g in ("time", "management") else "#1f5f8b" for g in s.index]
        ax.barh([names[g] for g in s.index], s["mean"], xerr=s["std"], color=cols, capsize=2)
        ax.set_xlabel("Increase in LOSO RMSE when permuted (cm)")
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_env_importance.png")
    plt.close(fig)


def fig_example_images():
    from extract_image_features import photo_index, YEAR_DIR
    idx = photo_index(2024)
    box = idx[idx.box == "111"]
    picks = [pd.Timestamp(x) for x in ["2024-09-21", "2024-10-19", "2024-11-13"]]
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.2))
    for j, day in enumerate(picks):
        f = sorted(box[box.date == day]["file"].tolist())[1]
        im = Image.open(f); im.draft("RGB", (800, 600)); im = im.convert("RGB").resize((640, 480))
        a = np.asarray(im, dtype=np.float32) / 255
        s = a.sum(-1) + 1e-6
        exg = (2 * a[..., 1] - a[..., 0] - a[..., 2]) / s
        axes[0, j].imshow(im); axes[0, j].set_title(day.strftime("%d %b %Y"), fontsize=9)
        axes[1, j].imshow(exg > 0.10, cmap="Greens")
        axes[1, j].set_title(f"Green cover {np.mean(exg > 0.10) * 100:.1f}%", fontsize=9)
        for ax in axes[:, j]:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG / "fig5_example_images.png")
    plt.close(fig)


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.axis("off")
    def box(x, y, w, h, text, c):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=c, edgecolor="black", lw=1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.5, wrap=True)
    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.3))
    box(0.0, 0.70, 0.20, 0.22, "Sensors (30 min)\n→ daily mean/min/max\n→ 7-day window + GDD", "#d6e4f0")
    box(0.0, 0.38, 0.20, 0.22, "Canopy photos\n→ ResNet-50 (PCA 10)\n+ green cover / colour", "#d6e4f0")
    box(0.0, 0.06, 0.20, 0.22, "Management\ncultivar, fertiliser\ntreatment", "#d6e4f0")
    box(0.33, 0.66, 0.24, 0.26, "Growth curve\nLogistic / Gompertz\nh(GDD) per cultivar", "#e8e8e8")
    box(0.33, 0.12, 0.24, 0.30, "RF / XGBoost\nlearns residual\nobserved − curve", "#f5d5d0")
    box(0.72, 0.38, 0.26, 0.26, "Hybrid prediction\nĥ = curve + residual", "#fbe9b7")
    arrow(0.20, 0.81, 0.33, 0.79)
    arrow(0.20, 0.78, 0.33, 0.33)
    arrow(0.20, 0.49, 0.33, 0.28)
    arrow(0.20, 0.17, 0.33, 0.22)
    arrow(0.57, 0.79, 0.72, 0.55)
    arrow(0.57, 0.27, 0.72, 0.46)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.savefig(FIG / "fig0_pipeline.png")
    plt.close(fig)


def main():
    d = load_data()
    pooled = pd.read_csv(OUT / "results_pooled.csv")
    fig_growth_curves(d)
    fig_model_comparison(pooled)
    lo = pooled[(pooled.protocol == "LOSO") & ~pooled.model.str.startswith("Reference")].copy()
    lo["family"] = lo.model.map(family)
    best = lo.sort_values("RMSE").groupby("family").first()["model"]
    fig_pred_vs_obs(best["Growth curve"], best["ML"], best["Hybrid"])
    fig_importance()
    fig_example_images()
    fig_pipeline()
    print("best per family (LOSO):", best.to_dict())


if __name__ == "__main__":
    main()
