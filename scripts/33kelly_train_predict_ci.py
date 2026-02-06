# scripts/30_train_predict_ci.py
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List, Any

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt


PHASES_3 = ["SubG1", "G1", "G2M"]
PHASE_COLORS = {"SubG1": "#f28e2b", "G1": "#4e79a7", "G2M": "#59a14f"}


def flow_cytometry_gt_table() -> pd.DataFrame:
    rows = [
        dict(genotype="KELLY", time=2,  concentration=0.0, treatment="CTRL", SubG1=3.56,  G1=58.71, G2M=37.10),
        dict(genotype="KELLY", time=2,  concentration=0.5, treatment="AURA", SubG1=2.94,  G1=59.23, G2M=37.33),
        dict(genotype="KELLY", time=2,  concentration=1.0, treatment="AURA", SubG1=5.53,  G1=64.88, G2M=29.08),
        dict(genotype="KELLY", time=2,  concentration=2.0, treatment="AURA", SubG1=7.53,  G1=64.16, G2M=27.70),

        dict(genotype="KELLY", time=6,  concentration=0.0, treatment="CTRL", SubG1=3.06,  G1=57.67, G2M=38.70),
        dict(genotype="KELLY", time=6,  concentration=0.5, treatment="AURA", SubG1=8.48,  G1=58.12, G2M=32.36),
        dict(genotype="KELLY", time=6,  concentration=1.0, treatment="AURA", SubG1=16.05, G1=51.39, G2M=31.10),
        dict(genotype="KELLY", time=6,  concentration=2.0, treatment="AURA", SubG1=21.09, G1=53.52, G2M=24.04),

        dict(genotype="KELLY", time=24, concentration=0.0, treatment="CTRL", SubG1=7.59,  G1=62.00, G2M=29.72),
        dict(genotype="KELLY", time=24, concentration=0.5, treatment="AURA", SubG1=21.98, G1=57.08, G2M=20.51),
        dict(genotype="KELLY", time=24, concentration=1.0, treatment="AURA", SubG1=40.71, G1=50.61, G2M=8.59),
        dict(genotype="KELLY", time=24, concentration=2.0, treatment="AURA", SubG1=62.65, G1=28.89, G2M=8.06),
    ]
    gt = pd.DataFrame(rows)
    s = gt[PHASES_3].sum(axis=1).replace(0, np.nan)
    gt[PHASES_3] = gt[PHASES_3].div(s, axis=0).fillna(0.0) * 100.0
    return gt


def gt_map_from_table(gt: pd.DataFrame) -> Dict[Tuple[str, int, float, str], torch.Tensor]:
    m: Dict[Tuple[str, int, float, str], torch.Tensor] = {}
    for _, r in gt.iterrows():
        key = (str(r["genotype"]).strip().upper(), int(r["time"]), float(r["concentration"]), str(r["treatment"]).strip().upper())
        t = torch.tensor([float(r["SubG1"]), float(r["G1"]), float(r["G2M"])], dtype=torch.float32)
        t = t / (t.sum() + 1e-8)
        m[key] = t
    return m


def global_target_from_gt(gt: pd.DataFrame) -> torch.Tensor:
    v = gt[PHASES_3].mean(axis=0).astype(float).values
    t = torch.tensor(v, dtype=torch.float32)
    return t / (t.sum() + 1e-8)


@dataclass
class TrainCfg:
    seed: int = 42
    batch_size: int = 256
    epochs: int = 25
    lr: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 0  # safer on Windows
    use_mask_channel: bool = True

    w_llp: float = 1.0
    w_global: float = 0.1
    w_entropy: float = 0.03

    val_image_frac: float = 0.25

    temperature_init: float = 0.8
    temperature_grid: Tuple[float, float, int] = (0.35, 2.0, 18)

    n_bootstrap: int = 500
    alpha: float = 0.05
    min_cells_for_boot: int = 20


def collate_keep_conds(batch):
    xs = torch.stack([b[0] for b in batch], dim=0)
    conds = [b[1] for b in batch]
    return xs, conds


def normalize_conds(conds_batch: Any) -> List[Tuple[str, int, float, str]]:
    if isinstance(conds_batch, list) and len(conds_batch) > 0 and isinstance(conds_batch[0], (tuple, list)) and len(conds_batch[0]) == 4:
        out = []
        for g, t, c, tr in conds_batch:
            out.append((str(g).upper(), int(t), float(c), str(tr).upper()))
        return out
    return []


class CellCropDataset(Dataset):
    def __init__(self, work_dir: Path, manifest: pd.DataFrame, split: str, val_images: set, augment: bool, use_mask_channel: bool, seed: int):
        self.work_dir = work_dir
        self.df = manifest.copy()
        self.split = split
        self.augment_enabled = bool(augment)
        self.use_mask_channel = bool(use_mask_channel)
        self.rng = np.random.default_rng(seed + (1 if split == "val" else 0))

        is_val = self.df["source_image"].isin(val_images)
        if split == "val":
            self.df = self.df[is_val].reset_index(drop=True)
        elif split == "train":
            self.df = self.df[~is_val].reset_index(drop=True)
        else:
            self.df = self.df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def _augment(self, img: np.ndarray, msk: np.ndarray):
        if not self.augment_enabled:
            return img, msk
        if self.rng.random() < 0.5:
            img = np.fliplr(img).copy()
            msk = np.fliplr(msk).copy()
        if self.rng.random() < 0.5:
            img = np.flipud(img).copy()
            msk = np.flipud(msk).copy()
        k = int(self.rng.integers(0, 4))
        if k:
            img = np.rot90(img, k).copy()
            msk = np.rot90(msk, k).copy()
        if self.rng.random() < 0.8:
            a = float(self.rng.uniform(0.7, 1.3))
            b = float(self.rng.uniform(-25, 25))
            img = np.clip(a * img.astype(np.float32) + b, 0, 255).astype(np.uint8)
        return img, msk

    def __getitem__(self, idx: int):
        r = self.df.iloc[idx]
        crop_abs = self.work_dir / Path(r["crop_path"])
        dat = np.load(crop_abs)
        img = dat["img"].astype(np.uint8)
        msk = dat["mask"].astype(np.uint8)

        img, msk = self._augment(img, msk)

        x_img = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        if self.use_mask_channel:
            x_msk = torch.from_numpy(msk).float().unsqueeze(0)
            x = torch.cat([x_img, x_msk], dim=0)
        else:
            x = x_img

        cond = (str(r["genotype"]).upper(), int(r["time"]), float(r["concentration"]), str(r["treatment"]).upper())
        return x, cond


def make_val_split(df: pd.DataFrame, seed: int, frac: float) -> set:
    rng = np.random.default_rng(seed)
    img_df = df[["source_image", "time"]].drop_duplicates()
    imgs = sorted(img_df["source_image"].unique().tolist())
    if len(imgs) <= 1:
        return set()
    n_val = max(2, min(4, int(len(imgs) * frac)))
    val = set()
    for t in sorted(img_df["time"].unique()):
        time_imgs = img_df[img_df["time"] == t]["source_image"].unique().tolist()
        if len(time_imgs) >= 2:
            val.add(rng.choice(time_imgs, size=1, replace=False).item())
    if len(val) < n_val:
        remain = [x for x in imgs if x not in val]
        need = n_val - len(val)
        if remain and need > 0:
            add = rng.choice(remain, size=min(need, len(remain)), replace=False)
            val.update(add.tolist())
    return val


class SmallCNN(nn.Module):
    def __init__(self, in_ch: int, n_classes: int = 3):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(in_ch, 24, 3, padding=1), nn.BatchNorm2d(24), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(48, 96, 3, padding=1), nn.BatchNorm2d(96), nn.ReLU(inplace=True),
            nn.Conv2d(96, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):
        return self.head(self.feat(x).flatten(1))


def llp_condition_kl(probs: torch.Tensor, conds: List[Tuple[str, int, float, str]], gt_map) -> torch.Tensor:
    device = probs.device
    by: Dict[Tuple[str, int, float, str], List[int]] = {}
    for i, c in enumerate(conds):
        by.setdefault(c, []).append(i)
    losses = []
    for c, idxs in by.items():
        if c not in gt_map:
            continue
        pred = probs[idxs].mean(0).clamp_min(1e-8)
        target = gt_map[c].to(device).clamp_min(1e-8)
        losses.append((pred * (pred.log() - target.log())).sum())
    if not losses:
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()


def entropy_penalty(probs: torch.Tensor) -> torch.Tensor:
    p = probs.clamp_min(1e-8)
    ent = -(p * p.log()).sum(dim=1)
    return -ent.mean()


@torch.no_grad()
def eval_condition_kl(model, dl, device: str, gt_map, temperature: float) -> float:
    model.eval()
    all_probs = []
    all_conds = []
    for x, cond in dl:
        x = x.to(device)
        conds = normalize_conds(cond)
        logits = model(x) / max(float(temperature), 1e-6)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        all_probs.append(probs)
        all_conds.extend(conds)
    if not all_probs:
        return float("nan")
    P = np.vstack(all_probs)
    by: Dict[Tuple[str, int, float, str], List[int]] = {}
    for i, c in enumerate(all_conds):
        by.setdefault(c, []).append(i)
    kls = []
    for c, idxs in by.items():
        if c not in gt_map:
            continue
        pred = np.clip(P[idxs].mean(axis=0), 1e-8, 1.0)
        targ = gt_map[c].detach().cpu().numpy()
        targ = np.clip(targ / (targ.sum() + 1e-12), 1e-8, 1.0)
        kls.append(float(np.sum(pred * (np.log(pred) - np.log(targ)))))
    return float(np.mean(kls)) if kls else float("nan")


def tune_temperature(model, dl_val, device: str, gt_map, grid: Tuple[float, float, int], t_init: float) -> float:
    tmin, tmax, tn = grid
    temps = np.linspace(float(tmin), float(tmax), int(tn))
    best_t = float(t_init)
    best = float("inf")
    for t in temps:
        kl = eval_condition_kl(model, dl_val, device, gt_map, temperature=float(t))
        if np.isfinite(kl) and kl < best:
            best = float(kl)
            best_t = float(t)
    return best_t


@torch.no_grad()
def predict_all_cells(model, dl, device: str, temperature: float) -> np.ndarray:
    model.eval()
    probs_all = []
    for x, _cond in dl:
        x = x.to(device)
        logits = model(x) / max(float(temperature), 1e-6)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        probs_all.append(probs)
    return np.vstack(probs_all) if probs_all else np.zeros((0,3), dtype=float)


def bootstrap_ci(probs: np.ndarray, n_boot: int, alpha: float, seed: int):
    rng = np.random.default_rng(seed)
    n = probs.shape[0]
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = probs[idx].mean(axis=1) * 100.0
    lo = np.quantile(boot, alpha/2, axis=0)
    hi = np.quantile(boot, 1-alpha/2, axis=0)
    sd = boot.std(axis=0, ddof=1)
    mu = boot.mean(axis=0)
    return mu, sd, lo, hi


def plot_pred_vs_gt_by_time(out_png: Path, pred_cond: pd.DataFrame, gt: pd.DataFrame):
    pred2 = pred_cond.copy()
    pred2["treatment"] = pred2["treatment"].astype(str).str.upper()
    gt2 = gt.copy()
    gt2["treatment"] = gt2["treatment"].astype(str).str.upper()

    times = sorted(set(pred2["time"].unique()).intersection(set(gt2["time"].unique())))
    if not times:
        return

    def ordered_pairs(df):
        pairs = df[["treatment","concentration"]].drop_duplicates().copy()
        pairs["treatment"] = pairs["treatment"].astype(str).str.upper()
        pairs["ord"] = pairs["treatment"].map(lambda x: 0 if x=="CTRL" else 1)
        pairs = pairs.sort_values(["ord","concentration"]).reset_index(drop=True)
        return list(zip(pairs["treatment"].tolist(), pairs["concentration"].astype(float).tolist()))

    def label(trt, conc):
        if trt=="CTRL" or float(conc)==0.0:
            return "Ctrl"
        return f"{conc:g}µM Aura"

    fig, axes = plt.subplots(len(times), 2, figsize=(14, 4*len(times)), sharey=True)
    if len(times)==1:
        axes = np.array([axes])

    for r, t in enumerate(times):
        pt = pred2[pred2["time"]==t].copy()
        gt_t = gt2[gt2["time"]==t].copy()
        pairs = ordered_pairs(pt)
        x = np.arange(len(pairs))

        axp = axes[r,0]
        bottom = np.zeros(len(pairs))
        for ph in PHASES_3:
            vals=[]
            for trt, conc in pairs:
                m = (pt["treatment"]==trt) & (pt["concentration"].astype(float)==float(conc))
                vals.append(float(pt[m][ph].iloc[0]) if m.any() else 0.0)
            axp.bar(x, vals, bottom=bottom, color=PHASE_COLORS[ph], edgecolor="white", linewidth=1)
            bottom += np.asarray(vals)
        axp.set_title(f"Pred (microscopy) — {t}h")
        axp.set_xticks(x)
        axp.set_xticklabels([label(trt,conc) for trt,conc in pairs])
        axp.set_ylim(0,100)
        axp.set_ylabel("Percent")

        axg = axes[r,1]
        bottom = np.zeros(len(pairs))
        for ph in PHASES_3:
            vals=[]
            for trt, conc in pairs:
                m = (gt_t["treatment"]==trt) & (gt_t["concentration"].astype(float)==float(conc))
                vals.append(float(gt_t[m][ph].iloc[0]) if m.any() else 0.0)
            axg.bar(x, vals, bottom=bottom, color=PHASE_COLORS[ph], edgecolor="white", linewidth=1)
            bottom += np.asarray(vals)
        axg.set_title(f"GT (cytometry) — {t}h")
        axg.set_xticks(x)
        axg.set_xticklabels([label(trt,conc) for trt,conc in pairs])
        axg.set_ylim(0,100)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_dir", required=True)
    ap.add_argument("--force_cpu", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=25)
    args = ap.parse_args()

    work_dir = Path(args.work_dir)
    figs_dir = work_dir / "figures"
    models_dir = work_dir / "models"
    figs_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    cfg = TrainCfg(epochs=int(args.epochs))
    (work_dir / "train_config.json").write_text(json.dumps(cfg.__dict__, indent=2, default=list), encoding="utf-8")

    gt = flow_cytometry_gt_table()
    gt.to_csv(work_dir / "cytometry_gt.csv", index=False)
    gt_map = gt_map_from_table(gt)
    global_t = global_target_from_gt(gt)

    manifest_path = work_dir / "manifest_cells.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing {manifest_path}. Run 20_build_crops_from_masks.py first.")
    manifest = pd.read_csv(manifest_path)

    # split
    val_imgs = make_val_split(manifest, seed=cfg.seed, frac=cfg.val_image_frac)

    ds_tr = CellCropDataset(work_dir, manifest, "train", val_imgs, augment=True, use_mask_channel=cfg.use_mask_channel, seed=cfg.seed)
    ds_va = CellCropDataset(work_dir, manifest, "val",   val_imgs, augment=False, use_mask_channel=cfg.use_mask_channel, seed=cfg.seed)

    # keep only GT conditions
    def in_gt(df):
        keys = list(zip(
            df["genotype"].astype(str).str.upper(),
            df["time"].astype(int),
            df["concentration"].astype(float),
            df["treatment"].astype(str).str.upper(),
        ))
        return pd.Series([k in gt_map for k in keys], index=df.index)

    ds_tr.df = ds_tr.df[in_gt(ds_tr.df)].reset_index(drop=True)
    ds_va.df = ds_va.df[in_gt(ds_va.df)].reset_index(drop=True)

    if len(ds_tr) == 0:
        raise RuntimeError("No training cells after GT filter.")

    bs_tr = int(min(cfg.batch_size, max(1, len(ds_tr))))
    bs_va = int(min(cfg.batch_size, max(1, len(ds_va))))

    dl_tr = DataLoader(ds_tr, batch_size=bs_tr, shuffle=True, num_workers=cfg.num_workers, pin_memory=False, drop_last=False, collate_fn=collate_keep_conds)
    dl_va = DataLoader(ds_va, batch_size=bs_va, shuffle=False, num_workers=cfg.num_workers, pin_memory=False, drop_last=False, collate_fn=collate_keep_conds)

    # device
    device = "cpu" if int(args.force_cpu) == 1 else ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    in_ch = 2 if cfg.use_mask_channel else 1
    model = SmallCNN(in_ch=in_ch, n_classes=3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)

    best_path = models_dir / "cnn_llp_best_state_dict.pt"
    best_val = float("inf")
    patience = 0

    for ep in range(cfg.epochs):
        model.train()
        losses = []
        for x, cond in dl_tr:
            x = x.to(device)
            conds = normalize_conds(cond)
            logits = model(x) / max(float(cfg.temperature_init), 1e-6)
            probs = torch.softmax(logits, dim=1)

            l_llp = llp_condition_kl(probs, conds, gt_map)
            batch_mean = probs.mean(0).clamp_min(1e-8)
            g = global_t.to(device).clamp_min(1e-8)
            l_global = (batch_mean * (batch_mean.log() - g.log())).sum()
            l_ent = entropy_penalty(probs)

            loss = cfg.w_llp*l_llp + cfg.w_global*l_global + cfg.w_entropy*l_ent
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))

        val = eval_condition_kl(model, dl_va, device, gt_map, temperature=float(cfg.temperature_init))
        scheduler.step(val if np.isfinite(val) else 0.0)

        if val < best_val:
            best_val = val
            torch.save(model.state_dict(), best_path)
            patience = 0
        else:
            patience += 1
            if patience >= 7:
                break

        print(f"Epoch {ep+1:02d}: train_loss={np.mean(losses):.4f} val_KL={val:.4f}")

    # load best
    state = torch.load(best_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # temperature scaling on val
    best_T = tune_temperature(model, dl_va, device, gt_map, cfg.temperature_grid, cfg.temperature_init)
    (models_dir / "temperature_scaling.json").write_text(json.dumps({"best_T": best_T}, indent=2), encoding="utf-8")

    # predict all
    ds_all = CellCropDataset(work_dir, manifest, "all", set(), augment=False, use_mask_channel=cfg.use_mask_channel, seed=cfg.seed)
    dl_all = DataLoader(ds_all, batch_size=int(min(cfg.batch_size, max(1, len(ds_all)))), shuffle=False, num_workers=0, pin_memory=False, collate_fn=collate_keep_conds)
    probs = predict_all_cells(model, dl_all, device, temperature=best_T)

    pred = probs.argmax(axis=1)
    pred_cells = manifest.copy()
    pred_cells["p_SubG1"] = probs[:,0]
    pred_cells["p_G1"] = probs[:,1]
    pred_cells["p_G2M"] = probs[:,2]
    pred_cells["phase_pred"] = [PHASES_3[int(i)] for i in pred]
    pred_cells.to_csv(work_dir / "cell_predictions.csv", index=False)

    # aggregate + bootstrap
    group_cols = ["genotype","time","concentration","treatment"]
    out_rows=[]
    for key, g in pred_cells.groupby(group_cols):
        P = g[["p_SubG1","p_G1","p_G2M"]].to_numpy(float)
        mean_pct = P.mean(axis=0)*100.0
        rec = {c: (key[i] if isinstance(key, tuple) else key) for i,c in enumerate(group_cols)}
        rec.update({"SubG1": float(mean_pct[0]), "G1": float(mean_pct[1]), "G2M": float(mean_pct[2]), "n_cells": int(len(g))})
        if len(g) >= cfg.min_cells_for_boot:
            mu, sd, lo, hi = bootstrap_ci(P, cfg.n_bootstrap, cfg.alpha, seed=cfg.seed + int(abs(hash(key))%100000))
            for i, ph in enumerate(PHASES_3):
                rec[f"{ph}_std"] = float(sd[i])
                rec[f"{ph}_ci_low"] = float(lo[i])
                rec[f"{ph}_ci_high"] = float(hi[i])
        else:
            for ph in PHASES_3:
                rec[f"{ph}_std"] = float("nan")
                rec[f"{ph}_ci_low"] = float("nan")
                rec[f"{ph}_ci_high"] = float("nan")
        out_rows.append(rec)

    pred_cond = pd.DataFrame(out_rows).sort_values(["time","treatment","concentration"])
    pred_cond.to_csv(work_dir / "predicted_phase_percentages.csv", index=False)

    # plot
    plot_pred_vs_gt_by_time(figs_dir / "pred_vs_gt_by_time.png", pred_cond, gt)
    print(f"Done. Work dir: {work_dir}")


if __name__ == "__main__":
    main()
