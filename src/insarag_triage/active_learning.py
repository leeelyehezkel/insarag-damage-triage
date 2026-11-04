"""Active-learning rounds with a query-by-committee (QBC) — Roboflow is only the labelling UI.

One round k:
  1. train every committee member (× seeds) on the CURRENT labelled set (original train + AL labels so far);
     evaluate each on the FROZEN val/test (comparable across rounds);
  2. score every unlabelled pool crop:
       committee = members' CORN cumulative probabilities P(y > t) for thresholds t = A|B … E|F
                   (each member averaged over its seeds);
       disagreement per threshold = BALD / JS for a binary variable:
                   H(mean_m p_m,t) − mean_m H(p_m,t)          (Houlsby 2011; ordinal → binary, Haas 2025b)
       score = Σ_t w_t · disagreement_t,  with higher weights at the B|C, C|D, D|E thresholds (C/D priority);
  3. select `budget` crops:
       quality filter (shorter side ≥ min_side px) → top `n_candidates` by score →
       focus group (committee predicts C or D, or the most-disputed threshold is B|C, C|D or D|E): `focus_share`
       of the budget; the rest from the other candidates →
       diversity: k-means on frozen embeddings within each group, crop nearest each centre,
       at most `max_per_photo` crops per original photo;
  4. upload ONLY the selected crops to the Roboflow classification project (batch "AL round k",
     tags al_round_k and pred_<grade>), the user labels them in the Roboflow UI;
  5. ingest_labels(): read the labelled crops back (export of the Roboflow project in folder format) and add
     them to TRAIN. Val/test never change.

Members are triage_lib.Config objects; frozen members retrain in seconds (cached features), partial
members fine-tune (minutes). References: Seung 1992; Settles 2012; Beluch 2018; Gontijo-Lopes 2022;
Kirsch 2019 (batch diversity); Kee 2018 (QBC + diversity); Gupte 2024 (cluster-based batch selection).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import core as L

THRESHOLDS = ["A|B", "B|C", "C|D", "D|E", "E|F"]
# C/D priority: disagreement at the boundaries that decide C and D counts double.
DEFAULT_WEIGHTS = {"A|B": 1.0, "B|C": 2.0, "C|D": 2.0, "D|E": 2.0, "E|F": 1.0}
FOCUS_THRESHOLDS = {"B|C", "C|D", "D|E"}
FOCUS_GRADES = {"C", "D"}


# Committee members
def member_cum_probs(cfg: L.Config, run_dir: Path, seed: int, paths) -> np.ndarray:
    """P(y > t) for t = 0..K-2 on `paths`, from a trained run (frozen: cached features + head.pt;
    partial/full: full model.pt)."""
    dev = L.device()
    if cfg.mode == "frozen":
        feats = L.embed(cfg, paths)                     # cached per crop
        d = feats.shape[1]                              # same head layout as TriageNet(head=cfg.head)
        if cfg.head == "mlp":
            mod = torch.nn.Sequential(torch.nn.Dropout(0.1), torch.nn.Linear(d, 256), torch.nn.GELU(),
                                      torch.nn.Dropout(0.2), torch.nn.Linear(256, L.K - 1))
        else:
            mod = torch.nn.Sequential(torch.nn.Dropout(0.1), torch.nn.Linear(d, L.K - 1))
        sd_dir = Path(run_dir) / f"seed{seed}"
        if (sd_dir / "head.pt").exists():
            sd = torch.load(sd_dir / "head.pt", map_location="cpu")
        else:                                   # early runs saved the whole model: take its head.* weights
            full = torch.load(sd_dir / "model.pt", map_location="cpu")
            sd = {k[len("head."):]: v for k, v in full.items() if k.startswith("head.")}
        mod.load_state_dict(sd)
        mod.eval()
        with torch.no_grad():
            logits = torch.cat([mod(feats[i:i + 4096]) for i in range(0, len(feats), 4096)])
    else:
        model = L.TriageNet(cfg.backbone, cfg.img_size, pretrained=False, head=cfg.head)
        model.load_state_dict(torch.load(Path(run_dir) / f"seed{seed}" / "model.pt", map_location="cpu"))
        model = model.to(dev).eval()
        tf = L.make_transforms(cfg.img_size, train=False, **model.norm)
        d = pd.DataFrame({"path": list(map(str, paths)), "y": 0})
        loader = torch.utils.data.DataLoader(L.CropDataset(d, tf), batch_size=cfg.batch_size,
                                             num_workers=cfg.num_workers)
        outs = []
        with torch.no_grad():
            for x, _ in loader:
                outs.append(model(x.to(dev)).float().cpu())
        logits = torch.cat(outs)
    return torch.cumprod(torch.sigmoid(logits), dim=1).numpy()


def train_committee(members: list[L.Config], df: pd.DataFrame, seeds, round_dir: Path, warm_start: bool = False,
                    warm_epochs: int = 10, partial_seeds=None, reset_head: bool = True,
                    cold_every: int | None = None) -> list[dict]:
    """Train every member × seed on df (its train split), skipping finished runs. Returns result dicts.

    warm_start: fine-tuned (partial/full) members continue from the PREVIOUS round's weights of the same
                seed (if present) and train at most `warm_epochs` epochs; frozen members always retrain
                from scratch (seconds). partial_seeds: fewer seeds for fine-tuned members (speed).
    reset_head: when warm-starting, keep the backbone but re-initialise the head (primacy bias, Nikishin 2022).
    cold_every: every n-th round trains from scratch anyway (limits accumulated drift; Dohare 2024)."""
    import dataclasses
    import time as _t
    round_dir = Path(round_dir)
    k = int(round_dir.name[1:]) if round_dir.name[1:].isdigit() else 0
    results = []
    for cfg in members:
        member_seeds = partial_seeds if (partial_seeds is not None and cfg.mode != "frozen") else seeds
        for s in member_seeds:
            out = round_dir / cfg.name / f"seed{s}"
            if (out / "result.json").exists():
                results.append(json.loads((out / "result.json").read_text()))
                continue
            run_cfg = cfg
            prev = round_dir.parent / f"r{k - 1}" / cfg.name / f"seed{s}" / "model.pt"
            cold_round = bool(cold_every) and k % cold_every == 0
            if warm_start and cfg.mode != "frozen" and prev.exists() and not cold_round:
                run_cfg = dataclasses.replace(cfg, epochs=warm_epochs, warmup_epochs=1,
                                              extra={**cfg.extra, "init_from": str(prev), "reset_head": reset_head})
            t0 = _t.time()
            print(f"=== {round_dir.name} · {cfg.name} seed {s}" + (" (warm start)" if run_cfg is not cfg else "") + " ===")
            results.append(L.run(run_cfg, df, s, out))
            print(f"    {(_t.time() - t0) / 60:.1f} min")
    return results


# Scoring
def _h(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def committee_scores(cum: dict[str, np.ndarray], weights=DEFAULT_WEIGHTS) -> pd.DataFrame:
    """cum: member name → (N, K-1) cumulative probabilities (already averaged over the member's seeds)."""
    P = np.stack(list(cum.values()))                     # (M, N, K-1)
    mean = P.mean(0)
    bald = _h(mean) - _h(P).mean(0)                     # per-threshold committee disagreement (N, K-1)
    w = np.array([weights[t] for t in THRESHOLDS])
    pred = (mean > 0.5).sum(1)                           # committee grade (rank-consistent)
    member_pred = {m: (p > 0.5).sum(1) for m, p in cum.items()}
    exp_grade = np.stack([p.sum(1) for p in P])         # E[grade] = Σ_t P(y > t) per member
    out = pd.DataFrame({f"bald_{t}": bald[:, i] for i, t in enumerate(THRESHOLDS)})
    out["score"] = bald @ w
    out["top_threshold"] = [THRESHOLDS[i] for i in bald.argmax(1)]
    out["pred"] = [L.CLASSES[v] for v in pred]
    out["grade_spread"] = exp_grade.std(0)              # member disagreement in grade units (reported)
    for m, v in member_pred.items():
        out[f"pred_{m}"] = [L.CLASSES[x] for x in v]
    return out


# Selection
def _diverse_pick(idx: np.ndarray, emb: np.ndarray, groups: np.ndarray, n: int, per_photo: dict,
                  max_per_photo: int, seed: int) -> list[int]:
    """k-means (k = n) on L2-normalised embeddings of idx; take the crop nearest each centre,
    respecting the per-photo cap (falls back to the next-nearest crop of the same cluster)."""
    from sklearn.cluster import KMeans
    if n <= 0 or len(idx) == 0:
        return []
    X = emb[idx] / (np.linalg.norm(emb[idx], axis=1, keepdims=True) + 1e-8)
    k = min(n, len(idx))
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
    chosen = []
    for c in range(k):
        members = np.flatnonzero(km.labels_ == c)
        order = members[np.argsort(np.linalg.norm(X[members] - km.cluster_centers_[c], axis=1))]
        for j in order:
            g = groups[idx[j]]
            if per_photo.get(g, 0) < max_per_photo:
                chosen.append(int(idx[j]))
                per_photo[g] = per_photo.get(g, 0) + 1
                break
    return chosen


def density(emb: np.ndarray, k: int = 20) -> np.ndarray:
    """Mean cosine similarity to the k nearest pool neighbours: high = typical crop, low = outlier."""
    X = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    S = X @ X.T
    np.fill_diagonal(S, -np.inf)
    k = min(k, len(X) - 1)
    return np.sort(S, axis=1)[:, -k:].mean(1)


def select_batch(pool: pd.DataFrame, scores: pd.DataFrame, emb: np.ndarray, budget: int = 100,
                 focus_share: float = 0.7, n_candidates: int = 500, max_per_photo: int = 1,
                 min_side: int = 64, seed: int = 0, ok_mask=None, density_power: float = 1.0) -> pd.DataFrame:
    """pool and scores are row-aligned; emb = frozen embeddings for the diversity step.

    ok_mask: boolean quality mask (crop_pool.quality_ok); if None, only `min_side` is applied.
    density_power: rank by disagreement × density^power (Kee 2018) so outliers are not preferred; 0 = off.
    """
    df = pd.concat([pool.reset_index(drop=True), scores.reset_index(drop=True)], axis=1)
    if ok_mask is not None:
        ok = np.asarray(ok_mask, bool)
    elif {"width", "height"} <= set(df):
        ok = (df[["width", "height"]].min(axis=1) >= min_side).to_numpy()
    else:
        ok = np.ones(len(df), bool)
    dens = density(emb)
    df["density"] = dens
    d01 = (dens - dens.min()) / (dens.max() - dens.min() + 1e-8)
    df["rank_score"] = df["score"] * (d01 ** density_power if density_power else 1.0)
    cand = df.index[ok].to_numpy()
    cand = cand[np.argsort(-df.loc[cand, "rank_score"].to_numpy())][:n_candidates]
    focus_mask = df.loc[cand, "pred"].isin(FOCUS_GRADES) | df.loc[cand, "top_threshold"].isin(FOCUS_THRESHOLDS)
    focus, other = cand[focus_mask.to_numpy()], cand[~focus_mask.to_numpy()]
    n_focus = min(len(focus), round(focus_share * budget))
    per_photo: dict = {}
    groups = df["group"].to_numpy()
    picked = _diverse_pick(focus, emb, groups, n_focus, per_photo, max_per_photo, seed)
    picked += _diverse_pick(other, emb, groups, budget - len(picked), per_photo, max_per_photo, seed)
    if len(picked) < budget:                              # top up from remaining candidates if needed
        rest = np.setdiff1d(cand, picked)
        picked += _diverse_pick(rest, emb, groups, budget - len(picked), per_photo, max_per_photo, seed)
    sel = df.loc[picked].copy()
    sel["group_in_batch"] = np.where(sel.index.isin(focus), "focus_C_D", "general")
    print(f"selected {len(sel)} (focus {int((sel.group_in_batch == 'focus_C_D').sum())}, "
          f"general {int((sel.group_in_batch == 'general').sum())}) from {len(cand)} candidates; "
          f"committee predictions: {sel['pred'].value_counts().to_dict()}")
    return sel


# Labels in / out
def upload_batch(sel: pd.DataFrame, round_no: int, api_key: str, workspace: str, project: str):
    from . import pool as C
    tags = [[f"al_round_{round_no}", f"pred_{p}"] for p in sel["pred"]]
    return C.upload_to_roboflow(sel["path"].tolist(), api_key, workspace, project,
                                batch_name=f"AL round {round_no}", tags_per_image=tags)


def ingest_labels(export_dir: Path, labelled_df: pd.DataFrame, round_no: int) -> pd.DataFrame:
    """Add newly labelled crops (Roboflow folder export) to TRAIN. Crops already present are ignored;
    crops whose photo is in val/test are refused (cannot happen for pool crops, checked anyway)."""
    new = L.collect(export_dir)
    key = ["group", "image", "ann"]
    known = labelled_df.set_index(key).index
    new = new[~new.set_index(key).index.isin(known)].copy()
    held = set(labelled_df.loc[labelled_df.split.isin(["val", "test"]), "group"].astype(str).str.upper())
    leak = new["group"].astype(str).str.upper().isin(held)
    if leak.any():
        print(f"REFUSED {int(leak.sum())} crops from val/test photos (leakage guard).")
        new = new[~leak]
    new["split"] = "train"
    new["al_round"] = round_no
    print(f"round {round_no}: +{len(new)} labelled crops → train. New label counts: "
          f"{new['label'].value_counts().reindex(L.CLASSES, fill_value=0).to_dict()}")
    out = pd.concat([labelled_df.assign(al_round=labelled_df.get('al_round', 0)), new], ignore_index=True)
    return out


def score_pool(members: list[L.Config], seeds, round_dir: Path, pool: pd.DataFrame,
               weights=DEFAULT_WEIGHTS) -> pd.DataFrame:
    """Committee scores for every pool crop; each member's probabilities averaged over its seeds."""
    cum = {}
    for cfg in members:                          # use the seeds that were actually trained for this member
        have = [s for s in seeds if (Path(round_dir) / cfg.name / f"seed{s}").exists()]
        cum[cfg.name] = np.mean([member_cum_probs(cfg, Path(round_dir) / cfg.name, s, pool["path"])
                                 for s in have], axis=0)
    return committee_scores(cum, weights)


def round_table(results: list[dict], round_no: int, n_train: int) -> pd.DataFrame:
    """Per member, on the FROZEN val and test sets (mean over seeds): overall accuracy (diag/6), plain
    accuracy, MAE (micro = all crops, macro = mean of classes), and accuracy + MAE for every class A–F."""
    rows = []
    for name in sorted({r["config"]["name"] for r in results}):
        rs = [r for r in results if r["config"]["name"] == name]
        row = dict(round=round_no, member=name, n_train=n_train, n_seeds=len(rs))
        for split in ("val", "test"):
            row[f"{split}_acc"] = float(np.mean([r[split]["overall_accuracy_diag_over_6"] for r in rs]))
            row[f"{split}_plain_acc"] = float(np.mean([r[split]["plain_accuracy_all_images"] for r in rs]))
            row[f"{split}_mae"] = float(np.mean([r[split]["mae_overall_micro"] for r in rs]))
            row[f"{split}_mae_macro"] = float(np.mean([r[split]["mae_overall_macro"] for r in rs]))
            for c in L.CLASSES:
                row[f"{split}_{c}"] = float(np.nanmean([r[split]["per_class_accuracy"][c] for r in rs]))
                row[f"{split}_mae_{c}"] = float(np.nanmean([r[split]["per_class_mae"][c] for r in rs]))
        rows.append(row)
    return pd.DataFrame(rows).round(4)


def load_round_results(al_dir: Path, round_no: int, members: list[L.Config], seeds) -> list[dict]:
    """Saved results (result.json) of one round; missing runs are skipped."""
    out = []
    for cfg in members:
        for s in seeds:
            f = Path(al_dir) / f"r{round_no}" / cfg.name / f"seed{s}" / "result.json"
            if f.exists():
                out.append(json.loads(f.read_text()))
    return out


def history(al_dir: Path, members: list[L.Config], seeds, labelled: pd.DataFrame) -> pd.DataFrame:
    """Held-out table for EVERY round trained so far, rebuilt from the saved result.json files.
    n_train of round k = original train + labels from rounds < k (what round k's models were trained on)."""
    tabs = []
    for d in sorted(Path(al_dir).glob("r*"), key=lambda p: int(p.name[1:]) if p.name[1:].isdigit() else 0):
        if not d.name[1:].isdigit():
            continue
        k = int(d.name[1:])
        res = load_round_results(al_dir, k, members, seeds)
        if res:
            n = int(((labelled.split == "train") & (labelled.get("al_round", 0) < k)).sum())
            tabs.append(round_table(res, k, n))
    return pd.concat(tabs, ignore_index=True) if tabs else pd.DataFrame()


def batch_report(sel: pd.DataFrame, labelled: pd.DataFrame, round_no: int) -> pd.DataFrame:
    """Each member's PRE-labelling prediction on this round's selected crops vs. the labels you gave them.
    `sel` = selected.csv of the round (has pred_<member> and the committee 'pred');
    `labelled` = rows of the labelled set with al_round == round_no. Excluded crops are not counted."""
    lab = labelled.assign(cid=labelled["path"].map(L.crop_id))[["cid", "y"]]
    s = sel.assign(cid=sel["path"].map(L.crop_id)).merge(lab, on="cid", how="inner")
    rows = []
    for col in ["pred"] + [c for c in s.columns if c.startswith("pred_")]:
        yp = s[col].map(L.CLASSES.index).to_numpy()
        m = L.evaluate(s["y"].to_numpy(), yp)
        row = dict(round=round_no, member="COMMITTEE" if col == "pred" else col[len("pred_"):], n=len(s),
                   batch_acc=float(np.nanmean(list(m["per_class_accuracy"].values()))),
                   batch_plain_acc=m["plain_accuracy_all_images"], batch_mae=m["mae_overall_micro"])
        row.update({f"batch_{c}": m["per_class_accuracy"][c] for c in L.CLASSES})
        row.update({f"n_{c}": m["support"][c] for c in L.CLASSES})
        rows.append(row)
    return pd.DataFrame(rows).round(4)


def progress(al_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """All rounds so far: held-out (round_table.csv) and batch (batch_report.csv) tables, stacked."""
    held = [pd.read_csv(p) for p in sorted(Path(al_dir).glob("r*/round_table.csv"))]
    batch = [pd.read_csv(p) for p in sorted(Path(al_dir).glob("r*/batch_report.csv"))]
    return (pd.concat(held, ignore_index=True) if held else pd.DataFrame(),
            pd.concat(batch, ignore_index=True) if batch else pd.DataFrame())


def plot_progress(held: pd.DataFrame, split: str = "test", path=None):
    """Left: overall accuracy per member over rounds. Right: per-class accuracy over rounds (one panel per class)."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    fig = plt.figure(figsize=(16, 7.5))
    ax0 = fig.add_subplot(2, 4, (1, 5))
    for m, d in held.groupby("member"):
        d = d.sort_values("round")
        ax0.plot(d["round"], d[f"{split}_acc"], marker="o", label=m)
    ax0.set(title=f"{split.upper()} overall accuracy (diag/6)", xlabel="round", ylabel="accuracy")
    ax0.grid(alpha=.3); ax0.legend(fontsize=7); ax0.xaxis.set_major_locator(MaxNLocator(integer=True))
    for i, c in enumerate(L.CLASSES):
        ax = fig.add_subplot(2, 4, [2, 3, 4, 6, 7, 8][i])
        for m, d in held.groupby("member"):
            d = d.sort_values("round")
            ax.plot(d["round"], d[f"{split}_{c}"], marker=".", label=m)
        ax.set(title=f"class {c}", ylim=(0, 1)); ax.grid(alpha=.3)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=120)
    return fig


# Tracking plots (x = number of training crops)
def _style(ax, title, ylabel, ylim=None):
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("training crops")
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(alpha=.3)


def plot_overall(held: pd.DataFrame, split: str = "test", path=None):
    """Overall accuracy (diag/6) and MAE per model, against the number of training crops."""
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4.8))
    for m, d in held.sort_values("n_train").groupby("member"):
        a1.plot(d["n_train"], d[f"{split}_acc"], marker="o", label=m)
        a2.plot(d["n_train"], d[f"{split}_mae"], marker="o", label=m)
        for _, r in d.iterrows():
            a1.annotate(f"R{int(r['round'])}", (r["n_train"], r[f"{split}_acc"]), fontsize=7,
                        xytext=(3, 3), textcoords="offset points", color="grey")
    _style(a1, f"{split.upper()} overall accuracy (mean per-class) — higher is better", "accuracy")
    _style(a2, f"{split.upper()} MAE in grades — lower is better", "MAE (grades)")
    a1.legend(fontsize=8)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=120)
    return fig


def plot_per_class(held: pd.DataFrame, split: str = "test", metric: str = "acc", path=None):
    """One panel per class A–F: accuracy (metric='acc') or MAE (metric='mae') per model vs training crops."""
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for ax, c in zip(axs.flat, L.CLASSES):
        col = f"{split}_{c}" if metric == "acc" else f"{split}_mae_{c}"
        for m, d in held.sort_values("n_train").groupby("member"):
            ax.plot(d["n_train"], d[col], marker=".", label=m)
        _style(ax, f"class {c}" + ("  (priority)" if c in FOCUS_GRADES else ""),
               "accuracy" if metric == "acc" else "MAE (grades)", (0, 1) if metric == "acc" else (0, None))
    axs.flat[0].legend(fontsize=7)
    fig.suptitle(f"{split.upper()} per-class {'accuracy (higher is better)' if metric == 'acc' else 'MAE (lower is better)'}",
                 fontsize=12)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=120)
    return fig


def plot_change(held: pd.DataFrame, split: str = "test", path=None):
    """Heatmap: change in accuracy per model × (overall, A–F) between the FIRST and the LATEST round."""
    import matplotlib.pyplot as plt
    first, last = held["round"].min(), held["round"].max()
    cols = [f"{split}_acc"] + [f"{split}_{c}" for c in L.CLASSES]
    a = held[held["round"] == first].set_index("member")[cols]
    b = held[held["round"] == last].set_index("member")[cols]
    d = (b - a).dropna(how="all")
    fig, ax = plt.subplots(figsize=(9, 0.6 * len(d) + 1.8))
    im = ax.imshow(d.to_numpy(), cmap="RdBu", vmin=-0.2, vmax=0.2, aspect="auto")
    ax.set_xticks(range(len(cols)), ["overall"] + L.CLASSES)
    ax.set_yticks(range(len(d)), d.index)
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            v = d.to_numpy()[i, j]
            ax.text(j, i, "" if np.isnan(v) else f"{v:+.2f}", ha="center", va="center", fontsize=9)
    ax.set_title(f"{split.upper()} accuracy change, round {first} → {last} (blue = better)", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=120)
    return fig


# Round status (which step is next?)
def status(al_dir: Path, labelled: pd.DataFrame) -> pd.DataFrame:
    """One row per round: trained? selected? uploaded? labels imported? → and the next step."""
    rows = []
    rounds = sorted(int(d.name[1:]) for d in Path(al_dir).glob("r*") if d.name[1:].isdigit())
    imported = set(labelled.get("al_round", pd.Series(dtype=int)).dropna().astype(int)) - {0}
    for k in rounds:
        d = Path(al_dir) / f"r{k}"
        rows.append(dict(round=k, trained=(d / "round_table.csv").exists() or any(d.glob("*/seed*/result.json")),
                         selected=(d / "selected.csv").exists(), uploaded=(d / "uploaded.txt").exists(),
                         labels_imported=k in imported))
    st = pd.DataFrame(rows)
    if st.empty:
        print("No rounds yet → start with ROUND = 1 (Step 2).")
        return st
    last = st.iloc[-1]
    if not last.trained:
        nxt = f"Step 2: train round {last['round']}"
    elif not last.selected:
        nxt = f"Step 3: select the batch of round {last['round']}"
    elif not last.uploaded:
        nxt = f"Step 4: upload round {last['round']}'s batch to Roboflow"
    elif not last.labels_imported:
        nxt = f"Step 5: label round {last['round']} in Roboflow, then import (Step 1 of the next round)"
    else:
        nxt = f"Step 2: train round {last['round'] + 1}"
    print("NEXT →", nxt)
    return st


def mark_uploaded(al_dir: Path, round_no: int, note: str = ""):
    (Path(al_dir) / f"r{round_no}" / "uploaded.txt").write_text(note or "uploaded")


# Images on local disk (new Colab session)
def ensure_images(labelled: pd.DataFrame, api_key: str, workspace: str, main_project: str, main_version: int,
                  label_project: str):
    """Re-download the Roboflow exports the labelled set points to, if they are not on this machine
    (e.g. after a new Colab session)."""
    import re as _re
    from roboflow import Roboflow
    roots = set()
    for p in labelled["path"]:
        m = _re.search(r"^(.*/(?:roboflow-v\d+|al-export-v\d+))/", str(p))
        if m:
            roots.add(m.group(1))
    rf = None
    for root in sorted(roots):
        if Path(root).exists():
            continue
        rf = rf or Roboflow(api_key=api_key)
        if "roboflow-v" in root:
            v = int(root.rsplit("-v", 1)[1])
            rf.workspace(workspace).project(main_project).version(v).download("folder", location=root)
        else:
            v = int(root.rsplit("-v", 1)[1])
            rf.workspace(workspace).project(label_project).version(v).download("folder", location=root)
    missing = [p for p in labelled["path"] if not Path(p).exists()]
    print(f"images on disk: {len(labelled) - len(missing)}/{len(labelled)}"
          + (f" — missing e.g. {missing[:2]}" if missing else ""))


# Upload with pre-filled classes
def upload_batch_prelabelled(sel: pd.DataFrame, round_no: int, api_key: str, workspace: str, project: str,
                             class_name=lambda g: f"damage_level_{g.lower()}", staging: Path | None = None):
    """Upload the batch as a CLASSIFICATION dataset in folder structure (train/<class>/<image>), so each image
    arrives with the committee's prediction as its class — you only correct the wrong ones in Roboflow.
    Falls back to an unlabelled upload with tags (al_round_k, pred_X) if the folder import is refused."""
    import shutil
    from roboflow import Roboflow
    root = Path(staging) / f"r{round_no}"
    if root.exists():
        shutil.rmtree(root)
    for _, r in sel.iterrows():
        d = root / "train" / class_name(r["pred"])
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy(r["path"], d / Path(r["path"]).name)
    try:
        Roboflow(api_key=api_key).workspace(workspace).upload_dataset(
            str(root), project, num_workers=4, batch_name=f"AL round {round_no}", num_retries=2)
        print(f"uploaded {len(sel)} images WITH predicted classes as batch 'AL round {round_no}'.")
        try:                                      # the folder import carries no tags → add them afterwards
            add_tags(sel, round_no, api_key, workspace, project, dry_run=False)
        except Exception as e:  # noqa: BLE001
            print(f"(tagging failed: {str(e)[:120]} — use the 'add tags' cell to retry)")
    except Exception as e:  # noqa: BLE001 — report and fall back
        print(f"pre-labelled import refused ({str(e)[:120]}) → uploading without classes, with tags pred_X")
        upload_batch(sel, round_no, api_key, workspace, project)


# Committee diversity (measured, not assumed)
def committee_diversity(al_dir: Path, round_no: int, members: list[L.Config], seeds):
    """Pairwise, on the frozen TEST set of one round (seeds pooled):
    - disagreement = share of crops where two members predict different grades (Kuncheva 2003);
    - error consistency = Cohen's kappa of 'right/wrong' between two members (Geirhos 2021):
      ~1 = they fail on the same crops (redundant), ~0 = independent errors (diverse).
    Returns (disagreement, error_consistency, accuracy) DataFrames."""
    import triage_analysis as A
    preds = {}
    for cfg in members:
        have = [s for s in seeds if (Path(al_dir) / f"r{round_no}" / cfg.name / f"seed{s}" / "test_predictions.csv").exists()]
        if have:
            p = A.load_test_predictions(Path(al_dir) / f"r{round_no}" / cfg.name, have)
            preds[cfg.name] = p.groupby("path").agg(y_true=("y_true", "first"), y_pred=("y_pred", "median"))
    names = list(preds)
    dis = pd.DataFrame(np.nan, index=names, columns=names)
    kap = pd.DataFrame(np.nan, index=names, columns=names)
    acc = {}
    for a in names:
        pa = preds[a]
        ca = (pa.y_pred.round() == pa.y_true).to_numpy()
        acc[a] = ca.mean()
        for b in names:
            pb = preds[b].reindex(pa.index)
            cb = (pb.y_pred.round() == pb.y_true).to_numpy()
            dis.loc[a, b] = float((pa.y_pred.round() != pb.y_pred.round()).mean())
            po = float((ca == cb).mean())
            pe = ca.mean() * cb.mean() + (1 - ca.mean()) * (1 - cb.mean())
            kap.loc[a, b] = (po - pe) / (1 - pe) if pe < 1 else np.nan
    return dis.round(3), kap.round(3), pd.Series(acc, name="plain_accuracy").round(3)


# Tags on images already in Roboflow
def add_tags(sel: pd.DataFrame, round_no: int, api_key: str, workspace: str, project: str, dry_run: bool = True):
    """Add tags `al_round_<k>` and `pred_<grade>` to this round's images that are already in Roboflow.

    Finds each image by its file name (matched on the crop id, so Roboflow's renaming does not matter) through
    the project search, then calls Roboflow's image-tags endpoint. dry_run=True only reports what would be
    tagged. Returns a table of (file, roboflow image id, tags, status)."""
    import requests
    from roboflow import Roboflow
    proj = Roboflow(api_key=api_key).workspace(workspace).project(project)
    want = {L.crop_id(p): (Path(p).name, [f"al_round_{round_no}", f"pred_{g}"]) for p, g in zip(sel["path"], sel["pred"])}
    found = {}
    for page in proj.search_all(offset=0, limit=100, fields=["id", "name", "tags"]):
        for im in page:
            cid = L.crop_id(im.get("name", ""))
            if cid in want:
                found[cid] = im
    rows = []
    for cid, (fname, tags) in want.items():
        im = found.get(cid)
        if im is None:
            rows.append(dict(file=fname, image_id=None, tags=" ".join(tags), status="NOT FOUND in Roboflow"))
            continue
        if dry_run:
            rows.append(dict(file=fname, image_id=im["id"], tags=" ".join(tags), status="would tag (dry run)"))
            continue
        r = requests.post(f"https://api.roboflow.com/{workspace}/{project}/images/{im['id']}/tags",
                          params={"api_key": api_key}, json={"operation": "add", "tags": tags}, timeout=30)
        rows.append(dict(file=fname, image_id=im["id"], tags=" ".join(tags),
                         status="tagged" if r.ok else f"FAILED {r.status_code}: {r.text[:80]}"))
    out = pd.DataFrame(rows)
    print(out["status"].str.split(":").str[0].value_counts().to_dict())
    return out


# Upload with verification (nothing silently lost)
def _find_in_roboflow(proj, sel: pd.DataFrame) -> dict:
    """crop id → Roboflow image record, for this batch's crops that the project search can see."""
    want = set(sel["path"].map(L.crop_id))
    found = {}
    for page in proj.search_all(offset=0, limit=100, fields=["id", "name", "tags"]):
        for im in page:
            cid = L.crop_id(im.get("name", ""))
            if cid in want:
                found[cid] = im
    return found


def _stage_and_import(rows: pd.DataFrame, round_no: int, api_key: str, workspace: str, project: str, class_name,
                      staging: Path, attempt: int):
    import shutil
    from roboflow import Roboflow
    root = Path(staging) / f"r{round_no}_try{attempt}"
    shutil.rmtree(root, ignore_errors=True)
    for _, r in rows.iterrows():
        d = root / "train" / class_name(r["pred"])
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy(r["path"], d / Path(r["path"]).name)
    Roboflow(api_key=api_key).workspace(workspace).upload_dataset(
        str(root), project, num_workers=4, batch_name=f"AL round {round_no}", num_retries=2)


def ensure_uploaded(sel: pd.DataFrame, round_no: int, api_key: str, workspace: str, project: str,
                    class_name=lambda g: f"damage_level_{g.lower()}", tag_label=None, upload_first: bool = True,
                    attempts: int = 3, wait_s: int = 30, staging: Path | None = None) -> pd.DataFrame:
    """Upload the batch WITH predicted classes, then VERIFY: find every crop in Roboflow, re-upload only the
    missing ones (up to `attempts` times), and add two tags to every image: al_round_<tag_label> and pred_<X>.

    upload_first=False → only check / repair / tag a batch that is already uploaded.
    wait_s: Roboflow needs a moment to index new images before search can see them.
    Returns one row per crop: file, image id, status (ok / MISSING)."""
    import time
    import requests
    from roboflow import Roboflow
    tag_label = round_no if tag_label is None else tag_label
    proj = Roboflow(api_key=api_key).workspace(workspace).project(project)
    if upload_first:
        _stage_and_import(sel, round_no, api_key, workspace, project, class_name, staging, 0)
        time.sleep(wait_s)
    found = _find_in_roboflow(proj, sel)
    for attempt in range(1, attempts + 1):
        missing = sel[~sel["path"].map(L.crop_id).isin(found)]
        print(f"check {attempt}: {len(sel) - len(missing)}/{len(sel)} images found in Roboflow")
        if missing.empty:
            break
        print(f"  re-uploading {len(missing)} missing crops …")
        _stage_and_import(missing, round_no, api_key, workspace, project, class_name, staging, attempt)
        time.sleep(wait_s)
        found = _find_in_roboflow(proj, sel)
    # tags: two separate tags per image; existing tags are kept
    n_tag = 0
    for cid, im in found.items():
        g = sel.loc[sel["path"].map(L.crop_id) == cid, "pred"].iloc[0]
        new = [t for t in (f"al_round_{tag_label}", f"pred_{g}") if t not in (im.get("tags") or [])]
        if new:
            r = requests.post(f"https://api.roboflow.com/{workspace}/{project}/images/{im['id']}/tags",
                              params={"api_key": api_key}, json={"operation": "add", "tags": new}, timeout=30)
            n_tag += r.ok
            if not r.ok:
                print(f"  tag failed for {im.get('name')}: {r.status_code} {r.text[:80]}")
    out = pd.DataFrame({"file": sel["path"].map(lambda p: Path(p).name),
                        "image_id": sel["path"].map(lambda p: (found.get(L.crop_id(p)) or {}).get("id")),
                        "pred": sel["pred"]})
    out["status"] = np.where(out["image_id"].notna(), "ok", "MISSING")
    print(f"RESULT: {int((out.status == 'ok').sum())}/{len(out)} in Roboflow · {n_tag} images got new tags"
          + ("" if (out.status == "ok").all() else f" · still missing: {out[out.status == 'MISSING'].file.tolist()[:5]}"))
    return out
