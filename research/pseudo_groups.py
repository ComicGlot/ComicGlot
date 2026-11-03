"""Pseudo-titles: group pages that probably come from the same series.

The Roboflow exports keep no title or chapter metadata (autogen pages are
renamed img_000123, early pages are 01_png ... 99_png), so held-out-title
evaluation needs a proxy. This script clusters whole-page embeddings from a
frozen encoder (style, palette, characters) and, where filenames still carry a
chapter id (<chapter>_<page>_png), measures how well clusters keep known
chapters together.

A cluster is a conservative stand-in for a title only when chapters rarely
split across clusters; the report prints that rate so the choice of k can be
defended in the paper. Over-merging (several titles in one cluster) is safe:
it only makes held-out evaluation stricter.

Usage:
    python -m research.pseudo_groups embed  --dataset DIR --work out/groups --encoder onnx:dinov2s_q.onnx [--files files.csv]
    python -m research.pseudo_groups cluster --work out/groups --k 20 40 80 --dup 0.8 --era

Group files written to --work, all in the (split, file, group) format that
carrier_probe analyse --groups reads:
  groups_k{k}.csv     page-style clusters (Ward on PCA of page embeddings)
  groups_dup{t}.csv   near-duplicate components: pages joined when cosine >= t.
                      Below about 0.8 one component swallows most pages and
                      GroupKFold then trains on a small remainder; the report
                      prints the largest component so this is visible.
  groups_era.csv      two groups, early bare-numbered pages vs later autogen /
                      named pages; use with --folds 2 for the cross-era test
"""
from __future__ import annotations

import argparse
import collections
import csv
import time
from pathlib import Path

import numpy as np

from research.common import available_splits, filename_group, iter_split, save_json, unique_sources
from research.carrier_probe import MEAN, STD, make_encoder


def page_tensor(path: Path, size: int = 224) -> np.ndarray:
    from PIL import Image
    im = Image.open(path).convert("RGB")
    im.draft("RGB", (2 * size, 2 * size))
    w, h = im.size
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(im, ((side - w) // 2, (side - h) // 2))
    a = np.asarray(canvas.resize((size, size), Image.BILINEAR), np.float32) / 255.0
    return ((a - MEAN) / STD).transpose(2, 0, 1)


def list_pages(dataset: Path):
    """One representative file per unique source (augmentation copies merged)."""
    pages = []
    for split in available_splits(dataset):
        groups, empty = unique_sources(iter_split(dataset, split))
        for copies in groups.values():
            lf = copies[0]
            if lf.image_path:
                pages.append((split, lf.image_path.name, [c.image_path.name for c in copies if c.image_path]))
        for lf in empty:
            if lf.image_path:
                pages.append((split, lf.image_path.name, [lf.image_path.name]))
    return pages


def cmd_embed(args) -> None:
    args.work.mkdir(parents=True, exist_ok=True)
    plist = args.work / "pages.csv"
    if not plist.exists():
        pages = list_pages(args.dataset)
        with open(plist, "w", newline="", encoding="utf-8") as fh:
            wr = csv.writer(fh)
            wr.writerow(["split", "file", "copies"])
            for split, f, copies in pages:
                wr.writerow([split, f, "|".join(copies)])
    rows = list(csv.DictReader(open(plist, encoding="utf-8")))
    out = args.work / "page_emb.npy"
    feats = np.load(out) if out.exists() else None
    done = 0 if feats is None else len(feats)
    enc = make_encoder(args.encoder)
    t0, chunks = time.time(), []
    while done < len(rows) and (args.budget <= 0 or time.time() - t0 < args.budget):
        part = rows[done:done + args.batch]
        chunks.append(enc(np.stack([page_tensor(args.dataset / r["split"] / "images" / r["file"]) for r in part])).astype(np.float32))
        done += len(part)
    if chunks:
        new = np.concatenate(chunks)
        feats = new if feats is None else np.concatenate([feats, new])
        np.save(out, feats)
    print(f"{0 if feats is None else len(feats)} of {len(rows)} pages embedded")


def chapter_split_rate(labels, chapters) -> dict:
    """Share of known chapters (>= 2 pages) whose pages land in more than one cluster."""
    by = collections.defaultdict(set)
    for lab, ch in zip(labels, chapters):
        if ch:
            by[ch].add(lab)
    multi = [c for c in by if sum(1 for ch in chapters if ch == c) >= 2]
    split = [c for c in multi if len(by[c]) > 1]
    return {"chapters_with_2plus_pages": len(multi), "split_across_clusters": len(split),
            "split_rate": len(split) / len(multi) if multi else None}


def cmd_cluster(args) -> None:
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import normalize
    rows = list(csv.DictReader(open(args.work / "pages.csv", encoding="utf-8")))
    X = np.load(args.work / "page_emb.npy")
    rows = rows[: len(X)]
    chapters = []
    for r in rows:
        kind, chapter, _ = filename_group(r["file"].split(".rf.")[0])
        chapters.append(f"{kind}:{chapter}" if chapter else None)
    Xc = normalize(X[:, : X.shape[1] // 2])
    Z = PCA(max(1, min(64, Xc.shape[1], len(Xc) - 1)), random_state=0).fit_transform(Xc)
    report = {}
    for k in args.k:
        if k >= len(Z):
            print(f"skip k={k}: only {len(Z)} pages")
            continue
        lab = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Z)
        sizes = collections.Counter(lab)
        report[k] = {"sizes_max": max(sizes.values()), "sizes_min": min(sizes.values()),
                     **chapter_split_rate(lab, chapters)}
        write_groups(args.work / f"groups_k{k}.csv", rows, [f"k{k}_{g}" for g in lab])
    if args.dup:
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components
        S = Xc @ Xc.T
        for t in args.dup:
            n, lab = connected_components(csr_matrix(S >= t), directed=False)
            sizes = collections.Counter(lab)
            report[f"dup{t}"] = {"components": int(n), "largest": max(sizes.values()),
                                 "largest_share": max(sizes.values()) / len(lab),
                                 **chapter_split_rate(lab, chapters)}
            write_groups(args.work / f"groups_dup{t}.csv", rows, [f"dup{t}_{g}" for g in lab])
    if args.era:
        era = ["early" if filename_group(r["file"].split(".rf.")[0])[0] == "bare" else "later" for r in rows]
        report["era"] = dict(collections.Counter(era))
        write_groups(args.work / "groups_era.csv", rows, era)
    save_json(report, args.work / "cluster_report.json")
    print(report)


def write_groups(path: Path, rows, labels) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["split", "file", "group"])
        for r, g in zip(rows, labels):
            for f in r["copies"].split("|"):
                wr.writerow([r["split"], f, g])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("embed")
    e.add_argument("--dataset", type=Path, required=True)
    e.add_argument("--work", type=Path, required=True)
    e.add_argument("--encoder", required=True)
    e.add_argument("--batch", type=int, default=16)
    e.add_argument("--budget", type=float, default=0)
    c = sub.add_parser("cluster")
    c.add_argument("--work", type=Path, required=True)
    c.add_argument("--k", type=int, nargs="+", default=[20, 40, 80])
    c.add_argument("--dup", type=float, nargs="*", default=[0.8], help="near-duplicate cosine thresholds")
    c.add_argument("--era", action="store_true", help="also write the two-group era split")
    args = ap.parse_args()
    {"embed": cmd_embed, "cluster": cmd_cluster}[args.cmd](args)


if __name__ == "__main__":
    main()
