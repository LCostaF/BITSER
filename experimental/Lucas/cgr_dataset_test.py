#!/usr/bin/env python3
"""
cgr_dataset.py - batch-generate forward CGR (FCGR) images from FASTA datasets.

Same CGR method as cgr_image.py (USM, bidirectional seeding, corner order
C,T,A,G = TL,TR,BL,BR, log10(count+1) intensity, inverted grayscale), but the
sequences come from FASTA files instead of the EGFR default.

Expected layout (next to this script):

    sequences/
        Dengue/   a.fasta  b.fasta ...
        SARSCOV/  ...
        HBV/      ...

Output (created automatically):

    images/
        Dengue/
            64x64/                   <- one folder per image size
                a/                      <- one folder per .fasta file
                    <sequence header>.png   <- one image per sequence
                b/ ...
            128x128/ ... 1024x1024/
        ...

Only the FORWARD image is generated. At most --limit sequences are converted
per FASTA file (default 5; use --limit 0 for no limit).
Each sequence is rendered at several sizes (default 64, 128, 256, 512, 1024,
i.e. 64x64 ... 1024x1024); change with --sizes, e.g. --sizes 256 512x256.

Usage:
    python cgr_dataset.py
    python cgr_dataset.py --limit 10 --sizes 128 256
    python cgr_dataset.py --cmap blue_red --dataset HBV

Requires: numpy, Pillow
"""

import argparse
import os
import re
import sys

import numpy as np
from PIL import Image

DEFAULT_ALPHABET = "CTAG"   # TL, TR, BL, BR
FASTA_EXT = (".fasta", ".fa", ".fna", ".fas")


# --------------------------------------------------------------------------
# FASTA reading
# --------------------------------------------------------------------------
def read_fasta(path, limit=0):
    """Yield (header, raw_sequence) for up to `limit` records (0 = all)."""
    header, chunks, n = None, [], 0
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                    n += 1
                    if limit and n >= limit:
                        return
                header, chunks = line[1:].strip(), []
            elif header is not None and line:
                chunks.append(line)
        if header is not None and not (limit and n >= limit):
            yield header, "".join(chunks)


def clean_sequence(raw):
    seq = raw.upper().replace("U", "T")
    return "".join(c for c in seq if c in "ACGT")


def safe_name(header, max_len=120):
    """Header -> valid filename (no extension)."""
    name = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", header)
    name = re.sub(r"\s+", "_", name).strip("._ ")
    return (name[:max_len].rstrip("._ ")) or "sequence"


def unique_path(folder, name, ext=".png"):
    path = os.path.join(folder, name + ext)
    k = 2
    while os.path.exists(path):
        path = os.path.join(folder, f"{name}_{k}{ext}")
        k += 1
    return path


# --------------------------------------------------------------------------
# CGR / USM (identical to cgr_image.py, forward only output)
# --------------------------------------------------------------------------
def corner_table(alphabet):
    return {l: (float(i & 1), float(i >> 1)) for i, l in enumerate(alphabet)}


def usm_iterate(seq, corners, seed=(0.5, 0.5)):
    pts = np.empty((len(seq), 2), dtype=np.float64)
    x, y = seed
    for i, ch in enumerate(seq):
        cx, cy = corners[ch]
        x += 0.5 * (cx - x)
        y += 0.5 * (cy - y)
        pts[i, 0] = x
        pts[i, 1] = y
    return pts


def forward_coordinates(seq, corners, seed_mode="bidirectional"):
    """Forward USM coordinates, with the site's seeding modes."""
    mid = (0.5, 0.5)
    rev = seq[::-1]
    fw = usm_iterate(seq, corners, mid)
    if seed_mode == "middle":
        return fw
    if seed_mode == "circular":
        for _ in range(1000):
            tail = fw[-1].copy()
            fw = usm_iterate(seq, corners, tuple(tail))
            if np.array_equal(tail, fw[-1]):
                break
        return fw
    bw = usm_iterate(rev, corners, mid)  # bidirectional
    for _ in range(1000):
        head, tail = bw[-1].copy(), fw[-1].copy()
        fw = usm_iterate(seq, corners, tuple(head))
        bw = usm_iterate(rev, corners, tuple(tail))
        if np.array_equal(head, bw[-1]) and np.array_equal(tail, fw[-1]):
            break
    return fw


def frequency_cgr(points, width, height):
    ix = np.minimum((points[:, 0] * width).astype(np.int64), width - 1)
    iy = np.minimum((points[:, 1] * height).astype(np.int64), height - 1)
    return np.bincount(iy * width + ix, minlength=width * height).reshape(height, width)


def apply_scale(counts, scale):
    c = counts.astype(np.float64)
    if scale == "log":
        c = np.log10(c + 1)
    elif scale == "sqrt":
        c = np.sqrt(c)
    m = c.max()
    return c / m if m > 0 else c


def site_colormap():
    t = np.zeros((256, 3), dtype=np.int64)
    for i in range(256):
        if i <= 32:
            r, g, b = 0, 0, 127 + 4 * i
        elif i <= 96:
            r, g, b = 0, 4 * (i - 33) + 3, 255
        elif i <= 160:
            r, g, b = 4 * (i - 96) - 1, 255, 255 - 4 * (i - 96)
        elif i <= 224:
            r, g, b = 255, 255 - 4 * (i - 160), 0
        else:
            r, g, b = 255 - 4 * (i - 223), 0, 0
        t[i] = [min(max(v, 0), 255) for v in (r, g, b)]
    return t.astype(np.uint8)


def save_png(counts, path, scale, cmap):
    val = np.floor(255 * apply_scale(counts, scale) + 0.5).astype(np.int64)
    if cmap == "gray":
        img = Image.fromarray((255 - val).astype(np.uint8), "L")
    else:
        img = Image.fromarray(site_colormap()[val], "RGB")
    img.save(path, format="PNG")


# --------------------------------------------------------------------------
# Batch driver
# --------------------------------------------------------------------------
def parse_sizes(items):
    sizes = []
    for it in items:
        m = re.fullmatch(r"(\d+)(?:[xX](\d+))?", it.strip())
        if not m or int(m.group(1)) < 1 or int(m.group(2) or m.group(1)) < 1:
            sys.exit(f"invalid size '{it}' (use N or WxH)")
        w = int(m.group(1))
        h = int(m.group(2) or w)
        if (w, h) not in sizes:
            sizes.append((w, h))
    return sizes


def build_parser():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description="Batch forward-CGR images from FASTA datasets.")
    p.add_argument("--sequences-dir", default=os.path.join(here, "sequences"))
    p.add_argument("--images-dir", default=os.path.join(here, "images"))
    p.add_argument("--dataset", action="append",
                   help="only process this dataset folder (repeatable); default: all")
    p.add_argument("--limit", type=int, default=5,
                   help="max sequences per FASTA file (default 5, 0 = no limit)")
    p.add_argument("--sizes", nargs="+", default=["64", "128", "256", "512", "1024"],
                   help="image sizes: N (square NxN) or WxH (default: 64 128 256 512 1024)")
    p.add_argument("--seed", choices=["bidirectional", "middle", "circular"],
                   default="bidirectional")
    p.add_argument("--scale", choices=["log", "sqrt", "linear"], default="log")
    p.add_argument("--cmap", choices=["gray", "blue_red"], default="gray")
    p.add_argument("--alphabet", default=DEFAULT_ALPHABET)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    sizes = parse_sizes(args.sizes)
    alphabet = args.alphabet.upper()
    if sorted(alphabet) != list("ACGT"):
        sys.exit("--alphabet must be a permutation of ACGT")
    if not os.path.isdir(args.sequences_dir):
        sys.exit(f"sequences folder not found: {args.sequences_dir}")
    corners = corner_table(alphabet)

    datasets = sorted(d for d in os.listdir(args.sequences_dir)
                      if os.path.isdir(os.path.join(args.sequences_dir, d)))
    if args.dataset:
        missing = [d for d in args.dataset if d not in datasets]
        if missing:
            sys.exit(f"dataset(s) not found in sequences dir: {missing}")
        datasets = args.dataset

    total = 0
    for ds in datasets:
        ds_dir = os.path.join(args.sequences_dir, ds)
        fastas = sorted(f for f in os.listdir(ds_dir)
                        if f.lower().endswith(FASTA_EXT)
                        and os.path.isfile(os.path.join(ds_dir, f)))
        print(f"== {ds}: {len(fastas)} FASTA file(s)")
        for fname in fastas:
            stem = os.path.splitext(fname)[0]
            out_dir = os.path.join(args.images_dir, ds)
            size_dirs = {}
            for (w, h) in sizes:
                size_dirs[(w, h)] = os.path.join(out_dir, f"{w}x{h}", stem)
                os.makedirs(size_dirs[(w, h)], exist_ok=True)
            n = 0
            for header, raw in read_fasta(os.path.join(ds_dir, fname), args.limit):
                seq = clean_sequence(raw)
                if not seq:
                    print(f"   [skip] empty sequence: {header[:60]}", file=sys.stderr)
                    continue
                pts = forward_coordinates(seq, corners, args.seed)  # once per sequence
                name = safe_name(header)
                for (w, h) in sizes:
                    counts = frequency_cgr(pts, w, h)
                    save_png(counts, unique_path(size_dirs[(w, h)], name),
                             args.scale, args.cmap)
                n += 1
            total += n
            print(f"   {fname}: {n} sequence(s) x {len(sizes)} size(s) -> {out_dir}")
            total += n * (len(sizes) - 1)
    print(f"Done. {total} image(s) written to {args.images_dir}")


if __name__ == "__main__":
    main()