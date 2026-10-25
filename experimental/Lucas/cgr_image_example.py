#!/usr/bin/env python3
"""
cgr_image.py - Chaos Game Representation (CGR / USM) images from sequences.

Reproduces the density plots of https://usm.github.io/fold/ (code: usm/fold
index.js + https://usm.github.io/3/usm.mjs):
  * USM generalisation of CGR: each point moves halfway to the corner of the
    next base. Default seeding is 'bidirectional', as on the site (forward
    pass seeded by the end of the backward pass and vice versa, iterated
    until stable). 'middle' (classic CGR, seed 0.5,0.5) and 'circular' are
    also available.
  * Corner order C, T, A, G  ->  top-left, top-right, bottom-left, bottom-right.
  * Forward and backward iterations, each producing a frequency CGR (FCGR).
  * Intensity as on the site: log10(count + 1) / log10(max count + 1),
    quantised to 0..255. Grayscale is inverted like the site's canvas
    (empty bin = white, densest bin = black). The colour option uses the
    site's own 256-entry blue->red colormap (empty bin = dark blue).
  * Default size 256x256 = 65536 bins (k-mer length 8).

Default test sequence: NG_007726.3 (Homo sapiens EGFR RefSeqGene, LRG_304),
the same one the website uses. It is looked up in this order:
    1. a local file  NG_007726.3.txt  (FASTA or raw) next to this script / cwd
    2. downloaded from NCBI (efetch) and cached to that file

Usage examples
--------------
    python cgr_image.py                                  # EGFR, 256x256, fwd+bwd, gray
    python cgr_image.py --width 512 --height 512
    python cgr_image.py --seq GATTACA --width 64 -o gattaca
    python cgr_image.py --file my.fasta --direction forward --cmap blue_red
    python cgr_image.py --scale linear                   # raw (non-log) counts

Output: PNG file(s), "<prefix>_forward.png" and/or "<prefix>_backward.png".

Requires: numpy, Pillow  (pip install numpy pillow)
"""

import argparse
import os
import sys
import urllib.request

import numpy as np
from PIL import Image

DEFAULT_ACCESSION = "NG_007726.3"
DEFAULT_ALPHABET = "CTAG"  # corner order matching the site


# --------------------------------------------------------------------------
# Sequence input
# --------------------------------------------------------------------------
def clean_sequence(text):
    """Strip FASTA headers/whitespace, upper-case, U->T, keep only A/C/G/T."""
    lines = [l.strip() for l in text.splitlines() if l.strip() and not l.startswith(">")]
    seq = "".join(lines).upper().replace("U", "T")
    kept = "".join(c for c in seq if c in "ACGT")
    dropped = len(seq) - len(kept)
    if dropped:
        print(f"[info] ignored {dropped} non-ACGT characters", file=sys.stderr)
    return kept


def load_default_sequence():
    """Load NG_007726.3 from a local file, else download from NCBI and cache."""
    fname = DEFAULT_ACCESSION + ".txt"
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), fname),
        os.path.join(os.getcwd(), fname),
    ]
    for path in candidates:
        if os.path.isfile(path):
            print(f"[info] using local sequence file {path}", file=sys.stderr)
            with open(path) as fh:
                return clean_sequence(fh.read())

    url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        f"?db=nuccore&id={DEFAULT_ACCESSION}&rettype=fasta&retmode=text"
    )
    print(f"[info] downloading {DEFAULT_ACCESSION} from NCBI ...", file=sys.stderr)
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            text = resp.read().decode()
    except Exception as exc:  # noqa: BLE001
        sys.exit(
            f"Could not download {DEFAULT_ACCESSION} ({exc}).\n"
            f"Save the sequence (FASTA or raw) as '{fname}' next to this script,\n"
            "or pass --file / --seq."
        )
    with open(candidates[0], "w") as fh:
        fh.write(text)
    print(f"[info] cached to {candidates[0]}", file=sys.stderr)
    return clean_sequence(text)


# --------------------------------------------------------------------------
# CGR / USM
# --------------------------------------------------------------------------
def corner_table(alphabet):
    """
    Corner of each letter in image space (u = horizontal, v = vertical, down),
    taken in row-major order: index 0 -> TL (0,0), 1 -> TR (1,0),
    2 -> BL (0,1), 3 -> BR (1,1).
    With alphabet 'CTAG': C=TL, T=TR, A=BL, G=BR (as in the site's plots).
    """
    return {letter: (float(i & 1), float(i >> 1)) for i, letter in enumerate(alphabet)}


def usm_iterate(seq, corners, seed=(0.5, 0.5)):
    """Iterate the USM/CGR map: p <- p + 0.5 * (corner - p). Returns (n,2)."""
    pts = np.empty((len(seq), 2), dtype=np.float64)
    x, y = seed
    for i, ch in enumerate(seq):
        cx, cy = corners[ch]
        x += 0.5 * (cx - x)
        y += 0.5 * (cy - y)
        pts[i, 0] = x
        pts[i, 1] = y
    return pts


def usm_coordinates(seq, corners, seed_mode="bidirectional"):
    """
    Returns {'forward': (n,2), 'backward': (n,2)} (points in sequence order).

    seed_mode 'middle'       : seed (0.5, 0.5) in both directions (classic CGR).
    seed_mode 'circular'     : each direction re-seeded from its own end point
                               until stable (sequence treated as circular).
    seed_mode 'bidirectional': forward seeded with the backward pass's final
                               point, backward seeded with the forward pass's
                               final point, repeated until stable (the site).
    """
    mid = (0.5, 0.5)
    rev = seq[::-1]
    fw = usm_iterate(seq, corners, mid)
    bw = usm_iterate(rev, corners, mid)  # reversed order: bw[-1] is position 0
    if seed_mode == "circular":
        for _ in range(1000):
            tail = fw[-1].copy()
            fw = usm_iterate(seq, corners, tuple(tail))
            if np.array_equal(tail, fw[-1]):
                break
        for _ in range(1000):
            head = bw[-1].copy()
            bw = usm_iterate(rev, corners, tuple(head))
            if np.array_equal(head, bw[-1]):
                break
    elif seed_mode == "bidirectional":
        for _ in range(1000):
            head, tail = bw[-1].copy(), fw[-1].copy()
            fw = usm_iterate(seq, corners, tuple(head))
            bw = usm_iterate(rev, corners, tuple(tail))
            if np.array_equal(head, bw[-1]) and np.array_equal(tail, fw[-1]):
                break
    return {"forward": fw, "backward": bw[::-1]}


def frequency_cgr(points, width, height):
    """Count points per bin -> (height, width) integer array (row 0 = y 0)."""
    ix = np.minimum((points[:, 0] * width).astype(np.int64), width - 1)
    iy = np.minimum((points[:, 1] * height).astype(np.int64), height - 1)
    counts = np.bincount(iy * width + ix, minlength=width * height)
    return counts.reshape(height, width)


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
def apply_scale(counts, scale):
    """Normalise to [0,1]. 'log' = log10(c+1)/log10(max+1), as on the site."""
    c = counts.astype(np.float64)
    if scale == "log":
        c = np.log10(c + 1)
    elif scale == "sqrt":
        c = np.sqrt(c)
    m = c.max()
    return c / m if m > 0 else c


def site_colormap():
    """The site's 256-entry blue_red (jet-like) colormap: [0,0,127] ... [127,0,0]."""
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
    # quantise to 0..255 exactly like the site: Math.round(255 * value)
    val = np.floor(255 * apply_scale(counts, scale) + 0.5).astype(np.int64)
    if cmap == "gray":
        # single-channel 8-bit; inverted like the site (0 count = white)
        img = Image.fromarray((255 - val).astype(np.uint8), "L")
    elif cmap == "blue_red":
        img = Image.fromarray(site_colormap()[val], "RGB")
    else:  # viridis (extra, not on the site)
        stops = np.array([[68, 1, 84], [59, 82, 139], [33, 145, 140],
                          [94, 201, 98], [253, 231, 37]], dtype=np.float64)
        pos = np.linspace(0, 1, len(stops))
        rgb = np.stack([np.interp(val / 255, pos, stops[:, k]) for k in range(3)], -1)
        img = Image.fromarray(rgb.astype(np.uint8), "RGB")
    img.save(path, format="PNG")
    return path


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(description="Generate CGR/FCGR PNG images from sequences.")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--seq", help="sequence string (ACGT)")
    src.add_argument("--file", help="FASTA or raw text file with the sequence")
    p.add_argument("--width", type=int, default=256, help="image width in pixels (default 256)")
    p.add_argument("--height", type=int, default=None, help="image height (default = width)")
    p.add_argument("-o", "--output", default="cgr", help="output prefix (default 'cgr')")
    p.add_argument("--direction", choices=["forward", "backward", "both"], default="both")
    p.add_argument("--seed", choices=["bidirectional", "middle", "circular"],
                   default="bidirectional", help="seeding mode (default bidirectional, as the site)")
    p.add_argument("--scale", choices=["log", "sqrt", "linear"], default="log")
    p.add_argument("--cmap", choices=["gray", "blue_red", "viridis"], default="gray")
    p.add_argument("--alphabet", default=DEFAULT_ALPHABET,
                   help="corner order of the 4 letters (default CTAG)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    width = args.width
    height = args.height or width
    if width < 1 or height < 1:
        sys.exit("width/height must be >= 1")
    alphabet = args.alphabet.upper()
    if sorted(alphabet) != list("ACGT"):
        sys.exit("--alphabet must be a permutation of ACGT")

    if args.seq:
        seq = clean_sequence(args.seq)
    elif args.file:
        with open(args.file) as fh:
            seq = clean_sequence(fh.read())
    else:
        seq = load_default_sequence()
    if not seq:
        sys.exit("empty sequence")
    print(f"[info] sequence length: {len(seq)}  |  image: {width}x{height} "
          f"({width * height} bins)", file=sys.stderr)

    corners = corner_table(alphabet)
    directions = ["forward", "backward"] if args.direction == "both" else [args.direction]
    coords = usm_coordinates(seq, corners, args.seed)
    for d in directions:
        pts = coords[d]
        counts = frequency_cgr(pts, width, height)
        out = f"{args.output}_{d}.png"
        save_png(counts, out, args.scale, args.cmap)
        print(f"saved {out}  (max bin count {counts.max()}, "
              f"occupied bins {(counts > 0).sum()})")


if __name__ == "__main__":
    main()