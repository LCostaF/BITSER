#!/usr/bin/env python3
"""
Compute Moran's I and Geary's C for grayscale images, using both queen and
rook contiguity.

Expected layout (relative to this script):

    images/<dataset>/<dimensions>/<variant>/<image>.png

Output mirrors it, with one main subfolder per contiguity type:

    metrics/queen/<dataset>/<dimensions>/<variant>/<image>.txt
    metrics/rook/<dataset>/<dimensions>/<variant>/<image>.txt

Each pixel is a spatial unit, its grayscale value (0-255) is the variable, and
neighbours are defined by contiguity on the pixel grid:

    queen : 8 neighbours (shared edge or corner)   [as spdep::poly2nb]
    rook  : 4 neighbours (shared edge only)

Weights are binary (w_ij = 1 if i and j are neighbours, else 0). The weight
matrix is never built explicitly (a 1024x1024 image would need a
1,048,576 x 1,048,576 matrix); the sums over neighbour pairs are computed with
shifted copies of the image, which gives exactly the same result.

    Moran's I = (N / S0) * sum_ij w_ij (x_i - m)(x_j - m) / sum_i (x_i - m)^2
    Geary's C = ((N-1) / (2 S0)) * sum_ij w_ij (x_i - x_j)^2 / sum_i (x_i - m)^2

with N = number of pixels, S0 = sum of all weights, m = mean of x.
Expected value under no spatial autocorrelation: E[I] = -1/(N-1), E[C] = 1.

Optional quadrant analysis (--quadrants): after the global values, the image is
split into 4 quadrants (TL, TR, BL, BR), each of those into 4 sub-quadrants,
and so on. I and C are computed independently for every quadrant (neighbours
never cross a quadrant border) and appended to the image's .txt, one block per
level. A region is only split while all its quadrants stay >= --min-size
(default 64), so a 1024x1024 image gives levels of 512, 256, 128 and 64 pixels
(4 + 16 + 64 + 256 quadrants), and a 64x64 image gets only its global values.
Labels are paths, e.g. "TR.BL" = bottom-left quadrant of the top-right quadrant.

Usage:
    python spatial_metrics.py
    python spatial_metrics.py --quadrants
    python spatial_metrics.py --quadrants --min-size 128
    python spatial_metrics.py --images-dir images --metrics-dir metrics
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".png"}

# (row offset, column offset). Each undirected pair appears twice in the
# full set (once per direction), matching a symmetric weight matrix.
ROOK_OFFSETS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
QUEEN_OFFSETS = ROOK_OFFSETS + [(-1, -1), (-1, 1), (1, -1), (1, 1)]

CONTIGUITIES = {"queen": QUEEN_OFFSETS, "rook": ROOK_OFFSETS}


def _overlap_slices(shape, dy, dx):
    """Slices (a, b) such that x[a] pairs each cell with its neighbour x[b]
    at offset (dy, dx), skipping cells whose neighbour falls outside."""
    h, w = shape

    def axis(n, d):
        if d > 0:
            return slice(0, n - d), slice(d, n)
        if d < 0:
            return slice(-d, n), slice(0, n + d)
        return slice(0, n), slice(0, n)

    ay, by = axis(h, dy)
    ax, bx = axis(w, dx)
    return (ay, ax), (by, bx)


def morans_i_geary_c(img, contiguity="queen"):
    """Return (I, C) for a 2-D array. NaN for both if the image is constant."""
    x = np.asarray(img, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("Expected a 2-D grayscale array")

    n = x.size
    offsets = CONTIGUITIES[contiguity]

    z = x - x.mean()
    denom = np.sum(z * z)

    s0 = 0            # sum of weights
    cross = 0.0       # sum_ij w_ij z_i z_j
    sqdiff = 0.0      # sum_ij w_ij (x_i - x_j)^2

    for dy, dx in offsets:
        a, b = _overlap_slices(x.shape, dy, dx)
        s0 += z[a].size
        cross += np.sum(z[a] * z[b])
        d = x[a] - x[b]
        sqdiff += np.sum(d * d)

    if denom == 0 or s0 == 0:
        return float("nan"), float("nan")

    moran = (n / s0) * (cross / denom)
    geary = ((n - 1) / (2.0 * s0)) * (sqdiff / denom)
    return float(moran), float(geary)


QUADRANT_NAMES = [("TL", 0, 0), ("TR", 0, 1), ("BL", 1, 0), ("BR", 1, 1)]


def quadrant_levels(arr, contiguity, min_size):
    """Recursively split the image into quadrants and compute I and C on each
    one independently (neighbours never cross a quadrant border).

    A region is split into 4 only if every resulting quadrant would still be
    at least `min_size` pixels in both height and width. Returns a list of
    levels; level k (k >= 1) is a list of dicts for the 4**k regions of that
    depth, in reading order, each with: label, rows, cols, I, C.
    Level 0 (the whole image) is not included.
    """
    h, w = arr.shape
    # Each region is (label, r0, r1, c0, c1); labels are paths like "TL.BR".
    current = [("", 0, h, 0, w)]
    levels = []

    while True:
        nxt = []
        for label, r0, r1, c0, c1 in current:
            rm, cm = (r0 + r1) // 2, (c0 + c1) // 2
            # The smaller half is the floor half; both must reach min_size.
            if (rm - r0) < min_size or (cm - c0) < min_size:
                continue
            for name, qr, qc in QUADRANT_NAMES:
                nr0, nr1 = (r0, rm) if qr == 0 else (rm, r1)
                nc0, nc1 = (c0, cm) if qc == 0 else (cm, c1)
                nxt.append((f"{label}.{name}" if label else name,
                            nr0, nr1, nc0, nc1))
        if not nxt:
            break

        level = []
        for label, r0, r1, c0, c1 in nxt:
            moran, geary = morans_i_geary_c(arr[r0:r1, c0:c1], contiguity)
            level.append({"label": label, "rows": (r0, r1 - 1),
                          "cols": (c0, c1 - 1), "I": moran, "C": geary})
        levels.append(level)
        current = nxt

    return levels


def write_result(out_path, arr, contiguity, moran, geary, levels):
    n = arr.size
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"Moran's I: {moran:.10f}\n")
        f.write(f"Geary's C: {geary:.10f}\n")
        f.write(f"Expected Moran's I: {-1.0 / (n - 1):.10f}\n")
        f.write("Expected Geary's C: 1.0000000000\n")
        f.write(f"Image size: {arr.shape[1]}x{arr.shape[0]}\n")
        f.write(f"Contiguity: {contiguity}\n")

        for depth, level in enumerate(levels, 1):
            r0, r1 = level[0]["rows"]
            c0, c1 = level[0]["cols"]
            f.write(f"\n# Level {depth}: {4 ** depth} quadrants of about "
                    f"{c1 - c0 + 1}x{r1 - r0 + 1}\n")
            f.write("# label | rows | cols | Moran's I | Geary's C\n")
            for q in level:
                f.write(f"{q['label']} | {q['rows'][0]}-{q['rows'][1]} | "
                        f"{q['cols'][0]}-{q['cols'][1]} | "
                        f"{q['I']:.10f} | {q['C']:.10f}\n")


def process_image(img_path, rel, metrics_dir, quadrants=False, min_size=64):
    """Compute both contiguity versions for one image; returns {name: (I, C)}."""
    with Image.open(img_path) as im:
        arr = np.array(im.convert("L"))  # grayscale, 0 = black, 255 = white

    results = {}
    for name in CONTIGUITIES:
        moran, geary = morans_i_geary_c(arr, name)
        levels = quadrant_levels(arr, name, min_size) if quadrants else []
        out_path = metrics_dir / name / rel.with_suffix(".txt")
        write_result(out_path, arr, name, moran, geary, levels)
        results[name] = (moran, geary)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images-dir", default="images")
    parser.add_argument("--metrics-dir", default="metrics")
    parser.add_argument("--quadrants", action="store_true",
                        help="also compute I and C for quadrants, sub-quadrants, "
                             "and so on, down to --min-size")
    parser.add_argument("--min-size", type=int, default=64,
                        help="smallest quadrant side in pixels; a region is only "
                             "split if all its quadrants stay at least this big "
                             "(default: 64)")
    args = parser.parse_args()

    if args.min_size < 2:
        sys.exit("--min-size must be at least 2")

    root = Path(__file__).resolve().parent
    images_dir = root / args.images_dir
    metrics_dir = root / args.metrics_dir

    if not images_dir.is_dir():
        sys.exit(f"Images folder not found: {images_dir}")

    files = sorted(p for p in images_dir.rglob("*")
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not files:
        sys.exit(f"No PNG images found under {images_dir}")

    failures = 0
    for i, img_path in enumerate(files, 1):
        rel = img_path.relative_to(images_dir)
        try:
            res = process_image(img_path, rel, metrics_dir,
                                args.quadrants, args.min_size)
            q, r = res["queen"], res["rook"]
            print(f"[{i}/{len(files)}] {rel}  "
                  f"queen I={q[0]:.6f} C={q[1]:.6f}  "
                  f"rook I={r[0]:.6f} C={r[1]:.6f}")
        except Exception as exc:  # keep going on a bad file
            failures += 1
            print(f"[{i}/{len(files)}] {rel}  FAILED: {exc}", file=sys.stderr)

    print(f"Done. {len(files) - failures} processed, {failures} failed. "
          f"Results in {metrics_dir / 'queen'} and {metrics_dir / 'rook'}")


if __name__ == "__main__":
    main()