#!/usr/bin/env python3
"""Supplementary Figure S3 - independent-cohort validation (combined panels).

Stitches the per-cohort pilot-curve products (QinJ_2012, YachidaS_2019;
produced by analysis/run_cli_pilot_sensitivity.py runs documented in
validation_datasets/results/) into one 2x2 figure: rows are cohorts,
columns are observed-size consistency (eval n = 20) and extrapolation
(eval n = 80).
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SRC_A = ROOT / "validation_datasets" / "results" / "QinJ_2012_core_matched" / "sensitivity_pilot_curves.png"
SRC_B = ROOT / "validation_datasets" / "results" / "YachidaS_2019_core_matched" / "sensitivity_pilot_curves.png"
OUT = ROOT / "figures" / "output"


def main() -> None:
    q = Image.open(SRC_A)
    y = Image.open(SRC_B)
    W, rowh = q.size
    label_h, gap = 90, 30
    canvas = Image.new("RGB", (W, label_h * 2 + rowh * 2 + gap), "white")
    draw = ImageDraw.Draw(canvas)
    font = None
    for fp in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf"):
        try:
            font = ImageFont.truetype(fp, 44)
            break
        except Exception:
            pass
    rows = [("a", "QinJ_2012 (T2D vs control)", q), ("c", "YachidaS_2019 (CRC vs control)", y)]
    yoff = 0
    for letter, title, img in rows:
        draw.text((10, yoff + 18), letter, fill="black", font=font)
        draw.text((W // 2 - 700, yoff + 18), title, fill="black", font=font)
        draw.text((W // 2 + 10, yoff + 18), "b" if letter == "a" else "d", fill="black", font=font)
        canvas.paste(img, (0, yoff + label_h))
        yoff += label_h + rowh + gap
    canvas.save(OUT / "suppfig3_independent_cohorts.png", dpi=(320, 320))
    # PDF export via matplotlib (PIL's PDF plugin needs JPEG support)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    arr = np.asarray(canvas)
    h, w = arr.shape[:2]
    fig = plt.figure(figsize=(w / 320, h / 320), dpi=320)
    fig.add_axes([0, 0, 1, 1]).imshow(arr)
    plt.axis("off")
    fig.savefig(OUT / "suppfig3_independent_cohorts.pdf")
    plt.close(fig)
    print(OUT / "suppfig3_independent_cohorts.png")


if __name__ == "__main__":
    main()
