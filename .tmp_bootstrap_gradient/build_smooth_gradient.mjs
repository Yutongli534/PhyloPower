import fs from "node:fs/promises";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const TMP_DIR = "/Users/liyutong/Desktop/phylopower/.tmp_bootstrap_gradient";
const FINAL_PPTX = "/Users/liyutong/Desktop/phylopower/figures/output/Bootstrap_PERMANOVA_editable_smooth_gradient.pptx";
const WHITE = "#FFFFFF";

const C = {
  W: WHITE,
  B1: "#C8E8F2",
  B2: "#82C5DA",
  B3: "#4E9DB5",
  G1: "#CBE9B9",
  G2: "#91CE7D",
  G3: "#59A064",
  G4: "#2F7E45",
  R1: "#F5B4A5",
  R2: "#E78672",
  R3: "#B95A4D",
};

const SOURCE_8 = [
  [C.W, C.B3, C.B2, C.B1, C.G1, C.G2, C.G3, C.G4],
  [C.B3, C.W, C.B3, C.B2, C.G2, C.G3, C.G4, C.R3],
  [C.B2, C.B3, C.W, C.B3, C.G3, C.G4, C.R3, C.R2],
  [C.B1, C.B2, C.B3, C.W, C.G4, C.R3, C.R2, C.R1],
  [C.G1, C.G2, C.G3, C.G4, C.W, C.B3, C.B2, C.B1],
  [C.G2, C.G3, C.G4, C.R3, C.B3, C.W, C.B3, C.B2],
  [C.G3, C.G4, C.R3, C.R2, C.B2, C.B3, C.W, C.B3],
  [C.G4, C.R3, C.R2, C.R1, C.B1, C.B2, C.B3, C.W],
];

const BOOTSTRAP_TOP_6 = [
  [C.W, C.B2, C.B3, C.G2, C.G3, C.G4],
  [C.B2, C.W, C.B3, C.G3, C.G4, C.G2],
  [C.B3, C.B3, C.W, C.G4, C.G3, C.B3],
  [C.G2, C.G3, C.G4, C.W, C.B2, C.B3],
  [C.G3, C.G4, C.G3, C.B2, C.W, C.B2],
  [C.G4, C.G2, C.B3, C.B3, C.B2, C.W],
];

const BOOTSTRAP_MIDDLE_6 = [
  [C.W, C.R2, C.R1, C.G2, C.G3, C.G4],
  [C.R2, C.W, C.R2, C.G3, C.G4, C.G3],
  [C.R1, C.R2, C.W, C.G4, C.G3, C.G2],
  [C.G2, C.G3, C.G4, C.W, C.R2, C.R1],
  [C.G3, C.G4, C.G3, C.R2, C.W, C.R2],
  [C.G4, C.G3, C.G2, C.R1, C.R2, C.W],
];

const BOOTSTRAP_BOTTOM_6 = [
  [C.W, C.B3, C.B2, C.R1, C.R2, C.R3],
  [C.B3, C.W, C.B3, C.R2, C.R3, C.R2],
  [C.B2, C.B3, C.W, C.R3, C.R2, C.R1],
  [C.R1, C.R2, C.R3, C.W, C.B3, C.B2],
  [C.R2, C.R3, C.R2, C.B3, C.W, C.B3],
  [C.R3, C.R2, C.R1, C.B2, C.B3, C.W],
];

function hexToRgb(hex) {
  const value = hex.replace("#", "");
  return [
    parseInt(value.slice(0, 2), 16),
    parseInt(value.slice(2, 4), 16),
    parseInt(value.slice(4, 6), 16),
  ];
}

function rgbToHex(rgb) {
  return `#${rgb
    .map((v) => Math.max(0, Math.min(255, Math.round(v))).toString(16).padStart(2, "0"))
    .join("")}`;
}

function mixRgb(a, b, t) {
  return a.map((value, i) => value * (1 - t) + b[i] * t);
}

function bilinear(c00, c01, c10, c11, tx, ty) {
  const top = mixRgb(hexToRgb(c00), hexToRgb(c01), tx);
  const bottom = mixRgb(hexToRgb(c10), hexToRgb(c11), tx);
  return rgbToHex(mixRgb(top, bottom, ty));
}

function upscaleWithSmoothGradient(base, targetN) {
  const baseN = base.length;
  const raw = Array.from({ length: targetN }, () => Array(targetN).fill(WHITE));
  for (let row = 0; row < targetN; row += 1) {
    const sourceY = (row * (baseN - 1)) / (targetN - 1);
    const y0 = Math.floor(sourceY);
    const y1 = Math.min(baseN - 1, y0 + 1);
    const ty = sourceY - y0;
    for (let col = 0; col < targetN; col += 1) {
      const sourceX = (col * (baseN - 1)) / (targetN - 1);
      const x0 = Math.floor(sourceX);
      const x1 = Math.min(baseN - 1, x0 + 1);
      const tx = sourceX - x0;
      raw[row][col] = bilinear(
        base[y0][x0],
        base[y0][x1],
        base[y1][x0],
        base[y1][x1],
        tx,
        ty
      );
    }
  }

  // Enforce exact mirror symmetry after interpolation and restore pure-white diagonal.
  for (let row = 0; row < targetN; row += 1) {
    raw[row][row] = WHITE;
    for (let col = row + 1; col < targetN; col += 1) {
      const a = hexToRgb(raw[row][col]);
      const b = hexToRgb(raw[col][row]);
      const mirrored = rgbToHex(a.map((value, i) => (value + b[i]) / 2));
      raw[row][col] = mirrored;
      raw[col][row] = mirrored;
    }
  }
  return raw;
}

function assertSymmetricWhiteDiagonal(matrix, label) {
  matrix.forEach((row, r) => {
    if (row[r] !== WHITE) throw new Error(`${label}: diagonal ${r + 1} is not white`);
    row.forEach((color, c) => {
      if (color !== matrix[c][r]) throw new Error(`${label}: asymmetric at ${r + 1},${c + 1}`);
    });
  });
}

function addText(slide, name, text, left, top, width, height, fontSize, color, bold = false) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    name,
    position: { left, top, width, height },
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = { fontSize, bold, color, alignment: "center", fontFamily: "Arial" };
  return shape;
}

function addArrow(slide, name, geometry, left, top, width, height, rotation, color) {
  slide.shapes.add({
    geometry,
    name,
    position: { left, top, width, height, rotation },
    fill: color,
    line: { style: "solid", fill: "none", width: 0 },
  });
}

function addMatrix(slide, name, matrix, left, top, displaySize) {
  const n = matrix.length;
  const cell = displaySize / n;
  for (let row = 0; row < n; row += 1) {
    for (let col = 0; col < n; col += 1) {
      const diagonal = row === col;
      slide.shapes.add({
        geometry: "rect",
        name: `${name}-r${row + 1}-c${col + 1}`,
        position: {
          left: left + col * cell,
          top: top + row * cell,
          width: cell + 0.08,
          height: cell + 0.08,
        },
        fill: matrix[row][col],
        line: {
          style: "solid",
          fill: diagonal ? "#CDD5D9" : "#FFFFFF",
          width: diagonal ? 0.45 : 0.12,
        },
      });
    }
  }
}

async function writeBlob(path, blob) {
  await fs.writeFile(path, new Uint8Array(await blob.arrayBuffer()));
}

async function main() {
  await fs.mkdir(TMP_DIR, { recursive: true });
  const matrices = [
    { label: "source-20x20", matrix: upscaleWithSmoothGradient(SOURCE_8, 20), x: 125, y: 260, size: 224 },
    { label: "bootstrap-top-14x14", matrix: upscaleWithSmoothGradient(BOOTSTRAP_TOP_6, 14), x: 600, y: 175, size: 112 },
    { label: "bootstrap-middle-14x14", matrix: upscaleWithSmoothGradient(BOOTSTRAP_MIDDLE_6, 14), x: 600, y: 345, size: 112 },
    { label: "bootstrap-bottom-14x14", matrix: upscaleWithSmoothGradient(BOOTSTRAP_BOTTOM_6, 14), x: 600, y: 515, size: 112 },
  ];
  matrices.forEach(({ matrix, label }) => assertSymmetricWhiteDiagonal(matrix, label));
  await fs.writeFile(
    `${TMP_DIR}/matrix-audit.json`,
    JSON.stringify({
      status: "passed",
      invariant: "white main diagonal and exact mirror symmetry",
      method: "bilinear RGB interpolation from original color anchors",
      matrices: matrices.map(({ label, matrix }) => ({ label, size: matrix.length })),
    }, null, 2)
  );

  const presentation = Presentation.create({ slideSize: { width: 1280, height: 720 } });
  const slide = presentation.slides.add();
  const ink = "#182126";
  slide.background.fill = "#FFFFFF";
  slide.shapes.add({
    geometry: "roundRect",
    name: "bootstrap-panel",
    position: { left: 38, top: 44, width: 1204, height: 638 },
    fill: "#EEF3F8",
    line: { style: "solid", fill: "none", width: 0 },
    borderRadius: 54,
  });

  // Arrows are created first and remain behind the editable heatmap cells.
  addArrow(slide, "source-to-top", "rightArrow", 390, 218, 170, 28, -28, ink);
  addArrow(slide, "source-to-middle", "rightArrow", 400, 342, 145, 28, 0, ink);
  addArrow(slide, "source-to-bottom", "rightArrow", 390, 468, 170, 28, 28, ink);
  addArrow(slide, "metric-to-top", "leftArrow", 750, 218, 170, 28, -28, ink);
  addArrow(slide, "metric-to-middle", "leftArrow", 762, 342, 150, 28, 0, ink);
  addArrow(slide, "metric-to-bottom", "leftArrow", 750, 468, 170, 28, 28, ink);

  addText(slide, "figure-title", "Bootstrap resampling & PERMANOVA", 145, 65, 990, 66, 44, ink, true);
  addText(slide, "omega-squared", "ω²", 940, 290, 170, 58, 46, ink, false);
  addText(slide, "p-value", "p value", 925, 350, 200, 58, 42, ink, false);
  addText(slide, "repeat-symbol", "↻", 100, 602, 70, 60, 50, "#39474C", false);
  addText(slide, "iteration-label", "B = 500 iterations", 160, 610, 390, 52, 38, ink, false);

  matrices.forEach(({ label, matrix, x, y, size }) => addMatrix(slide, label, matrix, x, y, size));
  [472, 488, 504].forEach((y, i) => {
    slide.shapes.add({
      geometry: "ellipse",
      name: `ellipsis-${i + 1}`,
      position: { left: 652, top: y, width: 8, height: 8 },
      fill: ink,
      line: { style: "solid", fill: "none", width: 0 },
    });
  });

  slide.speakerNotes.textFrame.setText(
    "[Sources]\n- User-provided reference image: codex-clipboard-8bee9934-b9f9-4d97-8716-31d90bce09f8.png\n- The original blue/green/red color regions were retained as anchors and bilinearly interpolated into finer grids. The main diagonal remains pure white and every matrix remains exactly symmetric."
  );

  await writeBlob(`${TMP_DIR}/slide-01.png`, await presentation.export({ slide, format: "png", scale: 2 }));
  await fs.writeFile(`${TMP_DIR}/slide-01.layout.json`, await (await slide.export({ format: "layout" })).text());
  const inspect = await presentation.inspect({ kind: "slide,shape,textbox,notes,image", maxChars: 60000 });
  await fs.writeFile(`${TMP_DIR}/inspect.ndjson`, inspect.ndjson);
  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(FINAL_PPTX);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
