import fs from "node:fs/promises";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const OUT_DIR = "/Users/liyutong/Desktop/phylopower/figures/output/bootstrap_matrix_exports";
const DECK_PATH = `${OUT_DIR}/Bootstrap_PERMANOVA_four_matrices_editable.pptx`;
const WHITE = "#FFFFFF";

const C = {
  W: WHITE,
  B1: "#C8E8F2", B2: "#82C5DA", B3: "#4E9DB5",
  G1: "#CBE9B9", G2: "#91CE7D", G3: "#59A064", G4: "#2F7E45",
  R1: "#F5B4A5", R2: "#E78672", R3: "#B95A4D",
};

const SOURCE_8 = [
  [C.W,C.B3,C.B2,C.B1,C.G1,C.G2,C.G3,C.G4],
  [C.B3,C.W,C.B3,C.B2,C.G2,C.G3,C.G4,C.R3],
  [C.B2,C.B3,C.W,C.B3,C.G3,C.G4,C.R3,C.R2],
  [C.B1,C.B2,C.B3,C.W,C.G4,C.R3,C.R2,C.R1],
  [C.G1,C.G2,C.G3,C.G4,C.W,C.B3,C.B2,C.B1],
  [C.G2,C.G3,C.G4,C.R3,C.B3,C.W,C.B3,C.B2],
  [C.G3,C.G4,C.R3,C.R2,C.B2,C.B3,C.W,C.B3],
  [C.G4,C.R3,C.R2,C.R1,C.B1,C.B2,C.B3,C.W],
];

const TOP_6 = [
  [C.W,C.B2,C.B3,C.G2,C.G3,C.G4],
  [C.B2,C.W,C.B3,C.G3,C.G4,C.G2],
  [C.B3,C.B3,C.W,C.G4,C.G3,C.B3],
  [C.G2,C.G3,C.G4,C.W,C.B2,C.B3],
  [C.G3,C.G4,C.G3,C.B2,C.W,C.B2],
  [C.G4,C.G2,C.B3,C.B3,C.B2,C.W],
];

const MIDDLE_6 = [
  [C.W,C.R2,C.R1,C.G2,C.G3,C.G4],
  [C.R2,C.W,C.R2,C.G3,C.G4,C.G3],
  [C.R1,C.R2,C.W,C.G4,C.G3,C.G2],
  [C.G2,C.G3,C.G4,C.W,C.R2,C.R1],
  [C.G3,C.G4,C.G3,C.R2,C.W,C.R2],
  [C.G4,C.G3,C.G2,C.R1,C.R2,C.W],
];

const BOTTOM_6 = [
  [C.W,C.B3,C.B2,C.R1,C.R2,C.R3],
  [C.B3,C.W,C.B3,C.R2,C.R3,C.R2],
  [C.B2,C.B3,C.W,C.R3,C.R2,C.R1],
  [C.R1,C.R2,C.R3,C.W,C.B3,C.B2],
  [C.R2,C.R3,C.R2,C.B3,C.W,C.B3],
  [C.R3,C.R2,C.R1,C.B2,C.B3,C.W],
];

function hexToRgb(hex) {
  const s = hex.slice(1);
  return [parseInt(s.slice(0,2),16), parseInt(s.slice(2,4),16), parseInt(s.slice(4,6),16)];
}

function rgbToHex(rgb) {
  return `#${rgb.map(v => Math.max(0,Math.min(255,Math.round(v))).toString(16).padStart(2,"0")).join("")}`;
}

function adjust(hex, amount) {
  const rgb = hexToRgb(hex);
  if (amount >= 0) return rgbToHex(rgb.map(v => v + (255 - v) * amount));
  return rgbToHex(rgb.map(v => v * (1 + amount)));
}

function resampleClear(base, targetN) {
  const baseN = base.length;
  const matrix = Array.from({length: targetN}, () => Array(targetN).fill(WHITE));
  for (let row = 0; row < targetN; row += 1) {
    for (let col = row; col < targetN; col += 1) {
      if (row === col) continue;
      const br = Math.round((row * (baseN - 1)) / (targetN - 1));
      const bc = Math.round((col * (baseN - 1)) / (targetN - 1));
      let color = base[br][bc];
      if (color === WHITE) {
        const neighbor = Math.min(baseN - 1, Math.max(0, br + (bc >= br ? 1 : -1)));
        color = base[br][neighbor];
      }
      const variation = (((row * 3 + col * 5) % 3) - 1) * 0.035;
      color = adjust(color, variation);
      matrix[row][col] = color;
      matrix[col][row] = color;
    }
  }
  return matrix;
}

function assertMatrix(matrix, label) {
  matrix.forEach((row, r) => {
    if (row[r] !== WHITE) throw new Error(`${label}: non-white diagonal`);
    row.forEach((value, c) => {
      if (value !== matrix[c][r]) throw new Error(`${label}: asymmetric`);
    });
  });
}

function addMatrix(slide, name, matrix, left, top, size) {
  const n = matrix.length;
  const cell = size / n;
  for (let r = 0; r < n; r += 1) {
    for (let c = 0; c < n; c += 1) {
      const diagonal = r === c;
      slide.shapes.add({
        geometry: "rect",
        name: `${name}-r${r + 1}-c${c + 1}`,
        position: {
          left: left + c * cell,
          top: top + r * cell,
          width: cell + 0.05,
          height: cell + 0.05,
        },
        fill: matrix[r][c],
        line: {
          style: "solid",
          fill: diagonal ? "#B8C4C9" : "#F8FAFB",
          width: diagonal ? 0.65 : 0.38,
        },
      });
    }
  }
}

async function writeBlob(path, blob) {
  await fs.writeFile(path, new Uint8Array(await blob.arrayBuffer()));
}

async function main() {
  await fs.mkdir(OUT_DIR, {recursive: true});
  const specs = [
    {name: "source-matrix-12x12", file: "01_source_matrix_12x12.png", matrix: resampleClear(SOURCE_8, 12)},
    {name: "bootstrap-top-8x8", file: "02_bootstrap_top_8x8.png", matrix: resampleClear(TOP_6, 8)},
    {name: "bootstrap-middle-8x8", file: "03_bootstrap_middle_8x8.png", matrix: resampleClear(MIDDLE_6, 8)},
    {name: "bootstrap-bottom-8x8", file: "04_bootstrap_bottom_8x8.png", matrix: resampleClear(BOTTOM_6, 8)},
  ];
  specs.forEach(({matrix, name}) => assertMatrix(matrix, name));

  const presentation = Presentation.create({slideSize: {width: 600, height: 600}});
  for (const spec of specs) {
    const slide = presentation.slides.add();
    slide.background.fill = WHITE;
    addMatrix(slide, spec.name, spec.matrix, 30, 30, 540);
    slide.speakerNotes.textFrame.setText(
      "[Sources]\n- Derived from the user-approved Bootstrap/PERMANOVA figure.\n- Native editable cells; pure-white diagonal and exact mirror symmetry preserved.",
    );
    await writeBlob(`${OUT_DIR}/${spec.file}`, await presentation.export({slide, format: "png", scale: 4}));
  }

  const deck = await PresentationFile.exportPptx(presentation);
  await deck.save(DECK_PATH);
  await fs.writeFile(
    `${OUT_DIR}/matrix_export_audit.json`,
    JSON.stringify({
      status: "passed",
      pngSize: "2400x2400",
      invariant: "pure-white diagonal and exact mirror symmetry",
      matrices: specs.map(({name, file, matrix}) => ({name, file, dimensions: `${matrix.length}x${matrix.length}`})),
    }, null, 2),
  );
}

main().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
