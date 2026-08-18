import fs from "node:fs/promises";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const TMP_DIR = "/Users/liyutong/Desktop/phylopower/.tmp_bootstrap_preserved";
const FINAL_PPTX = "/Users/liyutong/Desktop/phylopower/figures/output/Bootstrap_PERMANOVA_editable_preserved_pattern.pptx";

const COLORS = {
  N: "#E7E9E8",
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
  ["N", "B3", "B2", "B1", "G1", "G2", "G3", "G4"],
  ["B3", "N", "B3", "B2", "G2", "G3", "G4", "R3"],
  ["B2", "B3", "N", "B3", "G3", "G4", "R3", "R2"],
  ["B1", "B2", "B3", "N", "G4", "R3", "R2", "R1"],
  ["G1", "G2", "G3", "G4", "N", "B3", "B2", "B1"],
  ["G2", "G3", "G4", "R3", "B3", "N", "B3", "B2"],
  ["G3", "G4", "R3", "R2", "B2", "B3", "N", "B3"],
  ["G4", "R3", "R2", "R1", "B1", "B2", "B3", "N"],
];

const BOOTSTRAP_TOP_6 = [
  ["N", "B2", "B3", "G2", "G3", "G4"],
  ["B3", "N", "B3", "G3", "G4", "G2"],
  ["B2", "B3", "N", "G4", "G3", "B3"],
  ["G2", "G3", "G4", "N", "B2", "B3"],
  ["G3", "G4", "G3", "B2", "N", "B2"],
  ["G4", "G2", "B3", "B3", "B2", "N"],
];

const BOOTSTRAP_MIDDLE_6 = [
  ["N", "R2", "R1", "G2", "G3", "G4"],
  ["R2", "N", "R2", "G3", "G4", "G3"],
  ["R1", "R2", "N", "G4", "G3", "G2"],
  ["G2", "G3", "G4", "N", "R2", "R1"],
  ["G3", "G4", "G3", "R2", "N", "R2"],
  ["G4", "G3", "G2", "R1", "R2", "N"],
];

const BOOTSTRAP_BOTTOM_6 = [
  ["N", "B3", "B2", "R1", "R2", "R3"],
  ["B3", "N", "B3", "R2", "R3", "R2"],
  ["B2", "B3", "N", "R3", "R2", "R1"],
  ["R1", "R2", "R3", "N", "B3", "B2"],
  ["R2", "R3", "R2", "B3", "N", "B3"],
  ["R3", "R2", "R1", "B2", "B3", "N"],
];

function subdivide(matrix, factor) {
  const expanded = [];
  for (const row of matrix) {
    const wide = row.flatMap((value) => Array(factor).fill(value));
    for (let i = 0; i < factor; i += 1) expanded.push([...wide]);
  }
  return expanded;
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
  shape.text.style = {
    fontSize,
    bold,
    color,
    alignment: "center",
    fontFamily: "Arial",
  };
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
      slide.shapes.add({
        geometry: "rect",
        name: `${name}-r${row + 1}-c${col + 1}`,
        position: {
          left: left + col * cell,
          top: top + row * cell,
          width: cell + 0.1,
          height: cell + 0.1,
        },
        fill: COLORS[matrix[row][col]],
        line: { style: "solid", fill: "#FFFFFF", width: 0.35 },
      });
    }
  }
}

async function writeBlob(path, blob) {
  await fs.writeFile(path, new Uint8Array(await blob.arrayBuffer()));
}

async function main() {
  await fs.mkdir(TMP_DIR, { recursive: true });
  const presentation = Presentation.create({ slideSize: { width: 1280, height: 720 } });
  const slide = presentation.slides.add();
  const ink = "#11171A";
  slide.background.fill = "#FFFFFF";

  slide.shapes.add({
    geometry: "roundRect",
    name: "bootstrap-panel",
    position: { left: 38, top: 44, width: 1204, height: 638 },
    fill: "#EEF3FA",
    line: { style: "solid", fill: "none", width: 0 },
    borderRadius: 54,
  });

  // Arrows are behind all heatmap cells.
  addArrow(slide, "source-to-top", "rightArrow", 390, 218, 170, 28, -28, ink);
  addArrow(slide, "source-to-middle", "rightArrow", 400, 342, 145, 28, 0, ink);
  addArrow(slide, "source-to-bottom", "rightArrow", 390, 468, 170, 28, 28, ink);
  addArrow(slide, "metric-to-top", "leftArrow", 750, 218, 170, 28, -28, ink);
  addArrow(slide, "metric-to-middle", "leftArrow", 762, 342, 150, 28, 0, ink);
  addArrow(slide, "metric-to-bottom", "leftArrow", 750, 468, 170, 28, 28, ink);

  addText(
    slide,
    "figure-title",
    "Bootstrap resampling & PERMANOVA",
    145,
    65,
    990,
    66,
    44,
    ink,
    true
  );
  addText(slide, "omega-squared", "ω²", 940, 290, 170, 58, 46, ink, false);
  addText(slide, "p-value", "p value", 925, 350, 200, 58, 42, ink, false);
  addText(slide, "repeat-symbol", "↻", 100, 602, 70, 60, 50, "#39474C", false);
  addText(slide, "iteration-label", "B = 500 iterations", 160, 610, 390, 52, 38, ink, false);

  // Preserve the original 8×8 and 6×6 patterns, then split each cell 2×2.
  addMatrix(slide, "source-16x16", subdivide(SOURCE_8, 2), 125, 260, 224);
  addMatrix(slide, "bootstrap-top-12x12", subdivide(BOOTSTRAP_TOP_6, 2), 600, 175, 112);
  addMatrix(slide, "bootstrap-middle-12x12", subdivide(BOOTSTRAP_MIDDLE_6, 2), 600, 345, 112);
  addMatrix(slide, "bootstrap-bottom-12x12", subdivide(BOOTSTRAP_BOTTOM_6, 2), 600, 515, 112);

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
    "[Sources]\n- User-provided reference image: codex-clipboard-8bee9934-b9f9-4d97-8716-31d90bce09f8.png\n- Original palette and diagonal matrix structure preserved. Each original source cell was subdivided 2×2, yielding 16×16 and 12×12 native editable PowerPoint grids."
  );

  await writeBlob(
    `${TMP_DIR}/slide-01.png`,
    await presentation.export({ slide, format: "png", scale: 2 })
  );
  await fs.writeFile(
    `${TMP_DIR}/slide-01.layout.json`,
    await (await slide.export({ format: "layout" })).text()
  );
  const inspect = await presentation.inspect({
    kind: "slide,shape,textbox,notes,image",
    maxChars: 50000,
  });
  await fs.writeFile(`${TMP_DIR}/inspect.ndjson`, inspect.ndjson);

  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(FINAL_PPTX);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
