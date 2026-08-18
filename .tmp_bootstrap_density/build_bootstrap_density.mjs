import fs from "node:fs/promises";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const TMP_DIR = "/Users/liyutong/Desktop/phylopower/.tmp_bootstrap_density";
const FINAL_PPTX = "/Users/liyutong/Desktop/phylopower/figures/output/Bootstrap_PERMANOVA_editable_density_options.pptx";

const options = [
  { label: "OPTION A · 10×10 → 7×7", sourceN: 10, sampleN: 7 },
  { label: "OPTION B · 12×12 → 8×8", sourceN: 12, sampleN: 8 },
  { label: "OPTION C · 14×14 → 10×10", sourceN: 14, sampleN: 10 },
];

const PALETTE = [
  "#EDF1F0",
  "#C9E2EA",
  "#8CC3D3",
  "#4F97AE",
  "#9CCB8E",
  "#5E9C68",
  "#D99A88",
  "#B86456",
];

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

function valueAt(row, col, n) {
  const wave =
    1.10 * Math.sin((row + 1) * 0.72) +
    0.95 * Math.cos((col + 2) * 0.61) +
    0.80 * Math.sin((row - col) * 0.52) +
    0.45 * Math.cos((row + col) * 0.37);
  const normalized = Math.max(0, Math.min(0.999, (wave + 3.3) / 6.6));
  let index = Math.floor(normalized * PALETTE.length);
  if ((row + col + n) % 11 === 0) index = 0;
  return index;
}

function sourceMatrix(n) {
  return Array.from({ length: n }, (_, row) =>
    Array.from({ length: n }, (_, col) => valueAt(row, col, n))
  );
}

function resample(source, sampleN, variant) {
  const n = source.length;
  const rows = Array.from(
    { length: sampleN },
    (_, i) => (i * (variant + 2) + variant * 3 + Math.floor(i / 2)) % n
  );
  const cols = Array.from(
    { length: sampleN },
    (_, i) => (i * (variant + 3) + variant + Math.floor(i / 3)) % n
  );
  return rows.map((r) => cols.map((c) => source[r][c]));
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
          width: cell + 0.15,
          height: cell + 0.15,
        },
        fill: PALETTE[matrix[row][col]],
        line: { style: "solid", fill: "none", width: 0 },
      });
    }
  }
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

function buildSlide(slide, option, index) {
  const ink = "#25343A";
  slide.background.fill = "#FFFFFF";
  slide.shapes.add({
    geometry: "roundRect",
    name: `density-${index}-panel`,
    position: { left: 42, top: 88, width: 1196, height: 590 },
    fill: "#F0F4F8",
    line: { style: "solid", fill: "none", width: 0 },
    borderRadius: 52,
  });

  // Directional arrows are created before matrices and labels.
  addArrow(slide, `density-${index}-source-to-top`, "rightArrow", 385, 230, 170, 28, -28, ink);
  addArrow(slide, `density-${index}-source-to-middle`, "rightArrow", 394, 346, 145, 28, 0, ink);
  addArrow(slide, `density-${index}-source-to-bottom`, "rightArrow", 385, 466, 170, 28, 28, ink);
  addArrow(slide, `density-${index}-metric-to-top`, "leftArrow", 750, 230, 170, 28, -28, ink);
  addArrow(slide, `density-${index}-metric-to-middle`, "leftArrow", 762, 346, 150, 28, 0, ink);
  addArrow(slide, `density-${index}-metric-to-bottom`, "leftArrow", 750, 466, 170, 28, 28, ink);

  addText(slide, `density-${index}-option-label`, option.label, 54, 34, 500, 34, 22, ink, true);
  addText(
    slide,
    `density-${index}-title`,
    "Bootstrap resampling & PERMANOVA",
    180,
    108,
    920,
    58,
    42,
    "#11181B",
    true
  );
  addText(slide, `density-${index}-omega2`, "ω²", 930, 292, 180, 54, 44, "#11181B", false);
  addText(slide, `density-${index}-pvalue`, "p value", 920, 348, 200, 54, 42, "#11181B", false);
  addText(slide, `density-${index}-repeat-icon`, "↻", 94, 600, 70, 60, 50, "#39474C", false);
  addText(slide, `density-${index}-iterations`, "B = 500 iterations", 150, 607, 390, 52, 38, "#11181B", false);

  const source = sourceMatrix(option.sourceN);
  addMatrix(slide, `density-${index}-source`, source, 125, 255, 225);

  const centerX = 600;
  const centerSize = 112;
  const centerYs = [190, 350, 510];
  for (let variant = 0; variant < 3; variant += 1) {
    addMatrix(
      slide,
      `density-${index}-bootstrap-${variant + 1}`,
      resample(source, option.sampleN, variant + 1),
      centerX,
      centerYs[variant],
      centerSize
    );
  }

  [470, 486, 502].forEach((y, i) => {
    slide.shapes.add({
      geometry: "ellipse",
      name: `density-${index}-ellipsis-${i + 1}`,
      position: { left: 650, top: y, width: 8, height: 8 },
      fill: ink,
      line: { style: "solid", fill: "none", width: 0 },
    });
  });

  slide.speakerNotes.textFrame.setText(
    `[Sources]\n- User-provided reference image: codex-clipboard-8bee9934-b9f9-4d97-8716-31d90bce09f8.png\n- Recreated as native PowerPoint shapes. Every heatmap cell, arrow, label, and symbol is independently editable.\n- Density option: ${option.label}`
  );
}

async function writeBlob(path, blob) {
  await fs.writeFile(path, new Uint8Array(await blob.arrayBuffer()));
}

async function main() {
  await fs.mkdir(TMP_DIR, { recursive: true });
  const presentation = Presentation.create({ slideSize: { width: 1280, height: 720 } });

  options.forEach((option, index) => {
    const slide = presentation.slides.add();
    buildSlide(slide, option, index + 1);
  });

  for (const [index, slide] of presentation.slides.items.entries()) {
    const stem = `slide-${String(index + 1).padStart(2, "0")}`;
    await writeBlob(
      `${TMP_DIR}/${stem}.png`,
      await presentation.export({ slide, format: "png", scale: 2 })
    );
    await fs.writeFile(
      `${TMP_DIR}/${stem}.layout.json`,
      await (await slide.export({ format: "layout" })).text()
    );
  }

  await writeBlob(
    `${TMP_DIR}/montage.webp`,
    await presentation.export({ format: "webp", montage: true, scale: 1 })
  );

  const inspect = await presentation.inspect({
    kind: "slide,shape,textbox,notes,image",
    maxChars: 40000,
  });
  await fs.writeFile(`${TMP_DIR}/inspect.ndjson`, inspect.ndjson);

  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(FINAL_PPTX);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
