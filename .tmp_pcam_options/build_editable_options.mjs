import fs from "node:fs/promises";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const OUT_DIR = "/Users/liyutong/Desktop/phylopower/.tmp_pcam_options";
const FINAL_PPTX = "/Users/liyutong/Desktop/phylopower/figures/output/PCAM_editable_color_options.pptx";

const schemes = [
  {
    label: "OPTION A · MUTED BOTANICAL",
    background: "#FFFFFF",
    panel: "#EDF6F3",
    ink: "#25383D",
    node: "#7B888C",
    colors: ["#4F8D73", "#4B77AB", "#8B6C91", "#B85C58", "#C07C48", "#BEA64F"],
  },
  {
    label: "OPTION B · COOL SCIENTIFIC",
    background: "#FFFFFF",
    panel: "#EEF4F7",
    ink: "#243746",
    node: "#788893",
    colors: ["#2A7F78", "#3E8794", "#4776A6", "#6178A8", "#756FA0", "#8A6F91"],
  },
  {
    label: "OPTION C · COLOR-BLIND SAFE",
    background: "#FFFFFF",
    panel: "#F4F5F2",
    ink: "#2B3437",
    node: "#7D8587",
    colors: ["#009E73", "#0072B2", "#56B4E9", "#D55E00", "#E69F00", "#CC79A7"],
  },
  {
    label: "OPTION D · DUSTY PASTELS",
    background: "#FFFFFF",
    panel: "#F6F2F0",
    ink: "#3B393A",
    node: "#8A8585",
    colors: ["#789F8C", "#7F9DB6", "#A28FA8", "#C18A88", "#CEA078", "#C6B377"],
  },
  {
    label: "OPTION E · NEUTRAL + ACCENT",
    background: "#FFFFFF",
    panel: "#F2F3F2",
    ink: "#30383A",
    node: "#7E8586",
    colors: ["#4F817A", "#6B8792", "#858B91", "#9A7C7C", "#A08368", "#A2966C"],
  },
];

function hexToRgb(hex) {
  const value = hex.replace("#", "");
  return {
    r: parseInt(value.slice(0, 2), 16),
    g: parseInt(value.slice(2, 4), 16),
    b: parseInt(value.slice(4, 6), 16),
  };
}

function rgbToHex({ r, g, b }) {
  return `#${[r, g, b]
    .map((v) => Math.max(0, Math.min(255, Math.round(v))).toString(16).padStart(2, "0"))
    .join("")}`;
}

function tint(hex, whiteFraction) {
  const c = hexToRgb(hex);
  return rgbToHex({
    r: c.r + (255 - c.r) * whiteFraction,
    g: c.g + (255 - c.g) * whiteFraction,
    b: c.b + (255 - c.b) * whiteFraction,
  });
}

function addLine(slide, name, x1, y1, x2, y2, color, width = 4) {
  return slide.shapes.add({
    geometry: "line",
    name,
    position: { left: x1, top: y1, width: x2 - x1, height: y2 - y1 },
    fill: "none",
    line: { style: "solid", fill: color, width },
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
  shape.text.style = {
    fontSize,
    bold,
    color,
    alignment: "center",
    fontFamily: "Arial",
  };
  return shape;
}

function addCell(slide, name, left, top, width, height, fill, ink) {
  return slide.shapes.add({
    geometry: "rect",
    name,
    position: { left, top, width, height },
    fill,
    line: { style: "solid", fill: ink, width: 2 },
  });
}

function buildDiagram(slide, scheme, optionIndex) {
  slide.background.fill = scheme.background;

  slide.shapes.add({
    geometry: "roundRect",
    name: `option-${optionIndex}-panel`,
    position: { left: 45, top: 92, width: 1190, height: 572 },
    fill: scheme.panel,
    line: { style: "solid", fill: "none", width: 0 },
    borderRadius: 48,
  });

  // Connectors and tree branches are intentionally created before nodes.
  const branch = scheme.ink;
  addLine(slide, `option-${optionIndex}-root-left`, 70, 359, 112, 359, branch);
  addLine(slide, `option-${optionIndex}-root-vertical`, 112, 235, 112, 461, branch);
  addLine(slide, `option-${optionIndex}-top-clade-link`, 112, 277, 155, 277, branch);
  addLine(slide, `option-${optionIndex}-bottom-clade-link`, 112, 453, 155, 453, branch);

  addLine(slide, `option-${optionIndex}-top-clade-vertical`, 155, 211, 155, 295, branch);
  addLine(slide, `option-${optionIndex}-top-subclade-link`, 155, 232, 212, 232, branch);
  addLine(slide, `option-${optionIndex}-top-subclade-vertical`, 212, 211, 212, 253, branch);
  addLine(slide, `option-${optionIndex}-top-leaf-1`, 212, 211, 314, 211, branch);
  addLine(slide, `option-${optionIndex}-top-leaf-2`, 212, 253, 314, 253, branch);
  addLine(slide, `option-${optionIndex}-top-leaf-3`, 155, 295, 314, 295, branch);

  addLine(slide, `option-${optionIndex}-bottom-clade-vertical`, 155, 411, 155, 495, branch);
  addLine(slide, `option-${optionIndex}-bottom-subclade-link`, 155, 432, 212, 432, branch);
  addLine(slide, `option-${optionIndex}-bottom-subclade-vertical`, 212, 411, 212, 453, branch);
  addLine(slide, `option-${optionIndex}-bottom-leaf-1`, 212, 411, 314, 411, branch);
  addLine(slide, `option-${optionIndex}-bottom-leaf-2`, 212, 453, 314, 453, branch);
  addLine(slide, `option-${optionIndex}-bottom-leaf-3`, 155, 495, 314, 495, branch);

  addText(slide, `option-${optionIndex}-label`, scheme.label, 54, 34, 520, 34, 22, scheme.ink, true);
  addText(slide, `option-${optionIndex}-matrix-title`, "Taxon × Sample", 292, 124, 310, 50, 35, scheme.ink, true);
  addText(slide, `option-${optionIndex}-pcam-title`, "PCAM", 545, 584, 210, 58, 42, scheme.ink, false);

  const grid = scheme.colors.map((base) => [0.82, 0.58, 0.32, 0.08].map((w) => tint(base, w)));
  const cellW = 42;
  const cellH = 42;
  const matrixX = 345;
  const topY = 190;
  const bottomY = 390;

  for (let row = 0; row < 6; row += 1) {
    const y = row < 3 ? topY + row * cellH : bottomY + (row - 3) * cellH;
    for (let col = 0; col < 4; col += 1) {
      addCell(
        slide,
        `option-${optionIndex}-matrix-r${row + 1}-c${col + 1}`,
        matrixX + col * cellW,
        y,
        cellW,
        cellH,
        grid[row][col],
        scheme.ink
      );
    }
  }

  const leafYs = [211, 253, 295, 411, 453, 495];
  for (let row = 0; row < 6; row += 1) {
    slide.shapes.add({
      geometry: "ellipse",
      name: `option-${optionIndex}-taxon-marker-${row + 1}`,
      position: { left: 302, top: leafYs[row] - 12, width: 24, height: 24 },
      fill: tint(scheme.colors[row], 0.28),
      line: { style: "solid", fill: "#58676B", width: 2 },
    });
  }

  const internalNodes = [
    [102, 349],
    [145, 267],
    [202, 222],
    [145, 443],
    [202, 422],
  ];
  internalNodes.forEach(([x, y], i) => {
    slide.shapes.add({
      geometry: "ellipse",
      name: `option-${optionIndex}-internal-node-${i + 1}`,
      position: { left: x, top: y, width: 20, height: 20 },
      fill: scheme.node,
      line: { style: "solid", fill: "none", width: 0 },
    });
  });

  slide.shapes.add({
    geometry: "rightArrow",
    name: `option-${optionIndex}-transformation-arrow`,
    position: { left: 590, top: 328, width: 112, height: 48 },
    fill: scheme.ink,
    line: { style: "solid", fill: "none", width: 0 },
  });

  const stackXs = [760, 860, 960, 1060];
  const stackLabels = ["s2+s3", "s1+s4", "s1+s2", "s1+s4"];
  const stackColorColumns = [
    [2, 2, 2, 2, 2, 2],
    [3, 3, 3, 0, 0, 0],
    [0, 0, 0, 2, 1, 2],
    [0, 0, 0, 3, 3, 3],
  ];
  for (let stack = 0; stack < 4; stack += 1) {
    addText(
      slide,
      `option-${optionIndex}-stack-label-${stack + 1}`,
      stackLabels[stack],
      stackXs[stack] - 24,
      168,
      96,
      34,
      23,
      scheme.ink,
      true
    );
    for (let row = 0; row < 6; row += 1) {
      addCell(
        slide,
        `option-${optionIndex}-stack-${stack + 1}-segment-${row + 1}`,
        stackXs[stack],
        210 + row * 48,
        48,
        48,
        grid[row][stackColorColumns[stack][row]],
        scheme.ink
      );
    }
  }

  [1160, 1184, 1208].forEach((x, i) => {
    slide.shapes.add({
      geometry: "ellipse",
      name: `option-${optionIndex}-ellipsis-${i + 1}`,
      position: { left: x, top: 348, width: 10, height: 10 },
      fill: scheme.ink,
      line: { style: "solid", fill: "none", width: 0 },
    });
  });

  slide.speakerNotes.textFrame.setText(
    `[Sources]\n- User-provided reference image: codex-clipboard-27787804-124a-4d08-8c8b-98fd202c07d8.png\n- Recreated as native PowerPoint shapes. Every matrix cell, stacked segment, marker, branch, label, and arrow is independently editable.\n- Palette: ${scheme.label}`
  );
}

async function writeBlob(path, blob) {
  await fs.writeFile(path, new Uint8Array(await blob.arrayBuffer()));
}

async function main() {
  await fs.mkdir(OUT_DIR, { recursive: true });
  const presentation = Presentation.create({ slideSize: { width: 1280, height: 720 } });

  schemes.forEach((scheme, index) => {
    const slide = presentation.slides.add();
    buildDiagram(slide, scheme, index + 1);
  });

  for (const [index, slide] of presentation.slides.items.entries()) {
    const stem = `slide-${String(index + 1).padStart(2, "0")}`;
    await writeBlob(
      `${OUT_DIR}/${stem}.png`,
      await presentation.export({ slide, format: "png", scale: 2 })
    );
    await fs.writeFile(
      `${OUT_DIR}/${stem}.layout.json`,
      await (await slide.export({ format: "layout" })).text()
    );
  }

  await writeBlob(
    `${OUT_DIR}/montage.webp`,
    await presentation.export({ format: "webp", montage: true, scale: 1 })
  );

  const inspect = await presentation.inspect({
    kind: "slide,shape,textbox,notes,image",
    maxChars: 30000,
  });
  await fs.writeFile(`${OUT_DIR}/inspect.ndjson`, inspect.ndjson);

  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(FINAL_PPTX);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
