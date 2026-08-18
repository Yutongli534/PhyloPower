import fs from "node:fs/promises";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const TMP_DIR = "/Users/liyutong/Desktop/phylopower/.tmp_bootstrap_symmetric";
const FINAL_PPTX = "/Users/liyutong/Desktop/phylopower/figures/output/Bootstrap_PERMANOVA_editable_symmetric_fine_grid.pptx";
const WHITE = "#FFFFFF";

const COLOR_FAMILIES = [
  ["#D8EBF0", "#A4CED8", "#63A7B8"],
  ["#DFEDD7", "#ACD19C", "#6DA875"],
  ["#F4DDD6", "#E8AA9A", "#C97568"],
];

function symmetricMatrix(n, variant) {
  const matrix = Array.from({ length: n }, () => Array(n).fill(WHITE));
  const groupSize = Math.ceil(n / 4);
  for (let row = 0; row < n; row += 1) {
    for (let col = row + 1; col < n; col += 1) {
      const groupRow = Math.floor(row / groupSize);
      const groupCol = Math.floor(col / groupSize);
      const family = (groupRow + groupCol + variant) % COLOR_FAMILIES.length;
      const shade = (row * 5 + col * 3 + variant * 2 + Math.floor((col - row) / 2)) % 3;
      const color = COLOR_FAMILIES[family][shade];
      matrix[row][col] = color;
      matrix[col][row] = color;
    }
  }
  return matrix;
}

function assertSymmetricWhiteDiagonal(matrix, label) {
  for (let row = 0; row < matrix.length; row += 1) {
    if (matrix[row][row] !== WHITE) {
      throw new Error(`${label}: diagonal cell ${row + 1} is not white`);
    }
    for (let col = 0; col < matrix.length; col += 1) {
      if (matrix[row][col] !== matrix[col][row]) {
        throw new Error(`${label}: asymmetric cells (${row + 1}, ${col + 1})`);
      }
    }
  }
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
          fill: diagonal ? "#C9D3D8" : "#FFFFFF",
          width: diagonal ? 0.55 : 0.25,
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
    { label: "source-16x16", matrix: symmetricMatrix(16, 0), x: 125, y: 260, size: 224 },
    { label: "bootstrap-top-12x12", matrix: symmetricMatrix(12, 1), x: 600, y: 175, size: 112 },
    { label: "bootstrap-middle-12x12", matrix: symmetricMatrix(12, 2), x: 600, y: 345, size: 112 },
    { label: "bootstrap-bottom-12x12", matrix: symmetricMatrix(12, 3), x: 600, y: 515, size: 112 },
  ];

  matrices.forEach(({ matrix, label }) => assertSymmetricWhiteDiagonal(matrix, label));
  await fs.writeFile(
    `${TMP_DIR}/matrix-audit.json`,
    JSON.stringify(
      {
        status: "passed",
        invariant: "all diagonal cells are white and matrix[i][j] === matrix[j][i]",
        matrices: matrices.map(({ label, matrix }) => ({ label, size: matrix.length })),
      },
      null,
      2
    )
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

  // Arrows are created before matrix cells, so all cells remain unobstructed.
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
    "[Sources]\n- User-provided reference image: codex-clipboard-8bee9934-b9f9-4d97-8716-31d90bce09f8.png\n- Recreated as native PowerPoint shapes. All four matrices were programmatically verified to have a white main diagonal and exact mirror symmetry. Off-diagonal colors use a coordinated muted blue-green-coral palette."
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
