import fs from "node:fs/promises";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const SVG_PATH = "/Users/liyutong/Desktop/phylopower/figures/output/PCAM_recolored.svg";
const PPTX_PATH = "/Users/liyutong/Desktop/phylopower/figures/output/PCAM_recolored_editable.pptx";
const PREVIEW_PATH = "/Users/liyutong/Desktop/phylopower/.tmp_pcam_ppt/slide-1.png";
const LAYOUT_PATH = "/Users/liyutong/Desktop/phylopower/.tmp_pcam_ppt/slide-1.layout.json";

async function writeBlob(path, blob) {
  await fs.writeFile(path, new Uint8Array(await blob.arrayBuffer()));
}

async function main() {
  const presentation = Presentation.create({
    slideSize: { width: 1280, height: 720 },
  });

  const slide = presentation.slides.add();
  slide.background.fill = "#FFFFFF";

  const svgBytes = await fs.readFile(SVG_PATH);
  slide.images.add({
    name: "PCAM-editable-SVG",
    blob: new Uint8Array(svgBytes),
    contentType: "image/svg+xml",
    alt: "PCAM transforms a phylogenetic taxon-by-sample matrix into stacked synthetic-community profiles.",
    fit: "contain",
    position: { left: 40, top: 90, width: 1200, height: 500 },
  });

  slide.speakerNotes.textFrame.setText(
    "[Sources]\n- User-provided reference image: codex-clipboard-27787804-124a-4d08-8c8b-98fd202c07d8.png\n- Diagram redrawn as an original SVG with a muted, color-accessible palette."
  );

  await writeBlob(
    PREVIEW_PATH,
    await presentation.export({ slide, format: "png", scale: 2 })
  );
  await fs.writeFile(LAYOUT_PATH, await (await slide.export({ format: "layout" })).text());

  const snapshot = await presentation.inspect({
    kind: "slide,image,notes",
    maxChars: 6000,
  });
  await fs.writeFile(
    "/Users/liyutong/Desktop/phylopower/.tmp_pcam_ppt/inspect.ndjson",
    snapshot.ndjson
  );

  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(PPTX_PATH);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
