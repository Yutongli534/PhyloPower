import fs from "node:fs/promises";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const TMP_DIR = "/Users/liyutong/Desktop/phylopower/.tmp_bootstrap_clear";
const FINAL_PPTX = "/Users/liyutong/Desktop/phylopower/figures/output/Bootstrap_PERMANOVA_editable_clear_grid.pptx";
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
      if (row === col) {
        matrix[row][col] = WHITE;
        continue;
      }
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
  matrix.forEach((row,r) => {
    if (row[r] !== WHITE) throw new Error(`${label}: non-white diagonal`);
    row.forEach((v,c) => { if (v !== matrix[c][r]) throw new Error(`${label}: asymmetric`); });
  });
}

function addText(slide,name,text,left,top,width,height,fontSize,color,bold=false) {
  const s = slide.shapes.add({geometry:"textbox",name,position:{left,top,width,height},fill:"none",line:{style:"solid",fill:"none",width:0}});
  s.text = text;
  s.text.style = {fontSize,bold,color,alignment:"center",fontFamily:"Arial"};
}

function addArrow(slide,name,geometry,left,top,width,height,rotation,color) {
  slide.shapes.add({geometry,name,position:{left,top,width,height,rotation},fill:color,line:{style:"solid",fill:"none",width:0}});
}

function addMatrix(slide,name,matrix,left,top,size) {
  const n = matrix.length;
  const cell = size / n;
  for (let r=0;r<n;r+=1) for (let c=0;c<n;c+=1) {
    const diagonal = r===c;
    slide.shapes.add({
      geometry:"rect",
      name:`${name}-r${r+1}-c${c+1}`,
      position:{left:left+c*cell,top:top+r*cell,width:cell+0.05,height:cell+0.05},
      fill:matrix[r][c],
      line:{style:"solid",fill:diagonal?"#B8C4C9":"#F8FAFB",width:diagonal?0.65:0.38},
    });
  }
}

async function writeBlob(path,blob) { await fs.writeFile(path,new Uint8Array(await blob.arrayBuffer())); }

async function main() {
  await fs.mkdir(TMP_DIR,{recursive:true});
  const matrices = [
    {label:"source-12x12",matrix:resampleClear(SOURCE_8,12),x:125,y:260,size:224},
    {label:"bootstrap-top-8x8",matrix:resampleClear(TOP_6,8),x:600,y:175,size:112},
    {label:"bootstrap-middle-8x8",matrix:resampleClear(MIDDLE_6,8),x:600,y:345,size:112},
    {label:"bootstrap-bottom-8x8",matrix:resampleClear(BOTTOM_6,8),x:600,y:515,size:112},
  ];
  matrices.forEach(({matrix,label}) => assertMatrix(matrix,label));
  await fs.writeFile(`${TMP_DIR}/matrix-audit.json`,JSON.stringify({status:"passed",invariant:"white diagonal and mirror symmetry",matrices:matrices.map(({label,matrix})=>({label,size:matrix.length}))},null,2));

  const p = Presentation.create({slideSize:{width:1280,height:720}});
  const slide = p.slides.add();
  const ink = "#182126";
  slide.background.fill = "#FFFFFF";
  slide.shapes.add({geometry:"roundRect",name:"panel",position:{left:38,top:44,width:1204,height:638},fill:"#EEF3F8",line:{style:"solid",fill:"none",width:0},borderRadius:54});
  addArrow(slide,"source-to-top","rightArrow",390,218,170,28,-28,ink);
  addArrow(slide,"source-to-middle","rightArrow",400,342,145,28,0,ink);
  addArrow(slide,"source-to-bottom","rightArrow",390,468,170,28,28,ink);
  addArrow(slide,"metric-to-top","leftArrow",750,218,170,28,-28,ink);
  addArrow(slide,"metric-to-middle","leftArrow",762,342,150,28,0,ink);
  addArrow(slide,"metric-to-bottom","leftArrow",750,468,170,28,28,ink);
  addText(slide,"title","Bootstrap resampling & PERMANOVA",145,65,990,66,44,ink,true);
  addText(slide,"omega","ω²",940,290,170,58,46,ink,false);
  addText(slide,"pvalue","p value",925,350,200,58,42,ink,false);
  addText(slide,"repeat","↻",100,602,70,60,50,"#39474C",false);
  addText(slide,"iterations","B = 500 iterations",160,610,390,52,38,ink,false);
  matrices.forEach(({label,matrix,x,y,size}) => addMatrix(slide,label,matrix,x,y,size));
  [472,488,504].forEach((y,i)=>slide.shapes.add({geometry:"ellipse",name:`ellipsis-${i+1}`,position:{left:652,top:y,width:8,height:8},fill:ink,line:{style:"solid",fill:"none",width:0}}));
  slide.speakerNotes.textFrame.setText("[Sources]\n- User-provided reference image: codex-clipboard-8bee9934-b9f9-4d97-8716-31d90bce09f8.png\n- Revised to moderate-density native grids with discrete color steps, clear cell borders, a pure-white diagonal, and exact mirror symmetry.");
  await writeBlob(`${TMP_DIR}/slide-01.png`,await p.export({slide,format:"png",scale:2}));
  await fs.writeFile(`${TMP_DIR}/slide-01.layout.json`,await(await slide.export({format:"layout"})).text());
  const inspect = await p.inspect({kind:"slide,shape,textbox,notes,image",maxChars:40000});
  await fs.writeFile(`${TMP_DIR}/inspect.ndjson`,inspect.ndjson);
  const pptx = await PresentationFile.exportPptx(p);
  await pptx.save(FINAL_PPTX);
}

main().catch(e=>{console.error(e);process.exitCode=1;});
