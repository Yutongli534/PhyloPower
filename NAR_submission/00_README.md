# PhyloPower NAR 投稿文件包

投稿入口：https://mc.manuscriptcentral.com/nar （选 **Methods** 类型）

## 上传顺序与文件对应

| 步骤 | 系统要求 | 用哪个文件 |
|---|---|---|
| 1 | 主文稿（单一 PDF，图嵌入文内、图注在图下方） | `01_manuscript/PhyloPower727-LM.docx` → **在 Word 里红字转黑后另存为 PDF 上传** |
| 2 | 图文摘要（Graphical Abstract，单独文件） | `03_graphical_abstract/graphical_abstract.tiff`（600dpi，5:2 横版） |
| 3 | 高分辨率图文件（Fig 1–8，单独上传，tif/eps/png/可编辑 PDF） | `02_figures/Figure1_workflow.png`、`Figure2.pdf` … `Figure8.pdf`（PNG 均 ≥300dpi，PDF 为矢量） |
| 4a | 补充材料-合并 PDF（Supplementary data - for compiled PDF） | `04_supplementary/Supplementary_Figures.pdf`（S1–S3，共 3 页） |
| 4b | 补充材料-帮助文档（同 4a 类型） | `04_supplementary/Supplementary_Help_Document.pdf`（教程，共 4 页） |
| 4c | 补充示例数据文件（Supplementary data - NOT for compiled PDF） | `04_supplementary/example_data_files/` 下两个子目录（taxonomic_abundance 与 taxon_function，共 7 个文件，可分别上传或打包 zip 传一个） |
| 5 | Cover letter | `05_cover_letter/cover_letter_draft.txt`（填通讯作者署名；AI 披露句去留自定） |
| 6 | Key Points（3 条，每条 ≤100 字符） | 见本文件下方 |
| 7 | 系统内填写 | 全部作者 + 单位 + 邮箱 + 全员 ORCID、Funding 机构与基金号、推荐/回避审稿人 3–5 人 |

## Key Points（粘贴用）

1. First power-analysis tool covering both taxonomic and taxon-function coupled multi-omics data.
2. First beta-diversity power tool integrating phylogenetic-tree algorithms (Gemelli and PhyloFunc).
3. Lets researchers plan sample size for any target effect size, not just the pilot's effect.

## 提交前最后检查

- [ ] Word 里绿底占位全部填完（作者名单"…"、Funding、致谢、作者贡献、两个数据集 accession、COI 确认）
- [ ] 红字全部转黑（全选 → 字体颜色 → 自动）
- [ ] 通读一遍新 Fig 1 和 2.7/3.8/声明区
- [ ] 另存 PDF 后抽查几页确认图和页码正常
- [ ] 文稿 DOI（10.5281/zenodo.21991013）与 Zenodo 记录页一致

## 系统文件类型对照

- Manuscript file - clean → 主文稿 PDF（初投不需要 marked revisions）
- Figure → 02_figures/ 下 8 张图，每张单独传一次
- Graphical Abstract → 03 目录 tiff
- Supplementary data - for compiled PDF → Supplementary_Figures.pdf + Supplementary_Help_Document.pdf
- Supplementary data - NOT for compiled PDF → example_data_files/（zip 或逐个传）
- Other / Cover Letter → 05 目录（或直接粘贴到文本框）
- Response to Referees / marked revisions / TeX Suppl → 初投均不需要

## 备注

- 主文稿 PDF 必须由 Word 导出（本机无 LibreOffice，无法代为转换）；docx 里已嵌入全部图，图注在图下方。
- 高分辨率图文件名已按 Figure1–Figure8 规范命名，投稿系统里文件类型选 "Figure"。
- 补充材料文件名建议改为 `Supplementary_Figures.pdf`（已在用），系统里文件类型选 "Supplementary Data"。
