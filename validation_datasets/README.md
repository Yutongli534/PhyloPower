# PhyloPower validation datasets

Downloaded on 2026-07-25 from the official Bioconductor ExperimentHub and
PRIDE repositories. This directory preserves the upstream files unchanged;
conversion into PhyloPower input tables should be performed in a separate
`processed/` directory.

## Gene / taxonomic abundance datasets

### QinJ_2012 — type 2 diabetes

- ExperimentHub resource: `EH7235`
- Data: `raw/curatedMetagenomicData/QinJ_2012/2021-10-14.QinJ_2012.relative_abundance.rda`
- Intended comparison: type 2 diabetes versus control
- Source: <https://mghp.osn.xsede.org/bir190004-bucket01/ExperimentHub/curatedMetagenomicData/2021-10-14/QinJ_2012/2021-10-14.QinJ_2012.relative_abundance.rda>

### YachidaS_2019 — colorectal cancer

- ExperimentHub resource: `EH7259`
- Data: `raw/curatedMetagenomicData/YachidaS_2019/2021-10-14.YachidaS_2019.relative_abundance.rda`
- Intended comparison: colorectal cancer versus control
- Source: <https://mghp.osn.xsede.org/bir190004-bucket01/ExperimentHub/curatedMetagenomicData/2021-10-14/YachidaS_2019/2021-10-14.YachidaS_2019.relative_abundance.rda>

The abundance resources do not contain all information needed for sample
selection and tree construction. The matching Bioconductor data package is
therefore retained at
`raw/curatedMetagenomicData/_reference/curatedMetagenomicData_3.20.0.tar.gz`.
It contains `data/sampleMetadata.rda`, `R/sysdata.rda` (including the package
phylogenetic tree), citations, and the package documentation.

Package source:
<https://bioconductor.org/packages/release/data/experiment/src/contrib/curatedMetagenomicData_3.20.0.tar.gz>

## Protein / metaproteomic dataset

### PXD069517 — celiac disease with or without poly-autoimmunity

Directory: `raw/PRIDE/PXD069517/`

- `CD_28samples_3DBs_intensity_norm_MasterProteins.xlsx`: normalized
  master-protein intensities
- `MGDB_peptides_intensities_tax-funct_annotations.xlsx`: peptide
  intensities with taxonomic and functional annotations
- `Patients_metadata.xlsx`: patient metadata
- `sample_code_legend.xlsx`: sample-code lookup table

Project page: <https://www.ebi.ac.uk/pride/archive/projects/PXD069517>

Only processed quantitative and annotation files were downloaded. Instrument
RAW files and the large PSM export were intentionally excluded because they
are not required to construct the PhyloPower validation input.

## Integrity

Run the following command from this directory:

```sh
shasum -a 256 -c SHA256SUMS
```

## Prepared validation and first sensitivity result

PXD069517 has been converted without modifying the downloaded source files:

- `processed/PXD069517/protein_taxon_function.csv`
- `processed/PXD069517/group.csv`
- `processed/PXD069517/rooted-tree.nwk`
- `processed/PXD069517/taxon_mapping.csv`
- `processed/PXD069517/preparation_summary.json`

The reproducible converter is `../scripts/prepare_pxd069517.py`.

A formal pilot-size sensitivity run (100 bootstraps, 99 permutations) is
stored in `results/PXD069517_pilot_sensitivity/`. See its `README.md` for the
exact command, numerical interpretation, and the taxonomic-tree limitation.
