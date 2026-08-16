# Zenodo upload checklist

1. Review `.zenodo.json` and `CITATION.cff`.
2. Add every software creator who should receive credit, together with ORCID
   and affiliation where available.
3. Confirm and document the accessions, citations, and redistribution terms
   for the bundled example datasets.
4. Rebuild and test:

   ```bash
   python scripts/build_standalone.py
   python -m pytest -q
   python scripts/build_zenodo_release.py
   ```

5. Create a new Zenodo upload and select resource type **Software**.
6. Upload only `release/phylopower-0.1.0.zip`.
7. Compare its SHA-256 value with
   `release/phylopower-0.1.0.zip.sha256`.
8. Reserve a DOI if the manuscript needs it before publication.
9. Add the reserved DOI to `CITATION.cff` and the manuscript, rebuild the
   archive, replace the draft file, and publish only after final review.

Zenodo records are immutable after publication; corrections require a new
version.
