# Citation verification TODO (before any submission)

Per the ml-paper-writing rule (never hand-write BibTeX), the draft's
`thebibliography` entries were assembled from the repo's own verified survey
(`docs/research/fisheye-wide-fov-adaptation.md`, each entry there checked
against its arXiv abstract page on 2026-07-29) — NOT fetched programmatically.

Required pass before submission:
1. Fetch canonical BibTeX for every entry via arXiv/CrossRef API
   (`doi_to_bibtex` flow in the skill's citation-workflow reference), replace
   `thebibliography` with a proper `refs.bib`.
2. Three entries are explicitly marked `[PLACEHOLDER — verify]` in main.tex:
   - depthfisheye (ICCVM 2025 — found via web search only, venue/authors unverified)
   - drivingdepth (arXiv:2606.31488 — found via web search only)
   - adt (need the canonical ADT reference)
3. Add: SIFT (Lowe), MAGSAC++ (Barath et al.), Kannala–Brandt model — standard
   references intentionally left out of the draft until fetched.
4. Verify each cited claim against the papers (esp. the UniK3D contraction
   sentence and RayTun3R's ablation characterization).

## 2026-08-25 verification pass (DONE for arXiv entries)

All 13 arXiv citations fetched programmatically via the arXiv API — titles,
authors, years match; `refs.bib` generated from the fetched metadata.
Remaining manual items (2): DepthFisheye (ICCVM 2025, no arXiv) and Lowe's
SIFT (CrossRef fetch) — both flagged as PLACEHOLDER entries in refs.bib.
The draft's `thebibliography` should be replaced by refs.bib at next edit.
