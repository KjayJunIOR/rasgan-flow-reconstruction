# Public release checklist

## Rights and provenance

- [x] Confirm all relevant authors/collaborators approve public release.
- [x] Confirm university, employer, sponsor, and dataset obligations.
- [x] Compare the present codebase against the linked TensorLayer/SRGAN `master`
      and document retained lineage.
- [x] Use academic/non-commercial source-available terms consistent with the
      linked upstream restriction and retain explicit TensorLayer/SRGAN attribution.
- [x] If the exact historical TensorLayer/SRGAN starting commit is recovered,
      add its commit hash to `docs/PROVENANCE.md` for archival precision.
- [x] Verify no newly added third-party source is distributed without recording
      its license, notices, and provenance.

## Scientific and technical review

Development validation evidence is summarized in `docs/VALIDATION.md`.

- [ ] Run `pytest` in the target TensorFlow environment.
- [ ] Run `rasgan-selfcheck` on CPU and the intended GPU.
- [ ] Run the synthetic one-epoch content smoke test.
- [ ] Confirm a production checkpoint can be loaded by `rasgan-infer`.
- [ ] Record the exact Python/TensorFlow/CUDA/cuDNN/GPU environment.

## Portfolio presentation

- [ ] Add the final GitHub repository URLs to `pyproject.toml`.
- [ ] Decide whether the unpublished PDF and preliminary result images may be
      public; remove them if not.
- [ ] Update `CITATION.cff` with repository URL, release date, and DOI if any.
- [ ] Tag `v0.1.0` only after the clean-install commands succeed.
