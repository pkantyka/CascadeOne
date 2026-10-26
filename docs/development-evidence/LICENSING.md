# Licensing policy for public/commercial V1

This document records the engineering release policy for CascadeOne V1. It is not a substitute
for legal review. The project intentionally uses a narrower data surface than technically
available so that public and commercial release does not depend on mixed, unclear or
non-redistributable benchmark terms.

## Runtime/model dependencies

- convaiinnovations/laya-typed-decisions: Apache-2.0 at the pinned model revision.
- Cloudflare/clef-flash: Apache-2.0 at the pinned model revision.
- Upstream weights are not copied into CascadeOne release artifacts; the runtime resolves the
  pinned repositories or an existing cache.

## Training data

- Primary: tasksource/procedural-typed-decisions, pinned revision
  609513a3faddf729123266efe9861456fe748f95, Apache-2.0.
- Optional: helmo/synthetic-typed-decisions, pinned revision
  1827dc0d7f5a77f172107bfa25a49d2eda50b32e, MIT.
- LocalLLaMA/typed-decisions is not an approved training source in CascadeOne. Its test split
  is protected evaluation/audit data only.

## Protected contamination corpus

- LocalLLaMA/typed-decisions, pinned revision
  d0e2f0c42fef86cc15d1688d25a19f5ba7c85b18: 400-case test split, Apache-2.0.
- JevBench, pinned repository revision
  bb05a335bc809e61b20c0f745d25499a82b326fc: only
  datasets/public/original.jsonl, exactly 72 original public decisions. JevBench explicitly
  licenses its harness and these 72 original public decisions under MIT. The production gate
  verifies the exact Git blob, row count, normalized-content fingerprint and MIT scope.

The following are deliberately excluded from the V1 production data path:

- Decision Index: aggregate dataset terms are heterogeneous and some components restrict
  redistribution or commercial use.
- Full sysone-bench manifest: harness code is MIT, but benchmark rows preserve heterogeneous
  upstream dataset terms, including entries whose license is unknown, other or not_declared.
- Other JevBench imported/easy/hard/held-out material: CascadeOne V1 does not rely on it for
  the protected training gate; the release policy uses only the top-level MIT-covered
  72-decision original public set.

These excluded benchmarks may later be run externally under their own terms. Their dataset
contents must not be packaged into the CascadeOne release by default.

## Release artifact

The project publication license is a separate decision from upstream licenses and must be
explicitly approved in release_review.json. The exporter already refuses release when that
field is absent. Before publication, verify notices/attribution obligations for every shipped
third-party code dependency and confirm that no generated data/, benchmark rows, model
weights or Hugging Face caches are included in dist/.
