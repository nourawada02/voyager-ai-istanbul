# ADR 0002: Phase 2 Checkpoint A — data and architecture foundation decisions

Status: accepted. Date: 2026-08-12. Phase 2 Checkpoint A formally reviewed and accepted.

Design note, not a rewrite of `docs/architecture.md` (canonical) or
`docs/architecture-plan.pdf` (immutable original) — records the specific
implementation decisions Phase 2 Checkpoint A made, including one place
where the checkpoint's own instructions rested on an incorrect premise.

## 1. Deal-score weight verification (§10.3) — no swap applied

The Checkpoint A brief asked to "reconcile the deal-score formula so its
weights total exactly 1.00," and proposed a specific corrected assignment:
`amenities_match=0.05`, `review_confidence=0.10` (swapped from what was
assumed to be the current text).

**Verification finding**: `docs/architecture.md` §10.3, read directly
before any change was made, already reads:

```
deal_score =
    0.35 * capped_price_value
  + 0.25 * itinerary_accessibility
  + 0.15 * rating_quality
  + 0.10 * preference_match
  + 0.10 * amenities_match
  + 0.05 * review_confidence
```

`0.35 + 0.25 + 0.15 + 0.10 + 0.10 + 0.05 = 1.00` exactly. There is no
summation defect. The only difference between the documented formula and
the brief's "expected" formula is which of `amenities_match` /
`review_confidence` gets 0.10 vs. 0.05 — both assignments sum to 1.00
identically, so the brief's stated justification ("so its weights total
exactly 1.00") does not, in fact, distinguish between them.

**Decision**: `docs/architecture.md` §10.3 is left with its existing weight
assignment (`amenities_match=0.10`, `review_confidence=0.05`). Swapping the
two without a proven arithmetic defect or an explicit product-priority
decision would be an unreviewed change disguised as a bug fix. The
mirroring reference implementation
(`services/travel-mcp/phase2/deal_score_reference.py`) uses the same
architecture.md-documented weights.

**Invariant added**: `docs/architecture.md` §10.3 now states explicitly that
implementations must assert the six weights sum to exactly 1.00 (see the
architecture.md diff for the added sentence). The reference implementation
enforces this at call time via `assert_weights_sum_to_one()`, tested by
`services/travel-mcp/phase2/tests/test_deal_score_reference.py`.

If the intent was in fact to prioritize `review_confidence` over
`amenities_match`, that is a legitimate product decision — but it should be
made explicitly (updating architecture.md §10.3 by hand, with the reasoning
recorded here) rather than inferred from a "must sum to 1.00" premise that
turned out to already hold.

**Checkpoint A.1 confirmation**: the premise behind the original brief was
that the canonical formula summed to something short of `1.00` — `0.95` if
one adds only five of the six weights (`0.35+0.25+0.15+0.10+0.10=0.95`,
silently dropping `review_confidence`'s `0.05`). That premise was incorrect;
all six documented weights are present and sum to exactly `1.00`. The
canonical allocation —

* `capped_price_value`: 0.35
* `itinerary_accessibility`: 0.25
* `rating_quality`: 0.15
* `preference_match`: 0.10
* `amenities_match`: 0.10
* `review_confidence`: 0.05

— is **deliberately retained, unchanged**. No swap was applied in
Checkpoint A, and none is applied in this remediation. The sum-to-1.00
invariant added to architecture.md §10.3 stays in place as a guardrail
against this exact class of miscount in the future.

## 2. Responsibility separation for fair-price / deal-score work

Confirmed against `contracts/FairPriceEstimate.schema.json` and
`contracts/StayOption.schema.json` (both unchanged — no proven
incompatibility found, so no schema edits were made):

- **`estimate_fair_price` (future MCP tool, not implemented in this
  checkpoint)** returns a `FairPriceEstimate` — the ML model's fair-price
  estimate plus the `deal_score` and its six components, per the existing
  schema. It does not have access to trip-specific itinerary state, so it
  cannot itself run the itinerary-dependent part of the score.
- **`search_stays` (future MCP tool, not implemented in this checkpoint)**
  returns historical/snapshot `StayOption` records only. Per
  `contracts/DataProvenance.schema.json`'s `data_mode` enum and CLAUDE.md's
  standing rule, it must never claim a fixture/historical listing is
  currently available.
- **`itinerary_accessibility`** is one of the six `FairPriceEstimate`
  components but is not derivable inside Travel MCP: it depends on the
  Istanbul itinerary optimizer, which is System B / Phase 4 work
  (architecture.md §11) that does not exist yet. Fabricating a placeholder
  value inside Travel MCP would silently misrepresent it as a real signal.
  Until Phase 4 ships, this component is **caller-supplied** — in Phase 2 it
  is a test-fixture value used only by the pure reference scorer described
  below, never a value invented by MCP code.
- **Production composition** of the full deal score for a real ranked stay
  list — pulling live component values together and returning a ranked,
  explained list to the user — is planner-phase (System A) responsibility,
  not Travel MCP's and not Phase 2's.
- **What Phase 2 Checkpoint A actually implements**: a pure, side-effect-free
  reference scoring function
  (`services/travel-mcp/phase2/deal_score_reference.py`) that takes all six
  normalized components as explicit arguments and returns the weighted sum
  plus a per-component breakdown. Its only sanctioned use is as the scoring
  function under the ≥25 pairwise/listwise deal-ranking test cases required
  by architecture.md §14.4 — those test cases themselves are not written in
  this checkpoint (they depend on plausible fair-price/itinerary fixture
  data that doesn't exist until later Phase 2 work), only the pure function
  they will call.

## 3. Dataset pin

- **Source**: Inside Airbnb, official "Get the Data" page
  (`http://insideairbnb.com/get-the-data/`), resolved by hand — not
  constructed or guessed — to the Istanbul row's detailed listings link.
- **URL**: `https://data.insideairbnb.com/turkey/marmara/istanbul/2026-06-30/data/listings.csv.gz`
  — verified live via HTTP HEAD (200 OK, `Content-Type:
  application/x-gzip`, `Content-Length: 13047871`, `Last-Modified: Fri, 03
  Jul 2026`) before being pinned into
  `services/travel-mcp/phase2/data_pipeline/download_dataset.py`.
- **Scope**: only `listings.csv.gz` (detailed) is downloaded for Checkpoint
  A. `calendar.csv.gz`, `reviews.csv.gz`, and the neighbourhood GeoJSON were
  not downloaded — nothing in the Checkpoint A profiling or design work
  proved them mandatory.
- **License**: CC BY 4.0, per Inside Airbnb's stated terms; attribution text
  recorded in the manifest (`data/manifests/istanbul_listings_2026-06-30.manifest.json`).
- **Currency finding**: the CSV's `price` column is formatted with a
  leading `$`, but this is Inside Airbnb's fixed CSV template symbol, not a
  currency indicator. Cross-referencing the embedded `price_quote_raw` JSON
  shows the true currency is Turkish Lira (`₺`/TRY). Recorded in the
  manifest's `content.price_notation` field and in
  `services/travel-mcp/phase2/EXPERIMENT_DESIGN.md` §1 so this isn't
  rediscovered (or missed) during Checkpoint B.

## 4. Repository placement of data vs. pipeline code

- **Data** (raw download, processed output, manifests, model artifacts)
  lives at the **superproject root** (`data/`, `ml/`), reusing the
  pre-existing empty scaffold directories from the initial repo scaffold
  commit. This matches decision 6 below: Compose (superproject-owned) will
  eventually bind-mount these paths into the `mcp-server` container, which
  is far more natural from a root-level directory than reaching into a
  submodule's working tree.
- **Pipeline code** (downloader, manifest schema, profiler, experiment
  design, reference scorer, tests) lives in
  `services/travel-mcp/phase2/`, consistent with ADR 0001's ownership
  assignment of `FairPriceEstimate`/`StayOption` to `travel-mcp`. It accepts
  an explicit `--output-dir` rather than assuming the superproject's
  directory layout, so `travel-mcp` remains independently buildable/testable
  as its own submodule/repo (CLAUDE.md: "each `services/*` directory is an
  independent git submodule with its own remote and commit history").

## 5. Artifact delivery (offline training, no startup training)

- Training is **offline and reproducible** — run by a developer or CI
  outside the container, not inside it.
- The `mcp-server` container **never trains at startup**.
- Raw and processed datasets are **not baked into the Docker image**.
- The selected model bundle (model + preprocessing + feature schema +
  metrics, per architecture.md §10.1) and the serving-time processed
  dataset are **generated artifacts** — reproducible from
  `data/raw/` + `services/travel-mcp/phase2/`, not committed (see the
  `.gitignore` scoping below).
- Compose will, in a later checkpoint, mount the required generated
  artifacts **read-only** into the `mcp-server` service using explicit
  paths (e.g. a bind mount of `ml/artifacts/` to a fixed in-container path).
  **No Compose or Dockerfile changes are made in this checkpoint.**
- Travel MCP remains independent of Qdrant — nothing in this checkpoint
  gives the MCP server a vector-DB client, consistent with architecture.md
  §9.1 and ADR 0001 decision 5.
- Tool/startup behavior must **fail clearly** (no internal detail leaked,
  per CLAUDE.md's error-handling rule) when an expected artifact is missing
  or incompatible with the running code's expected schema version — this is
  a requirement for the Checkpoint B/C tool implementation, recorded here so
  it isn't dropped.

## 6. `.gitignore` scoping

Narrow, path-scoped ignore rules were added (not blanket `*.csv` /
`*.json` / `*.joblib` patterns):

```
data/raw/*
!data/raw/.gitkeep
data/processed/*
!data/processed/.gitkeep
ml/artifacts/*
!ml/artifacts/.gitkeep
```

`data/manifests/` (small aggregate JSON, no record-level data) is
intentionally **not** ignored — it is meant to be committed for
auditability. The pre-existing `ml\artifacts\*.joblib` line (added in the
initial scaffold commit, `ff6f254`) used Windows backslashes, which git does
not treat as a path separator in `.gitignore` — it was non-functional.
Replaced as part of this change.

**Checkpoint A.2 correction**: this section originally also left
`ml/reports/` untracked-but-not-ignored (committed), reasoning that
human-readable metrics writeups were worth keeping for audit trail. §7
below locks a stricter, simpler rule — "reports containing generated
metrics ... remain generated artifacts and are not committed" — which
supersedes that. `ml/reports/*` is now gitignored the same way
`data/raw/`, `data/processed/`, and `ml/artifacts/` are (see §7). The
directory is currently empty, so this correction has no committed content
to walk back.

## 7. Repository and artifact ownership (Checkpoint A.2)

- **`services/travel-mcp` owns** Phase 2 data-processing, training,
  evaluation, model-bundle loading, and future MCP-serving code (`search_stays`,
  `estimate_fair_price`). All of it lives under `services/travel-mcp/phase2/`
  today — `data_pipeline/` (download, manifest, profiling, side mapping,
  real-dataset validation), `deal_score_reference.py`, `EXPERIMENT_DESIGN.md`,
  and `tests/`. This matches ADR 0001's ownership assignment of
  `FairPriceEstimate`/`StayOption` to `travel-mcp`.
- **The superproject owns** cross-service architecture documentation
  (`docs/architecture.md`, `docs/adr/*`), the dataset manifest
  (`data/manifests/*.json`), the generated-data/model directory conventions
  (`data/raw/`, `data/processed/`, `ml/artifacts/`, `ml/reports/` — empty,
  gitignored scaffolding that establishes *where* generated artifacts land,
  not code that produces them), and — later — the Compose mounts that wire
  those directories into the `mcp-server` container.
- **Raw datasets, processed datasets, reports containing generated metrics,
  and model bundles are generated artifacts and are not committed.** All
  four now share the same `.gitignore` treatment (§6): `data/raw/*`,
  `data/processed/*`, `ml/artifacts/*`, `ml/reports/*`, each with a tracked
  `.gitkeep` so the directory structure itself survives a clean clone.
- **One implementation of parsing, features, and model-bundle compatibility
  logic** — not duplicated between the superproject and Travel MCP. The
  superproject holds no parsing/feature/model-loading code of its own (only
  documentation and generated-artifact directories); all of that logic lives
  exclusively in `services/travel-mcp/phase2/`.
- **Compose will eventually mount** the selected serving dataset and model
  bundle read-only into Travel MCP, from the superproject's `data/`/`ml/`
  directories, using explicit paths — not implemented this checkpoint (no
  Compose or Dockerfile changes).
- **Travel MCP remains independent of Qdrant** — unchanged, still true, no
  vector-DB client anywhere in `services/travel-mcp`.

**Conformance check against existing paths**: all of the above already
holds, with one mismatch found and corrected in this same remediation —
`ml/reports/` was previously tracked-not-ignored (§6's original text),
contradicting "reports containing generated metrics ... are not committed."
Fixed by extending `.gitignore` (§6, above). No other mismatch found; no
broad relocation was needed or performed.

## 8. Dependency-lock direction (Checkpoint A.2 — decision only, not applied)

Checkpoint A.1's dependency-layout audit found that `phase0/requirements.lock`
is the only file the Dockerfile actually installs, while `phase0/`, `phase1/`,
and `phase2/` each carry their own `requirements.in`/`requirements-test.in`
that are never mechanically merged into it. Phase 2 is the first phase whose
declared runtime deps (`httpx`, `pandas`) are genuinely absent from that lock.
This section locks the intended fix's shape without applying it.

**Decision**:

- `phase0/requirements.lock` stops being treated as the permanent production
  lock for the whole service — it was named for, and originally scoped to,
  Phase 0 only, and next treating it as "the" lock any time a later phase
  adds a dependency invites exactly the drift just found.
- Travel MCP will instead have **one canonical service-level runtime lock
  covering the complete application** (all phases' runtime code), replacing
  `phase0/requirements.lock` as the thing the Dockerfile installs.
- **Test/development dependencies stay separate from runtime dependencies**
  — a canonical test/dev requirements file, distinct from the canonical
  runtime lock, so a production image never pulls in `pytest` et al.
- **Proposed exact filenames** (service-root level, replacing the
  phase0-scoped names):
  - `services/travel-mcp/requirements.in` — canonical runtime dependency
    *source* (uncompiled, human-edited).
  - `services/travel-mcp/requirements-test.in` — canonical test/dev
    dependency source.
  - `services/travel-mcp/requirements.lock` — canonical **compiled,
    fully-pinned** runtime lock; this is the one file the Dockerfile installs
    going forward.
- **`phase0/requirements.in`, `phase1/requirements.in`, `phase2/requirements.in`**
  (and their `-test.in` counterparts) **remain** — but strictly as
  incremental, per-phase *scope documentation* ("what this phase's own code
  needs"), not as competing production sources of truth. Nothing installs
  from them directly once the canonical lock exists.
- `phase0/requirements.lock` itself is retired (deleted or left as a
  historical artifact, decided at migration time) once
  `services/travel-mcp/requirements.lock` exists and the Dockerfile points
  at it.
- **The Dockerfile will later install the canonical service runtime lock**
  instead of `phase0/requirements.lock` — not changed in this checkpoint.
- **This migration (new lock files + Dockerfile wiring) must happen before
  any container-based Phase 2 acceptance test** — i.e. before Checkpoint B's
  `search_stays`/`estimate_fair_price` MCP tools are expected to actually run
  inside the built image, since today's Dockerfile-installed dependency set
  does not include `httpx` or `pandas`.

**Not done in this checkpoint**: no file was renamed, no Dockerfile edited,
no new dependency installed. `phase2/requirements.in`/`requirements-test.in`
(created in Checkpoint A) are unchanged and remain valid as the future
canonical file's input once the migration above happens.
