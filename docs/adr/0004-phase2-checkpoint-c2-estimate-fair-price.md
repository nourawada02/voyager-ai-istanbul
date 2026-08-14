# ADR 0004: Phase 2 Checkpoint C.2.1 — `estimate_fair_price` as a standalone MCP tool

Status: accepted. Date: 2026-08-14.

Short design note, not a rewrite of `docs/architecture.md` (canonical) or
ADR 0003 (the `search_stays` serving contract, unchanged by this
checkpoint). Records why `estimate_fair_price` exists as its own MCP tool
and the specific decisions its contract makes.

## 1. Why a distinct tool, not just a field inside `search_stays`

Every `search_stays` result item already nests a `fair_price` object
(`contracts/SearchStaysResult.schema.json`), but that is not the same as
callers being able to ask "what would this system say about listing X"
without first running a full search. `docs/architecture.md` §8's tool
inventory and §17's Phase 2 gate both name `search_stays` and
`estimate_fair_price` as two separate tools — the accepted architecture
does not describe the nested field as satisfying the second one. The
Checkpoint C.2 readiness audit found no accepted request contract or
public service method for a standalone lookup; this checkpoint closes
that gap rather than reinterpreting the nested field to cover it.

## 2. Input: a stable historical `stay_id`, never raw model features

Two shapes were possible for `EstimateFairPriceRequest`: a lookup by the
dataset's own stable listing id, or a caller-supplied feature vector
(bedrooms, location, etc.). The stable-id shape was chosen and the
feature-vector shape is explicitly prohibited:

- A raw-feature endpoint would let any caller probe the model with
  synthetic inputs never grounded in a real historical listing — a
  different, unreviewed capability (closer to a general pricing
  simulator) that was never part of the accepted architecture.
- It would also require re-deriving/validating the same feature
  engineering `phase2/ml/prepare_dataset.py` already owns, duplicating
  logic the pure C.1 pipeline exists specifically to keep in one place.
- A stable-id lookup keeps `estimate_fair_price` answering exactly the
  question `search_stays` already answers per row — "what is the fair
  price for this real listing in this snapshot" — through the identical
  code path (`AccommodationSearchService._build_stay_and_fair_price`,
  shared by both tools), so the two tools can never silently diverge in
  rounding, clamping, or provenance.

`contracts/EstimateFairPriceRequest.schema.json` therefore has no field
shaped like a coordinate, a feature name, or a raw numeric attribute —
`additionalProperties: false` makes supplying one a hard schema
rejection, not an implementation choice to ignore it.

## 3. Currency restricted to TRY

The frozen Checkpoint B bundle (`ml/artifacts/phase2_bundle_2026-06-30.joblib`)
was trained and calibrated in TRY only (`ServingBundle.currency`, checked
at startup by the C.1 loader). No accepted currency-conversion contract
exists anywhere in this system. `currency` is therefore `const: "TRY"` —
present and required (not defaulted away) so a future second currency is
an explicit, visible schema change, exactly mirroring `SearchStaysRequest`'s
own `currency` field and ADR 0003 §6's reasoning for keeping `ranking_mode`
explicit rather than an implicit default.

## 4. Historical, not-live semantics (unchanged from `search_stays`)

`EstimateFairPriceResult` carries the same manifest-derived
`snapshot_date`/`dataset_sha256`/`bundle_sha256`/`provenance`/`disclaimer`
fields as `SearchStaysResult`, sourced from the identical
`AccommodationResources.provenance` (Checkpoint C.1.1) — never request
execution time, never an implied current price or availability. This
tool makes exactly the same "historical snapshot, not live" claim
`search_stays` already makes, via the same data, not a second copy of it.

## 5. Unsupported-room and unknown-id behavior

- A `stay_id` absent from the currently loaded snapshot returns the new
  `STAY_NOT_FOUND` error (exact match only — no fuzzy or nearest-id
  lookup), never a fabricated or nearest-neighbor estimate.
- A `stay_id` that resolves to a `Hotel room` or `Shared room` listing is
  rejected with the existing `UnsupportedRoomTypeError` →
  `UNSUPPORTED_ROOM_TYPE`, the identical policy `search_stays` already
  enforces (ADR 0003 §5) for the identical reason (no reliable sealed-test
  evidence for those room types) — not a second, separately-tunable
  policy.

## 6. Reuse of the frozen bundle and the C.1 prediction/serialization path

`AccommodationSearchService.estimate_fair_price(stay_id)` performs the
same steps `search()` performs for one already-loaded row: a single
vectorized `ServingBundle.predict_with_interval()` call (never refit,
never a second model), then the same private
`_build_stay_and_fair_price()` helper `search()`'s own result-assembly
loop calls — the one place either tool builds a `StayDisplay`/
`FairPriceDisplay` pair, so monetary rounding (`Decimal` half-up),
interval clamping, and provenance construction cannot silently diverge
between the two tools. No filesystem or model loading occurs per call —
`estimate_fair_price` reuses the same `AccommodationResources` the MCP
server's lifespan already loaded once at startup (Checkpoint C.1/C.2.1).

## 7. Not decided here

- A second currency or a conversion contract — out of scope, no accepted
  design exists.
- Any endpoint accepting raw/synthetic features — explicitly rejected by
  §2 above, not merely deferred.
