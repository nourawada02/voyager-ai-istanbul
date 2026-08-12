# ADR 0003: Phase 2 Checkpoint C.0 — accommodation-serving contract and safety design

Status: accepted (design-only checkpoint; no production tool implemented).
Date: 2026-08-12. Revised in Checkpoint C.0.1 (same date) after the initial
C.0 draft was not accepted; C.0.1 formally reviewed and accepted.

Design note, not a rewrite of `docs/architecture.md` (canonical) or
`docs/architecture-plan.pdf` (immutable original) — records the specific
contract decisions Checkpoint C.0/C.0.1 made, including places where an
already-accepted Phase 1 contract structurally conflicts with this
checkpoint's requirements and how each conflict is resolved.

## Checkpoint C.0.1 revisions (this pass)

C.0's proposed request contract asked for `ranking_mode: "deal_score_desc"`
with a silent fallback to `fair_price_asc` when scoring was incomplete —
dishonest, since Travel MCP can never actually compute `deal_score` in
isolation. C.0.1 corrects nine issues, each detailed in its own section
below:

1. **§1 Ranking architecture** — replaced the dishonest request/fallback
   with a two-stage architecture: Travel MCP performs only a deterministic
   `preliminary_price_value_desc` ranking it actually owns; System A alone
   computes the final `deal_score_desc` ranking later, outside this tool.
   Every fallback-selection function was deleted, not just renamed.
2. **§3 FairPriceEstimate truth table** — closed a loophole where the old
   `not: {required: [a, b]}` pattern only forbade *both* forbidden fields
   appearing together, not each individually; also newly forbids 1.1-only
   state fields on a 1.0.0 payload, and incomplete-only fields on a
   `complete` payload. `SearchStaysResult`'s nested `stay`/`fair_price` are
   now pinned to exactly `schema_version: "1.1.0"`.
3. **§4 No-result behavior** — was ambiguous between "empty success" and a
   reserved-but-unused `NO_MATCHING_STAYS`. Locked to exactly one: zero
   eligible rows is always `NO_MATCHING_STAYS`; a success always has
   `stays.length >= 1` and `predictions_produced: true` (now a `const`).
4. **§5 Room-type validation ordering** — `room_types` is no longer a
   closed 4-value schema enum (which let an unrecognized string fail as a
   generic schema error before reaching the semantic policy check). It is
   now a bounded-but-open string list; every value, known-unsupported or
   genuinely unrecognized, reaches the same `UnsupportedRoomTypeError` path
   and produces one uniform `UNSUPPORTED_ROOM_TYPE` error with structured,
   machine-readable `details`.
5. **New §5a: canonical district registry** — `contracts/IstanbulDistrictRegistry.schema.json`
   + the canonical instance at `contracts/examples/valid/IstanbulDistrictRegistry.json`,
   derived 1:1 from `side_mapping.py`'s accepted 39-district table. No such
   registry existed before this pass.
6. **§6 Date/budget** — unchanged in substance from C.0; semantic tests now
   use explicitly constructed dates, never "today".
7. **New §7a: monetary rounding and interval clamping** — removed from the
   "not decided" list. Decimal half-up to the nearest kuruş is the one
   documented minor-units conversion rule; a negative conformal lower bound
   is clamped to 0 at the serving boundary only, disclosed via a new
   `lower_bound_clamped` flag, never altering the trained artifact or
   calibration.
8. **§10 Resource limits** — added `REQUEST_TOO_BROAD` (>5,000 scored
   candidates) as a typed, non-retriable error instead of silent
   sampling/truncation.
9. **§9 Success-field alignment** — `SearchStaysResult` now also requires
   `currency`, `data_mode`, and `model_version` at the top level, and every
   returned item's `fair_price.interval` is required.

This ADR governs the *contract* for the future `search_stays` MCP tool. It
does not implement `search_stays`, does not create an MCP server/tool
registration, does not modify Compose/Dockerfiles, and does not retrain or
re-evaluate the Checkpoint B model. The frozen Checkpoint B result (dataset
`496db43e...4769dff`, bundle `81f857fd...856c15f`, promoted
`hist_gradient_boosting`, 40.7% improvement) is treated as read-only input.

## 1. Files inspected

- `docs/architecture.md` §§4.2, 7.2–7.3, 8, 8.1, 8.2, 9.1, 9.4, 10, 11, 12,
  13.1–13.5, 14.3–14.6, 14.10, 15.1.
- `docs/adr/0002-phase2-checkpoint-a.md` (full).
- `services/travel-mcp/phase2/EXPERIMENT_DESIGN.md` (full, prior checkpoints).
- `services/travel-mcp/phase2/deal_score_reference.py` (full).
- `services/travel-mcp/phase2/ml/bundle.py` (full, prior checkpoints; the
  `ServingBundle` schema is the artifact-loading contract's source of truth).
- `contracts/StayOption.schema.json`, `contracts/FairPriceEstimate.schema.json`,
  `contracts/DataProvenance.schema.json`, `contracts/DataQuality.schema.json`,
  `contracts/ErrorEnvelope.schema.json`, `contracts/ProviderResponseEnvelope.schema.json`,
  `contracts/TripRequest.schema.json`, `contracts/LocalPlanRequest.schema.json`,
  `contracts/TripPlan.schema.json` (reference check only).
- `contracts/examples/valid/StayOption.json`,
  `contracts/fixtures/stays_success.json`, `contracts/fixtures/stays_partial_result.json`.
- Installed pinned MCP SDK (`mcp==2.0.0`, the exact pinned disposable
  environment from Checkpoint B): `mcp.server.mcpserver.MCPServer.tool`.

## 2. Existing `search_stays` / accommodation contracts found

- `docs/architecture.md:212` — tool inventory row: `search_stays | System A | Find snapshot, sandbox, or live stay candidates`.
- `docs/architecture.md:223-224` — provider list: `StayProvider: InsideAirbnbSnapshotProvider + FixtureStayProvider (mandatory), LiveStayProvider (optional)`.
- `docs/architecture.md:231-247` (§8.2) — the required `ProviderResponseEnvelope` shape every MCP tool response (including `search_stays`) must use.
- `contracts/StayOption.schema.json` (all 42 lines) — the per-listing candidate shape, already referenced by `contracts/TripPlan.schema.json:24` (`stays: [{"$ref": "StayOption.schema.json"}]`).
- `contracts/FairPriceEstimate.schema.json` (all 55 lines) — the ML fair-price + deal-score shape.
- `contracts/fixtures/stays_success.json`, `contracts/fixtures/stays_partial_result.json` — existing fixtures already correctly use `data_mode: "historical"` and `provider: "inside-airbnb-snapshot"`, never `"live"` — this precedent is preserved, not changed.
- `docs/architecture.md:432` (§13.5 failure policy) — `ML artifact invalid/missing -> Fall back to neighborhood + room-type median baseline`.

No `estimate_fair_price`-specific request/response pair or room-type policy exists yet anywhere in the accepted contracts. This checkpoint adds them.

## 3. Conflict found and resolution (do not silently replace)

**Conflict**: `contracts/FairPriceEstimate.schema.json:30-34,40` requires
`deal_score` and all six `components` (including `itinerary_accessibility`)
unconditionally on every instance. But `itinerary_accessibility` is
System-B-owned and caller-supplied (`docs/architecture.md:196`,
`deal_score_reference.py:9-11,60-68` — `DealScoreComponents` requires all six
as caller-supplied floats, none fabricated). Travel MCP, called in isolation
by `search_stays`/`estimate_fair_price` without a System-A/System-B
itinerary round-trip, structurally **cannot** always produce
`itinerary_accessibility`. The current schema has no way to express "deal
score not yet computable" — it can only be silently satisfied by fabricating
a neutral value (explicitly forbidden by this checkpoint's §8) or by
violating the schema.

**Resolution (explicit, versioned, additive-only migration)**: `deal_score`
and `components` become conditionally required, gated by a new required
`scoring_status` enum (`"complete"` | `"incomplete"`), via `if`/`then`.
`schema_version` changes from `const: "1.0.0"` to `enum: ["1.0.0", "1.1.0"]`
so every existing Phase 1 fixture/example (which sets `"1.0.0"` and has no
`scoring_status`) is grandfathered as-is via a `1.0.0`-branch in the same
`if`/`then` (see file). No existing field was renamed, removed, or
retyped. New producers set `schema_version: "1.1.0"` and must set
`scoring_status` explicitly. `contracts/examples/valid/FairPriceEstimate.json`
and `contracts/fixtures/*` were re-validated unchanged against the new
schema and still pass.

`StayOption.schema.json` has a lower-stakes version of the same pattern:
it has no `room_type`, `availability_status`, or `snapshot_date` field,
all three structurally required by this checkpoint's §4/§3. These are added
as new **optional** properties (not required at the base-contract level,
since `StayOption` is also used by non-accommodation-ML providers/paths per
`docs/architecture.md:222-228`), with the same `enum: ["1.0.0","1.1.0"]`
version widening. `search_stays`'s own result contract (§7 below) tightens
these three to required via a non-conflicting `allOf` (`{"$ref":
"StayOption.schema.json"}` plus a sibling `{"required": [...]}​` — this
does not hit the classic `allOf` + `additionalProperties:false` composition
trap because the second schema adds no new properties, only tightens
`required`).

Both migrations were re-validated: every pre-existing fixture/example under
`contracts/examples/` and `contracts/fixtures/` referencing `StayOption` or
`FairPriceEstimate` still validates against the updated schemas unchanged.

## 4. Truthful snapshot-data limitations (locked)

The dataset is Inside Airbnb, snapshot date `2026-06-30`, dataset SHA-256
`496db43edf236a53737bfe2f0f868ad4f812ff5db0f61798dba7de4ee4769dff` — frozen,
not live inventory. The contract must never let a response, by field
presence or by omission, imply: current availability, a live price, a
successful booking, current host status, that a listing remains active, or
that a calibrated interval guarantees a real future price (it is a
finite-sample split-conformal interval over historical residuals — a
statistical coverage statement about the *model*, not a promise about any
future real-world price).

Accepted dates (`check_in`/`check_out`) are used **only** to compute
`nights` and an `estimated_total` (`nightly_price × nights` in TRY minor
units) — never to filter, imply, or claim availability. This is enforced at
the schema level: nothing in `SearchStaysRequest` or `SearchStaysResult` is
named or typed in a way that could represent "available"/"bookable"; the
result always exposes `availability_status: "unknown"` (a one-value enum,
i.e. functionally a typed constant that cannot silently drift to something
else later without a schema change forcing review).

Every successful `SearchStaysResult` exposes, at the envelope level
(`ProviderResponseEnvelope`) and/or the result level: `retrieved_at`/
`data_mode: "historical"` (envelope, already-existing fields — reused, not
duplicated), `currency: "TRY"` (envelope), `snapshot_date` and
`dataset_sha256` (new, result-level — the provenance *reference*, not a
re-verification; Travel MCP's loader is what actually verifies the checksum
at startup, §9), `availability_status: "unknown"` (new, per-item),
`model_version`/`configuration_hash` (existing `FairPriceEstimate.model_version`
plus new `bundle_sha256`), and `predictions_produced`/`scoring_status`
(new — whether a prediction and a complete deal score were actually
produced). A fixed disclaimer string is required in `SearchStaysResult.disclaimer`
(new, required, minLength-bounded, content asserted by a contract test to
contain "snapshot" and not contain any of "available now"/"book"/"confirmed").

## 5. Room-type safety policy (locked, serving-support only — not a model-selection change)

V1 supported room types: `Entire home/apt`, `Private room` only. This is a
**serving-support** restriction layered on top of the already-frozen
Checkpoint B model/artifact — it changes nothing about model selection,
promotion, or the frozen 40.7% result.

Unsupported, with reasons (from the frozen, unchanged sealed-test report
`ml/reports/phase2_final_2026-06-30.json.subgroup_metrics.room_type`):
- `Hotel room`: test n=23, MAE≈18,749.91 TRY, R²=−0.531 (worse than the
  mean baseline — a real, disclosed failure mode carried since the B.1
  audit, never remediated).
- `Shared room`: below the predetermined subgroup-sufficiency threshold
  (`DEFAULT_MIN_SUBGROUP_SIZE = 20`, `phase2/ml/metrics.py:16`) in the test
  partition, so no subgroup metric exists for it at all — there is no
  evidence basis to serve it, good or bad.
- Any room-type string not in `SUPPORTED_ROOM_TYPES` — whether it is a known
  Inside Airbnb value (`Hotel room`, `Shared room`) or a genuinely
  unrecognized future string — is unsupported and rejected. **Checkpoint
  C.0.1 correction**: both cases produce the *same* `UNSUPPORTED_ROOM_TYPE`
  semantic error, never a generic schema-level `INVALID_REQUEST` for the
  unrecognized-string case — see §4a below for why that distinction was
  removed.

Required behavior (all enforced in the request/response contracts below,
independent of any implementation):
- Default `SearchStaysRequest` (no `room_types` filter given) is defined to
  mean "all supported types" — never "all four" — so a caller who never
  thinks about room type still gets a safe result.
- `room_types: ["Hotel room"]` or any single unsupported value fails with
  `UNSUPPORTED_ROOM_TYPE`, listing `supported_room_types` in the error
  details.
- `room_types: ["Private room", "Hotel room"]` (mixed) fails the same way —
  the whole request is rejected, not silently narrowed to `["Private room"]`.
- No unsupported row is ever passed to the learned model or the strong
  baseline (enforced structurally, §7 step 5: prediction only runs after
  the hard room-type filter).
- No Hotel-room-specific correction factor, calibration adjustment, or
  routing rule is introduced — confirmed absent from every schema/fixture/
  test added in this checkpoint.

**Evidence that would be required before expanding support** (recorded, not
pursued here): for `Hotel room`, either (a) a materially larger sealed-test
subgroup from a future dataset refresh with re-run promotion evaluation
showing MAE competitive with the strong baseline on that subgroup
specifically, or (b) a dedicated hotel-room model/baseline trained and
sealed-test-evaluated on its own right (not a correction factor bolted onto
the existing generalist model). For `Shared room`, simply reaching
n≥20 in a future sealed test with a reported subgroup MAE/R² — sufficiency
alone, evaluated honestly, not assumed to be fine by extrapolation.

### 4a. Unified semantic rejection, structured error details (Checkpoint C.0.1)

C.0's `SearchStaysRequest.room_types` was a closed 4-value schema `enum`
(the full known vocabulary), so a caller-typed value outside that vocabulary
(e.g. a future 5th room type, or a typo) failed generic JSON Schema
validation as `INVALID_REQUEST` — a *different* rejection path than a
known-but-unsupported value (`Hotel room`), which reached the semantic
`UnsupportedRoomTypeError`. This split the "your room type doesn't work"
experience into two inconsistent error shapes for what is, from the
caller's perspective, the same situation. **Correction**: `room_types`
items are now only bounded (`minLength: 1, maxLength: 64`, up to 8 items,
`SearchStaysRequest.schema.json`), not enum-restricted; every value —
known-unsupported or unrecognized — reaches
`serving_contract_reference.resolve_effective_room_types()` and produces
`UnsupportedRoomTypeError` uniformly, always mapped to one
`UNSUPPORTED_ROOM_TYPE` error.

The error's machine-readable support information must not live only inside
the human-readable `message` string (easy to misparse, not contractually
stable). `ErrorEnvelope` gained an additive, optional, bounded `details`
object (`schema_version` widened to `enum: ["1.0.0", "1.1.0"]`, exactly the
same non-breaking pattern as the earlier `StayOption`/`FairPriceEstimate`
migrations). `details` is flat (`maxProperties: 10`), and every value is a
short string/number/boolean or a short list of short strings — no nested
objects, no arbitrary depth, so it can never carry a stack trace or a
structured dump of internal state. `UNSUPPORTED_ROOM_TYPE` populates
exactly three keys: `requested_room_types` (everything the caller sent),
`unsupported_room_types` (the offending subset), `supported_room_types`
(always `["Entire home/apt", "Private room"]`).

## 5a. Canonical Istanbul district registry (Checkpoint C.0.1, new)

No Phase 1 district catalog contract existed (searched: every `contracts/*.schema.json`
and `contracts/examples/`; the only district-adjacent hit was `POI.schema.json`,
which identifies individual points of interest, not a district roster).
Created `contracts/IstanbulDistrictRegistry.schema.json` plus the single
authoritative instance at `contracts/examples/valid/IstanbulDistrictRegistry.json`
(not merely a validation fixture — this file *is* the registry; the schema's
`minItems`/`maxItems: 39` makes a smaller "sample" instance impossible, so
there is structurally one list, never a second conflicting one).

Derived 1:1 from `services/travel-mcp/phase2/data_pipeline/side_mapping.py`'s
already-accepted `_DISTRICTS_BY_SIDE` table (25 European + 14 Asian = 39
real districts; `side_mapping.py` itself has 40 *lookup keys* because
`"Eyup"`/`"Eyupsultan"` are two alias spellings of one real district — the
registry correctly has one canonical row, `district_eyup`, for it).
Cross-checked programmatically (not by eye): every registry `district_id`
maps to a `side_mapping.py` entry with a matching `side`, and no registry
entry is unmatched.

Each entry: `district_id` (ASCII, matches `StayOption.district_id`'s
existing pattern and `side_mapping.py`'s own transliterated names, e.g.
`district_besiktas`), `canonical_name` (correct Turkish Unicode spelling,
current official name — e.g. `"Beşiktaş"`, `"Eyüpsultan"`), `side`
(`"EUROPEAN"` | `"ASIAN"`, uppercase). Contains only these three fields —
no raw listing rows, no `host_id`, no dataset-derived statistics (contract-
tested).

**Uppercase-vs-lowercase side normalization, documented explicitly**: the
registry's own canonical `side` is uppercase (`EUROPEAN`/`ASIAN`), matching
this ADR's usage elsewhere; the already-accepted Phase 1
`StayOption.side` enum is lowercase (`european`/`asian`) and is **not**
changed by this checkpoint (changing an already-shipped enum's casing would
be a breaking change with no safety benefit). The one documented,
deterministic normalization rule connecting them: `side.lower()`. A future
`UNKNOWN_DISTRICT` semantic check resolves a requested `district_id`
against this registry and, if found, can supply `side` for cross-validation
against a caller-supplied `side` filter using that exact lowercasing rule —
never a second, independently-maintained side table.

## 6. Request contract — `SearchStaysRequest` (new: `contracts/SearchStaysRequest.schema.json`)

Reconciled with Phase 1 rather than inventing parallel names: `session_id`/
`trace_id` (same UUID pattern as `TripRequest`/`LocalPlanRequest`), `budget`
(same `{amount_minor_units, currency}` shape as `TripRequest.budget`),
`district_id` values (same `^district_[a-z0-9_]+$` pattern as
`StayOption.district_id`), `side` (same `["european","asian"]` enum as
`StayOption.side`).

Resolved decisions:
- **Guest count**: `guest_count`, integer, `minimum: 1`, `maximum: 12` — the
  same upper bound as `TripRequest.traveler_count` (`contracts/TripRequest.schema.json:21`),
  reused rather than inventing a different cap. Required.
- **Dates**: `check_in`/`check_out`, both optional but **mutually
  required together** (`dependentRequired`) — a lone date is a
  `CONTRADICTORY_FILTERS` case, not silently ignored. `check_out` must be
  strictly after `check_in` (contract test, since JSON Schema alone cannot
  express cross-field date ordering — documented as a semantic validation
  rule enforced by the (future) implementation, tested here via fixtures
  marked invalid). Dates only ever drive `nights`/`estimated_total`; the
  schema description explicitly states they never imply availability.
- **Budget semantics**: `budget` is optional; when present, `budget.period`
  (`enum: ["nightly", "total"]`) is **required** alongside it — the
  ambiguity is resolved by forcing the caller to state it, never guessed.
  `period: "total"` without both `check_in` and `check_out` present is a
  `CONTRADICTORY_FILTERS` case (a total needs a night count).
- **Currency**: `currency` is `const: "TRY"` — fixed for V1, present so the
  contract is self-describing and future currency support is an explicit,
  visible schema change, not a silent default.
- **District / side filters**: `district_ids` (array of `district_id`-pattern
  strings, `maxItems: 20`, no duplicates via `uniqueItems: true`), `side`
  (single enum value, optional).
- **Room-type filter**: `room_types`, array of bounded-but-open strings
  (`minLength: 1, maxLength: 64`, up to 8 items) -- not an enum, per S4a:
  every value must reach the semantic UNSUPPORTED_ROOM_TYPE check uniformly,
  never split into a schema-level rejection for unrecognized strings and a
  semantic one for known-but-unsupported strings. `uniqueItems: true`,
  `minItems: 1` when present (an explicit empty array is a
  `CONTRADICTORY_FILTERS` case per the omitted/null/empty rule below, not
  "no filter").
- **Minimums**: `min_bedrooms`, `min_beds` (integers, `minimum: 0`),
  `min_bathrooms` (number, `minimum: 0`, since Inside Airbnb bathrooms can
  be fractional, e.g. 1.5 — confirmed by `NUMERIC_FEATURES` in
  `phase2/ml/prepare_dataset.py`).
- **Amenities**: `required_amenities`, array of non-empty strings,
  `maxItems: 20`, `uniqueItems: true`.
- **Result limit**: `result_limit`, integer, `minimum: 1`, `maximum: 50`
  (strict cap — §10). Required, no implicit default inside the schema
  (a caller must state it), so the cap is always visible in the request,
  not buried in server-side config.
- **Ranking mode**: `ranking_mode`, `const: "preliminary_price_value_desc"`
  (Checkpoint C.0.1 correction — was `"deal_score_desc"`, which asked for a
  ranking Travel MCP cannot produce). One deterministic mode, present (not
  omitted) so a future second mode is an explicit, visible addition rather
  than a silent behavior change. There is no request field for the final
  `deal_score_desc` ranking — that composition belongs entirely to System A
  (§1).
- **Unknown fields**: `additionalProperties: false`, matching every existing
  Phase 1 request contract — unknown fields are rejected, never ignored.
- **No filesystem/model/checksum/URL/executable input**: the schema has no
  string field with an unconstrained free-form "path", "model", "checksum",
  or "url"/"expression" semantic — every string field is either a closed
  enum, a constrained pattern (`district_id`, UUID), or a length-bounded
  free-text amenity name with `pattern: "^[^\\n\\r\\x00]{1,64}$"` (blocks
  control characters and caps length; still just an opaque label matched
  against the dataset's own amenity vocabulary, never interpreted as a path
  or code). No field accepts a bundle path, model name, dataset checksum, or
  URL from the caller — those are process-startup configuration only (§9).
- **Omitted vs. null vs. empty**: every optional field's JSON Schema type
  excludes `null` (no field is typed `["string","null"]` etc.), so an
  explicit `null` fails schema validation (`INVALID_REQUEST`) rather than
  being interpreted as "no filter." An omitted key means "no filter" /
  "unconstrained." An explicit empty array on a filter field
  (`district_ids: []`, `room_types: []`, `required_amenities: []`) is
  schema-valid but is defined as a semantic `CONTRADICTORY_FILTERS` case
  (a filter matching nothing) via `minItems: 1` where the field is an array
  filter — enforced at the schema level directly, not left to prose.
- **Unicode district names**: `SearchStaysRequest` accepts only canonical
  `district_id`s (ASCII `snake_case`, e.g. `district_uskudar`), reusing the
  existing POI/district canonicalization convention already established at
  `docs/architecture.md:280-282` (§9.4: aliases resolved to canonical IDs
  *before* retrieval, original wording preserved only for the reply).
  Travel MCP performs no Unicode fuzzy-matching itself; canonicalizing
  "Üsküdar" → `district_uskudar` is the caller's (System A's)
  responsibility, upstream of this contract, exactly mirroring how POI
  aliasing already works.

## 7. Response contract — `SearchStaysResult` (new: `contracts/SearchStaysResult.schema.json`)

This is the shape of `ProviderResponseEnvelope.result` (§8.2) for
`search_stays` specifically; `ProviderResponseEnvelope`'s own fields
(`data_mode`, `currency`, `retrieved_at`, `quality`, ...) are reused
unchanged, not duplicated inside `result`.

Top level: `schema_version` (`const: "1.0.0"`), `snapshot_date`,
`dataset_sha256` (pattern `^[0-9a-f]{64}$`), `bundle_sha256` (same pattern),
`disclaimer` (required string, content-tested), `predictions_produced`
(boolean), `excluded_row_count` (integer ≥0 — rows dropped for an
unexpected per-row prediction failure, never silently merged into the
result), `stays` (array, `maxItems` = the request's `result_limit`,
enforced by a contract test not just documentation).

Each `stays[]` item (`$defs/resultItem`), composed via `allOf` over the
existing `StayOption` (never duplicated field-by-field) plus new
sibling/required fields:
- `stay`: `{"$ref": "StayOption.schema.json"}` — `stay_id`, `name`,
  `district_id`, `side`, `coordinates`, `nightly_price` (**the observed
  snapshot price** — reused directly, no separate duplicate field), `rating`,
  `review_count`, `amenities`, `provenance` all inherited unchanged.
  `name`/`stay.name` (a listing title, drawn straight from the dataset) and
  a new optional `snapshot_url` are explicitly documented as **display
  metadata only** — `SearchStaysResult.$defs.resultItem` description states
  they are never fed to the model as a feature (matching
  `phase2/ml/prepare_dataset.py`'s `FEATURE_COLUMNS`, which excludes both).
- `room_type`: required, `enum: ["Entire home/apt", "Private room"]` —
  restricted to the **supported** set only, at the schema level, so a
  result row cannot represent an unsupported type even by implementation
  bug; a `Hotel room` value here would fail contract validation.
  `property_type` (optional, free descriptive string, display-only).
- `capacity`: optional object (`accommodates`, `bedrooms`, `beds`,
  `bathrooms`), mirroring `phase2/ml/prepare_dataset.py`'s numeric feature
  names exactly.
- `availability_status`: required, `enum: ["unknown"]` (§4).
- `fair_price`: required, `$ref: FairPriceEstimate.schema.json` (the
  updated, `scoring_status`-aware version, §3) — `estimated_fair_price`,
  `interval` (new optional field added to `FairPriceEstimate` alongside the
  `scoring_status` migration: `{lower_minor_units, upper_minor_units,
  currency, target_coverage}`, from `ConformalCalibration`), `deal_score`/
  `components` present only when `scoring_status: "complete"`.
- `estimated_total`: optional (`{amount_minor_units, currency, nights}`),
  present **only if and only if** the request supplied both dates —
  enforced by a contract test pairing request/response fixtures.
- `rank`: required integer ≥1, unique and gapless within one response
  (§8's deterministic tie-breakers, below).
- `warnings`: optional array of free-text strings — purely informational
  disclosures (e.g. "wide prediction interval"), never a numeric adjustment.

Rounding: a `display` sub-object may be added by the future implementation
for presentation-rounded strings; **not defined in this checkpoint's
schema** because it's additive and non-blocking — the schema explicitly
documents (in `description`) that `amount_minor_units` fields are the only
values ranking/comparison may use, and rounding happens only in a caller-
facing display layer, never before ranking.

Never exposed (absent from every schema/field added this checkpoint, by
construction, not by a redaction step): `host_id`, `train`/`validation`/
`calibration`/`test` split assignment, any `preprocessing`/`ColumnTransformer`
internal, any local filesystem path, any stack trace, any row index or
training-time identifier.

## 7a. Monetary rounding and interval clamping (Checkpoint C.0.1, new — removed from "not decided")

**Rounding**: model inference and ranking always use unrounded numeric
values (already true — `phase2/ml/bundle.py`'s `predict()`/
`predict_with_interval()` return `float`/`np.ndarray`, never pre-rounded;
`preliminary_capped_price_value`, §1, is likewise full-precision). The one
documented, deterministic conversion to the serialized integer minor units
every contract requires: **Decimal half-up to the nearest kuruş**
(`serving_contract_reference.to_minor_units()`/`try_amount_to_minor_units()`,
`decimal.ROUND_HALF_UP` — explicitly not Python's `round()` default
banker's rounding, which would round `0.5` to `0`, not `1`). Display/UI
formatting (thousands separators, locale) is derived from the already-
rounded minor-units integer and never feeds back into ranking or a second
independent rounding pass.

**Interval clamping**: `phase2/ml/calibration.py`'s split-conformal
interval (`point - quantile`) can be negative — a real, already-disclosed
property of the frozen Checkpoint B calibration, unrelated to this
checkpoint and not being changed here. `FairPriceEstimate.interval` now
requires `lower_bound_clamped: boolean` alongside a `lower_minor_units`
that is itself schema-constrained to `minimum: 0` — i.e. the *serialized*
lower bound is always non-negative, and the flag discloses whether that
required a clamp. `target_coverage` and every other calibration provenance
field are carried through unchanged. This is explicitly a **serving
representation rule**: it clamps a value at the response boundary; it does
not retrain, refit, or recalibrate anything, and the underlying
`ConformalCalibration` object, its `quantile_try`, and `n_calibration` are
untouched (`clamp_lower_bound()` is a pure function of one integer, with no
access to the bundle, dataset, or calibration object at all).

## 8. Filtering, prediction, and ranking order (locked pipeline)

1. Validate and normalize the request against `SearchStaysRequest`
   (schema validation, then semantic checks: date ordering, budget/period/
   date consistency, empty-array-filter rejection) — reject before any
   dataset or model access.
2. Enforce the room-type policy (§5): resolve the effective room-type set
   (all-supported if omitted; exact requested set if given, first checking
   every requested value against the supported set and failing closed on
   the first unsupported value found — not partially applying the request).
3. Load *only* the already-verified snapshot dataset and the already-strict-
   loaded bundle (§9) — no re-parsing, no re-checksumming per request.
4. Apply hard eligibility filters (room type, district/side, guest
   capacity ≥ `guest_count`, min bedrooms/beds/bathrooms, required
   amenities subset) — a candidate failing any hard filter is excluded, full
   stop; hard filters are never downgraded to a scored preference.
5. Produce predictions only for the remaining eligible, supported rows,
   via the bundle's own fail-closed `predict()` (Checkpoint B.3) — a
   per-row prediction failure excludes that row and increments
   `excluded_row_count`; it never aborts the whole request and never
   substitutes another estimator (already structurally guaranteed by
   `ServingBundle.predict()`, reused unchanged).
6. Compute the calibrated interval (`predict_with_interval`, reused
   unchanged), clamp a negative lower bound to 0 (§7a), and, when both
   dates are present, `estimated_total`.
7. Compute the preliminary ranking value for every eligible row:
   `preliminary_capped_price_value = cap_price_value((estimated_fair_price
   - observed_nightly_price) / estimated_fair_price)` — the already-accepted
   capped-price-value component, applied only to price information Travel
   MCP owns. Separately, attempt the full six-component deal score;
   `scoring_status: "complete"` only if all six are available, otherwise
   `"incomplete"` with `missing_components` — currently always
   `["itinerary_accessibility"]` in Travel-MCP-only serving, since System B
   is never called from inside `search_stays` (§7.2–7.3: that composition
   is System A's job, entirely outside this tool). Deal-score incompleteness
   never removes a row from results and never blocks price/interval
   disclosure or the preliminary ranking — only the (usually absent)
   deal-score sub-object is affected, and it is **never** used for ranking
   here regardless of its status (§1).
8. Rank: **always** by `preliminary_capped_price_value` descending — this is
   the only ranking basis `search_stays` ever produces, disclosed via the
   fixed `ranking_basis: "preliminary_price_value_desc"` field. There is no
   fallback to a second basis and no code path that ranks by `deal_score` —
   that ranking is computed later, by System A, entirely outside this tool,
   once System B supplies `itinerary_accessibility`. Deterministic
   tie-breakers, applied in order on full-precision values: (1)
   `preliminary_capped_price_value` descending, (2) `stay.nightly_price`
   ascending, (3) calibrated interval width (`upper - lower`) ascending, (4)
   `stay_id` ascending (final, always-distinct tiebreak — guarantees total
   ordering).
9. Truncate to `result_limit` **after** ranking, never before (a lower-
   ranked-but-cheaper row must not squeeze out a better one by truncating
   early).
10. Attach `rank` (post-truncation position), provenance
    (`ProviderResponseEnvelope` fields), and `warnings`.

Missing-value handling: a candidate missing a value needed by a *requested*
hard filter (e.g. `min_bathrooms` requested but the row's `bathrooms` is
NaN pre-imputation) is excluded, not imputed at the filter stage — the
model's own train-only-fit imputation (`preprocessing.py`) still applies
downstream to whatever features it always imputes, but that is a modeling
detail, never used to satisfy a caller's explicit hard filter. Excess model
uncertainty (a wide calibrated interval) affects **disclosure only**
(surfaced via `fair_price.interval` and an optional `warnings` entry when
the interval width exceeds a documented multiple of the point estimate) —
it never affects eligibility or ranking, so uncertainty cannot be used to
quietly hide or promote a listing.

**Empty-result behavior (Checkpoint C.0.1 §3 — locked to exactly one
behavior)**: zero eligible rows after hard filtering is **always**
`NO_MATCHING_STAYS`, a typed, non-retriable error — never a "successful"
empty response. `SearchStaysResult.stays` has `minItems: 1` and
`predictions_produced` is `const: true`, so a success representing zero
results is not even schema-representable. The error's `details` names which
filter *categories* eliminated results (e.g. `["room_type",
"min_bedrooms"]`) without exposing any row data or internal paths.

## 9. Deal-score completeness policy

See §3 (schema conflict/resolution) and §8 step 7/step 8. Summary: the six
weights (`0.35/0.25/0.15/0.10/0.10/0.05`, sum `1.00`) are unchanged and
untunable by this checkpoint — reused verbatim from
`deal_score_reference.py` (imported, not reimplemented, when the production
tool is eventually built). No component is ever silently renormalized when
one is missing: a missing `itinerary_accessibility` produces
`scoring_status: "incomplete"`, never a 5-component reweighting and never a
fabricated neutral (e.g. `0.5`) value — `deal_score_reference.py`'s own
`DealScoreComponents.__post_init__` already makes a partial call
structurally impossible (all six are required constructor arguments), which
this contract's `scoring_status` gate mirrors at the wire level. Output-only
rating/review fields (`stay.rating`, `stay.review_count`) are never used as
*model* inputs (confirmed: absent from `phase2/ml/prepare_dataset.py`'s
`FEATURE_COLUMNS`) — they may only feed `rating_quality`/`review_confidence`
deal-score components, which are a separate, non-ML, deterministic Python
computation per `docs/architecture.md` §10.3. Checkpoint C.0.1: `deal_score`
(complete or absent) is never consulted for ranking inside `search_stays` —
ranking always uses `preliminary_capped_price_value` (§1, §8) regardless of
`scoring_status`.

## 10. Artifact and dataset loading contract (design only, not implemented)

A single process-level loader (future `phase2/serving/loader.py` or
equivalent — not created this checkpoint) that:
- Accepts `manifest_path`, `dataset_path`, `bundle_path` as explicit startup
  configuration (env vars or CLI args), never from a tool call parameter —
  `SearchStaysRequest` has no field that could name a path, matching §6's
  "no filesystem/model/checksum/URL input" rule.
- Calls the existing, unchanged `phase2.ml.bundle.load_bundle()` (strict:
  schema version, model/preprocessing presence, `fallback_behavior`
  validation including the Checkpoint B.3 exact-hierarchy check) and the
  existing `phase2.ml.prepare_dataset.verify_manifest_and_checksum()` at
  startup, once.
- Additionally verifies, at load time, that the loaded bundle's
  `dataset_sha256`/`snapshot_date`/`currency` match the configured
  manifest/dataset being served — `BundleIncompatibleError` (existing,
  reused) on any mismatch.
- Fails the process startup loudly on any verification failure — never
  starts in a "partially serving" state. This is a readiness concern,
  distinct from a per-call tool error (§11): a `search_stays` call is never
  reached at all if the loader failed, versus a `SERVING_RESOURCE_UNAVAILABLE`
  tool-level error which implies the process is up but a call-time resource
  problem occurred.
- Loads the dataset/bundle once at startup and reuses the in-memory objects
  for every request — no per-request re-read, no re-download, no retrain.
- Performs no network access and no training at any point.
- Writes nothing at serving time (matches `phase2/ml/*`'s existing
  read/predict-only surface — `predict()`/`predict_with_interval()` are
  already pure functions of their inputs, reused unchanged).
- Has no dependency on Qdrant/vector-db — confirmed nothing in this design
  references embeddings or retrieval; accommodation search is entirely
  structured-data + ML, per `docs/architecture.md` §9.1's corpus boundary
  (RAG holds only stable knowledge, never dynamic/live-adjacent facts).
- Compose/Dockerfile mounts for `manifest_path`/`dataset_path`/`bundle_path`
  remain future work (explicitly out of scope this checkpoint, per the
  instruction) and are noted here only as a requirement that they
  eventually be **read-only** mounts, since the loader never writes.

## 11. Error taxonomy and resource limits

`ErrorEnvelope.schema.json` gains an additive optional `details` field
(`schema_version` widened to `enum: ["1.0.0", "1.1.0"]` — §4a; `error_code`
was already an open `SCREAMING_SNAKE_CASE` pattern, not a closed enum, so no
change needed there). New codes, all "safe for callers" (message is a
fixed, safe, human-readable string per code; structured, bounded, flat
`details` where noted; no interpolated internal detail anywhere):

| error_code | Caller-safe? | retriable | Trigger | `details` keys |
|---|---|---|---|---|
| `INVALID_REQUEST` | yes | false | schema validation failure, unknown field, null on a non-nullable field, empty-array filter | -- |
| `UNSUPPORTED_ROOM_TYPE` | yes | false | any requested room type outside `["Entire home/apt","Private room"]` -- known-unsupported or genuinely unrecognized, uniformly (§4a) | `requested_room_types`, `unsupported_room_types`, `supported_room_types` |
| `UNKNOWN_DISTRICT` | yes | false | `district_id` not in `IstanbulDistrictRegistry` (§5a) | `requested_district_ids`, `unknown_district_ids` |
| `CONTRADICTORY_FILTERS` | yes | false | one date without the other; `budget.period="total"` without both dates; mutually exclusive filter combination | -- |
| `NO_MATCHING_STAYS` | yes | false | **locked, always raised** (Checkpoint C.0.1 §3) whenever hard filtering leaves zero eligible rows -- a success can never represent this state | `eliminated_by_filter_categories` |
| `REQUEST_TOO_BROAD` | yes | false | hard filtering left more than 5,000 candidate rows to score (Checkpoint C.0.1 §8) -- never sampled or truncated silently | `candidate_count`, `limit` |
| `ARTIFACT_INCOMPATIBLE` | yes (generic message only) | false | bundle failed strict load (mirrors existing `BundleIncompatibleError`, message never includes the exception text itself) | -- |
| `DATASET_MISMATCH` | yes | false | configured dataset checksum != bundle's recorded `dataset_sha256` | -- |
| `SERVING_RESOURCE_UNAVAILABLE` | yes | true | loader not yet ready / process not warmed up | -- |
| `PREDICTION_INPUT_INCOMPATIBLE` | yes | false | mirrors `BundleIncompatibleError` from `_validate_prediction_input` (Checkpoint B.3) surfaced through the tool boundary, generic message only | -- |
| `INCOMPLETE_SCORE_DEPENDENCIES` | yes | false | reserved for a caller that requests deal-score-only output that cannot ever be completed without System B; not raised in the default row-level flow, where incompleteness is disclosed via `scoring_status` instead of an error | -- |

None of these ever carry a stack trace, exception `repr`, or local path —
every message is a fixed string per code, and `details` is schema-bounded
to flat, short, machine-readable values only (contract-tested, §12).

Resource limits (all enforced by the request schema or documented as
server-side constants, whichever is stricter):
- `result_limit`: schema `maximum: 50`.
- Candidate rows scored per request: **locked** at `<=5000`
  (`MAX_CANDIDATE_ROWS_SCORED`, `serving_contract_reference.py`) — a
  generous multiple of `result_limit` to allow filtering headroom, far
  below the full 23,943-row accepted dataset. Exceeding it is
  `REQUEST_TOO_BROAD`, never silent sampling/truncation; boundary-tested at
  exactly 5,000 (accepted) and 5,001 (rejected) using synthetic counts only
  — no dataset access.
- Date range: `check_out - check_in <= 90` nights (schema `description`;
  cross-field, enforced semantically, tested with explicitly constructed
  dates, never "today").
- Amenities count: schema `maxItems: 20` on `required_amenities`.
- Filter-list sizes: `district_ids maxItems: 20`, `room_types maxItems: 8`
  (bounded, not vocabulary-sized, since `room_types` is no longer a closed
  enum — §4a).
- Request string lengths: every free-text field pattern-bounded to 64–256
  characters (amenity names 64, `room_types` items 64, `warnings`/
  `disclaimer` strings unbounded only because they are *server-authored*,
  never caller-supplied).

## 12. Not decided in this checkpoint

- The canonical `district_id` table's full membership is now decided (§5a,
  `IstanbulDistrictRegistry`) — no longer open.
- Monetary rounding and interval clamping are now decided (§7a) — no longer
  open.
- No-result behavior is now decided (§3/§8/§11: always `NO_MATCHING_STAYS`)
  — no longer open.
- Presentation/display formatting *beyond* the one documented minor-units
  rounding rule (e.g. locale-specific thousands separators) remains
  unspecified — low-stakes, additive, deferred without blocking this
  checkpoint.
- Compose/Dockerfile mount configuration for the loader's startup paths.
- The production `search_stays` implementation, MCP tool registration, and
  wiring into `phase2/serving/` — all explicitly out of scope, Checkpoint C.1+.
