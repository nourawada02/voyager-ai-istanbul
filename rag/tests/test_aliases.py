"""Tests for the POI alias catalog -- resolving aliases/transliterations
to canonical poi_id without duplicating district truth."""

from __future__ import annotations

import json
from pathlib import Path

from rag.poi_catalog import POIS, all_aliases, by_id


def test_every_poi_district_id_is_a_real_canonical_district():
    registry_path = (
        Path(__file__).resolve().parent.parent.parent
        / "contracts" / "examples" / "valid" / "IstanbulDistrictRegistry.json"
    )
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    canonical_ids = {d["district_id"] for d in registry["districts"]}
    for poi in POIS:
        assert poi.district_id in canonical_ids, poi.poi_id


def test_alias_catalog_resolves_transliterations_to_canonical_poi_id():
    aliases = all_aliases()
    assert aliases["ayasofya"] == "poi_hagia_sophia"
    assert aliases["آيا صوفيا"] == "poi_hagia_sophia"
    assert aliases["hagia sofia"] == "poi_hagia_sophia"
    assert aliases["kapalıçarşı"] == "poi_grand_bazaar"
    assert aliases["البازار الكبير"] == "poi_grand_bazaar"


def test_alias_lookup_is_case_insensitive():
    aliases = all_aliases()
    assert aliases["AYASOFYA".casefold()] == "poi_hagia_sophia"


def test_by_id_raises_for_unknown_poi():
    import pytest

    with pytest.raises(KeyError):
        by_id("poi_does_not_exist")


def test_no_duplicate_alias_maps_to_two_different_pois():
    """A real integrity check: if two POIs accidentally shared an alias
    string, all_aliases() would silently let the later one win -- assert
    every raw (poi, alias) pair is unique so that can never happen
    unnoticed."""
    seen: dict[str, str] = {}
    for poi in POIS:
        for label in (poi.name, *poi.aliases):
            key = label.casefold()
            assert key not in seen or seen[key] == poi.poi_id, f"{label!r} claimed by both {seen.get(key)} and {poi.poi_id}"
            seen[key] = poi.poi_id
