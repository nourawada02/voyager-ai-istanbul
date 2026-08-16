"""Small canonical POI catalog for the Phase 3 corpus's own attraction
documents. Every entry validates against the already-accepted
`contracts/POI.schema.json` (reused unchanged) and reuses `district_id`
values from the existing 39-district `IstanbulDistrictRegistry` -- no
district truth is duplicated here, only referenced. Coordinates are
public, well-known landmark coordinates, not derived from any corpus
document text.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CatalogPOI:
    poi_id: str
    name: str
    aliases: tuple[str, ...]
    district_id: str
    side: str
    lat: float
    lon: float
    category: str

    def to_dict(self) -> dict:
        return {
            "schema_version": "1.0.0",
            "poi_id": self.poi_id,
            "name": self.name,
            "aliases": list(self.aliases),
            "district_id": self.district_id,
            "side": self.side,
            "coordinates": {"lat": self.lat, "lon": self.lon},
            "category": self.category,
        }


POIS: tuple[CatalogPOI, ...] = (
    CatalogPOI(
        poi_id="poi_hagia_sophia",
        name="Hagia Sophia",
        aliases=("Ayasofya", "آيا صوفيا", "Hagia Sofia", "Aya Sofya"),
        district_id="district_fatih",
        side="european",
        lat=41.0086,
        lon=28.9802,
        category="historic_religious_site",
    ),
    CatalogPOI(
        poi_id="poi_topkapi_palace",
        name="Topkapı Palace",
        aliases=("Topkapi Palace", "Topkapı Sarayı", "طوب قابي سراي", "Topkapi Sarayi"),
        district_id="district_fatih",
        side="european",
        lat=41.0115,
        lon=28.9833,
        category="historic_palace",
    ),
    CatalogPOI(
        poi_id="poi_grand_bazaar",
        name="Grand Bazaar",
        aliases=("Kapalıçarşı", "Kapali Carsi", "البازار الكبير", "Covered Bazaar"),
        district_id="district_fatih",
        side="european",
        lat=41.0106,
        lon=28.9681,
        category="historic_market",
    ),
    CatalogPOI(
        poi_id="poi_galata_tower",
        name="Galata Tower",
        aliases=("Galata Kulesi", "برج غالطة"),
        district_id="district_beyoglu",
        side="european",
        lat=41.0256,
        lon=28.9741,
        category="historic_tower",
    ),
)


def by_id(poi_id: str) -> CatalogPOI:
    for poi in POIS:
        if poi.poi_id == poi_id:
            return poi
    raise KeyError(poi_id)


def all_aliases() -> dict[str, str]:
    """Maps every alias (and the canonical name itself), casefolded, to
    poi_id -- the alias catalog required by the corpus boundary. Never a
    second district table: district_id is looked up on the POI, not
    re-derived."""
    out: dict[str, str] = {}
    for poi in POIS:
        out[poi.name.casefold()] = poi.poi_id
        for alias in poi.aliases:
            out[alias.casefold()] = poi.poi_id
    return out
