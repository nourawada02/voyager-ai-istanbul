"""Narrowly scoped OPTIONAL live smoke test for the real Open-Meteo
adapter (Checkpoint Phase 4 C.1, Step 14). Disabled during normal
`pytest` runs -- enabled only by setting the explicit environment flag
below. Makes exactly one bounded request chain (geocode Istanbul, then
fetch current weather for it) -- never a loop, never repeated polling.
No credential of any kind is used or could leak: Open-Meteo's
non-commercial API requires none. Prints only non-secret, already-public
operational fields (provider name, status, timezone, a few weather
values) -- never a raw response body, and never writes any file.

Enable with:
    VOYAGER_LIVE_WEATHER_GATE=1 python -m pytest providers/tests/test_weather_openmeteo_live.py -v -s
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest

from providers.tests.conftest import make_validator
from providers.weather import WeatherQuery
from providers.weather_openmeteo import OpenMeteoWeatherProvider

LIVE_GATE_ENV_VAR = "VOYAGER_LIVE_WEATHER_GATE"

pytestmark = pytest.mark.skipif(
    os.environ.get(LIVE_GATE_ENV_VAR) != "1",
    reason=f"live Open-Meteo network gate is disabled by default; set {LIVE_GATE_ENV_VAR}=1 to enable",
)


def test_live_open_meteo_istanbul_current_weather(envelope_validator, registry):
    provider = OpenMeteoWeatherProvider()  # real transport, real clock -- the only test file in this project that uses them
    today = datetime.now(timezone.utc).date().isoformat()
    query = WeatherQuery(location="Istanbul", timezone="", date_from=today, date_to=today)

    envelope = provider.fetch_weather(query)

    envelope_errors = list(envelope_validator.iter_errors(envelope))
    assert not envelope_errors, [e.message for e in envelope_errors]
    result_validator = make_validator("WeatherResult", registry)
    result_errors = list(result_validator.iter_errors(envelope["result"]))
    assert not result_errors, [e.message for e in result_errors]

    assert envelope["provider"] == "open-meteo"
    assert envelope["status"] == "success", envelope.get("quality", {}).get("assumptions")
    assert envelope["retrieved_at"]
    assert envelope["source_urls"]
    assert envelope["result"]["timezone"] == "Europe/Istanbul"
    assert envelope["result"]["kind"] == "current_observation"
    assert envelope["result"]["attribution"]["license"] == "CC BY 4.0"

    print(
        "LIVE OPEN-METEO GATE: "
        f"provider={envelope['provider']} status={envelope['status']} "
        f"location={envelope['result']['location']} timezone={envelope['result']['timezone']} "
        f"issued_at={envelope['result']['issued_at']} "
        f"condition={envelope['result']['observation']['condition']} "
        f"temperature_c={envelope['result']['observation']['temperature']} "
        f"attribution={envelope['result']['attribution']['notice']}"
    )
