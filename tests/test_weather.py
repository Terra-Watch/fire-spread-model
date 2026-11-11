import httpx
import pandas as pd

from firespread import weather


def stub(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_parses_hourly_and_caches(tmp_path, monkeypatch):
    monkeypatch.setattr(weather, "CACHE", tmp_path)
    calls = []

    def handler(req):
        calls.append(str(req.url))
        return httpx.Response(200, json={"hourly": {"time": ["2024-03-01T00:00", "2024-03-01T01:00"], **{v: [1.5, 2.5] for v in weather.VARS}}})

    with stub(handler) as c:
        a = weather.fetch("hist", 21.5, 86.3, "2024-03-01", "2024-03-01", client=c, pause=0)
        b = weather.fetch("hist", 21.5, 86.3, "2024-03-01", "2024-03-01", client=c, pause=0)
    assert len(calls) == 1 and a.equals(b)
    assert list(a.columns) == weather.VARS and a.index[0] == pd.Timestamp("2024-03-01 00:00", tz="UTC") and a.wind_speed_10m.tolist() == [1.5, 2.5]


def test_previous_day_forecast_reads_the_suffixed_columns(tmp_path, monkeypatch):
    monkeypatch.setattr(weather, "CACHE", tmp_path)
    seen = {}

    def handler(req):
        seen["hourly"] = req.url.params["hourly"]
        return httpx.Response(200, json={"hourly": {"time": ["2024-03-01T00:00"], **{f"{v}_previous_day1": [None] for v in weather.VARS}}})

    with stub(handler) as c:
        w = weather.fetch("prev1", 21.5, 86.3, "2024-03-01", "2024-03-01", client=c, pause=0)
    assert "wind_speed_10m_previous_day1" in seen["hourly"] and w.wind_speed_10m.isna().all()


def test_a_source_that_does_not_cover_the_dates_gives_an_empty_frame(tmp_path, monkeypatch):
    monkeypatch.setattr(weather, "CACHE", tmp_path)
    with stub(lambda req: httpx.Response(400, json={"reason": "out of range"})) as c:
        w = weather.fetch("prev1", 21.5, 86.3, "2019-03-01", "2019-03-02", client=c, pause=0)
    assert w.empty and list(w.columns) == weather.VARS
