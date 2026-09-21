"""冷凝风险：Magnus 露点/裕量插值窗口单测 + 接口集成测试（内存 SQLite）。"""

from __future__ import annotations

import datetime as dt
import os

os.environ["CASE_DB_URL"] = "sqlite:///:memory:"
os.environ["CASE_PENALTY"] = "4.0"

import pytest
from fastapi.testclient import TestClient

from app.condensation import compute_condensation_risk, dew_point_c
from app.database import Base, engine, init_db
from app.main import app
from app.segmentation import Sample

BASE = dt.datetime(2026, 9, 1, tzinfo=dt.timezone.utc)


@pytest.fixture()
def client():
    Base.metadata.drop_all(bind=engine)
    init_db()
    with TestClient(app) as c:
        yield c
    Base.metadata.drop_all(bind=engine)


def _sample(minutes: float, t: float, h: float = 100.0) -> Sample:
    return Sample(BASE + dt.timedelta(minutes=minutes), t, h)


def _reading(sn, ts, t, h):
    return {"device_sn": sn, "ts": ts, "temperature": t, "humidity": h}


def _ts(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value)


def _risk(samples, *, surface=10.0, threshold=0.0, max_gap=600.0, min_duration=0.0):
    return compute_condensation_risk(
        samples,
        surface_temp_c=surface,
        margin_threshold_c=threshold,
        max_gap_seconds=max_gap,
        min_duration_seconds=min_duration,
    )


# ---------- 纯算法：Magnus 露点 ----------


def test_dew_point_saturated_equals_temperature():
    # RH=100% 时露点等于温度本身
    assert dew_point_c(20.0, 100.0) == pytest.approx(20.0)
    assert dew_point_c(0.0, 100.0) == pytest.approx(0.0)
    assert dew_point_c(-10.0, 100.0) == pytest.approx(-10.0)


def test_dew_point_reference_values():
    # 公认参考值：20°C / 50%RH → 约 9.26°C；20°C / 95%RH → 约 19.17°C
    assert dew_point_c(20.0, 50.0) == pytest.approx(9.2552, abs=1e-3)
    assert dew_point_c(20.0, 95.0) == pytest.approx(19.1736, abs=1e-3)
    assert dew_point_c(30.0, 60.0) == pytest.approx(21.3854, abs=1e-3)


# ---------- 纯算法：裕量线性插值与风险窗口 ----------
# 以下用例湿度均为 100%（露点 = 温度），表面温度 10°C、阈值 0：
# 裕量 = 10 − T，T=12 时裕量 −2（在险），T=8 时裕量 +2（安全）。


def test_window_entry_exit_exact_interpolation():
    result = _risk(
        [_sample(0, 8.0), _sample(10, 12.0), _sample(20, 12.0), _sample(30, 8.0)]
    )
    assert result.valid_sample_count == 4
    assert len(result.windows) == 1
    w = result.windows[0]
    # 裕量 +2 → −2：中点进入；−2 → +2：中点离开
    assert w.start_ts == BASE + dt.timedelta(minutes=5)
    assert w.end_ts == BASE + dt.timedelta(minutes=25)
    assert w.duration_seconds == 1200.0
    # 最低裕量并列时取最早读数
    assert w.min_margin_c == pytest.approx(-2.0)
    assert w.min_margin_ts == BASE + dt.timedelta(minutes=10)


def test_margin_equal_threshold_is_risk():
    # 裕量恰为阈值（≤ 语义）也算风险
    result = _risk([_sample(0, 10.0), _sample(10, 10.0)])
    assert len(result.windows) == 1
    w = result.windows[0]
    assert w.start_ts == BASE
    assert w.end_ts == BASE + dt.timedelta(minutes=10)
    assert w.duration_seconds == 600.0
    assert w.min_margin_c == pytest.approx(0.0)
    assert w.min_margin_ts == BASE


def test_min_duration_filters_windows():
    samples = [_sample(0, 8.0), _sample(10, 12.0), _sample(20, 12.0), _sample(30, 8.0)]
    # 窗口持续 1200 秒：恰好等于 min_duration 保留，超过则剔除
    assert len(_risk(samples, min_duration=1200.0).windows) == 1
    assert _risk(samples, min_duration=1201.0).windows == ()


def test_gap_breaks_windows_not_merged():
    # 10 → 300 分钟间隔超过 max_gap：两个窗口独立存在，绝不跨断档合并
    samples = [_sample(0, 12.0), _sample(10, 12.0), _sample(300, 12.0), _sample(310, 12.0)]
    result = _risk(samples)
    assert len(result.windows) == 2
    first, second = result.windows
    assert first.start_ts == BASE
    assert first.end_ts == BASE + dt.timedelta(minutes=10)
    assert second.start_ts == BASE + dt.timedelta(minutes=300)
    assert second.end_ts == BASE + dt.timedelta(minutes=310)
    for w in result.windows:
        assert w.duration_seconds == 600.0
        assert w.min_margin_c == pytest.approx(-2.0)
    # 恰好等于 max_gap 仍视为连续：合成一个窗口
    joined = _risk(
        [_sample(0, 12.0), _sample(10, 12.0)], max_gap=600.0
    )
    assert len(joined.windows) == 1
    assert joined.windows[0].duration_seconds == 600.0


def test_zero_humidity_breaks_curve():
    # 湿度为 0 露点无定义：其两侧区间全部断开，孤立有效点不产生窗口
    result = _risk([_sample(0, 12.0), _sample(10, 12.0, 0.0), _sample(20, 12.0)])
    assert result.valid_sample_count == 2
    assert result.windows == ()


def test_zero_humidity_separates_windows_not_merged():
    samples = [
        _sample(0, 12.0),
        _sample(10, 12.0),
        _sample(20, 12.0, 0.0),
        _sample(30, 12.0),
        _sample(40, 12.0),
    ]
    result = _risk(samples)
    assert result.valid_sample_count == 4
    assert len(result.windows) == 2
    assert result.windows[0].end_ts == BASE + dt.timedelta(minutes=10)
    assert result.windows[1].start_ts == BASE + dt.timedelta(minutes=30)


def test_isolated_point_no_window():
    # 单点无法构成区间
    assert _risk([_sample(0, 12.0)]).windows == ()
    # 两侧区间均断开的孤立有效点同样不产生窗口
    samples = [_sample(0, 12.0), _sample(300, 12.0)]
    assert _risk(samples).windows == ()


def test_touching_threshold_zero_duration_window():
    # 裕量 2 → 0 → 2：仅在中间读数瞬间触及阈值，零时长窗口
    result = _risk([_sample(0, 8.0), _sample(10, 10.0), _sample(20, 8.0)])
    assert len(result.windows) == 1
    w = result.windows[0]
    assert w.start_ts == w.end_ts == BASE + dt.timedelta(minutes=10)
    assert w.duration_seconds == 0.0
    assert w.min_margin_c == pytest.approx(0.0)
    # min_duration > 0 时被剔除
    assert _risk(
        [_sample(0, 8.0), _sample(10, 10.0), _sample(20, 8.0)], min_duration=1.0
    ).windows == ()


def test_no_risk_no_windows():
    assert _risk([_sample(0, 8.0), _sample(10, 9.0)]).windows == ()


def test_custom_threshold_and_surface():
    # 表面 15°C、阈值 1.0：T=12（露点 12）裕量 3 安全；T=16 裕量 −1 在险
    result = _risk(
        [_sample(0, 12.0), _sample(10, 16.0)], surface=15.0, threshold=1.0
    )
    assert len(result.windows) == 1
    w = result.windows[0]
    # 裕量 3 → −1，交于裕量 = 1 处：f = (1 − 3) / (−1 − 3) = 0.5
    assert w.start_ts == BASE + dt.timedelta(minutes=5)
    assert w.end_ts == BASE + dt.timedelta(minutes=10)
    assert w.min_margin_c == pytest.approx(-1.0)
    assert w.min_margin_ts == BASE + dt.timedelta(minutes=10)


def test_unsorted_input_deterministic():
    samples = [
        _sample(0, 8.0),
        _sample(10, 12.0),
        _sample(20, 12.0),
        _sample(30, 8.0),
        _sample(300, 12.0),
        _sample(310, 12.0),
    ]
    ordered = _risk(samples)
    shuffled = _risk(samples[::-1])
    assert ordered == shuffled


# ---------- 接口集成 ----------


def _ingest(client, sn, rows):
    r = client.post("/api/v1/readings/batch", json={"readings": rows})
    assert r.status_code == 200, r.text


def _ingest_case_c1(client):
    # 08:00→08:30（+08:00），湿度 100%：露点 = 温度，表面 10°C
    _ingest(client, "CASE-C1", [
        _reading("CASE-C1", "2026-09-01T08:00:00+08:00", 8.0, 100.0),
        _reading("CASE-C1", "2026-09-01T08:10:00+08:00", 12.0, 100.0),
        _reading("CASE-C1", "2026-09-01T08:20:00+08:00", 12.0, 100.0),
        _reading("CASE-C1", "2026-09-01T08:30:00+08:00", 8.0, 100.0),
    ])


def test_condensation_endpoint_basic(client):
    _ingest_case_c1(client)
    r = client.get(
        "/api/v1/cases/CASE-C1/condensation-risk", params={"surface_temp_c": 10}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["device_sn"] == "CASE-C1"
    assert body["surface_temp_c"] == 10.0
    assert body["margin_threshold_c"] == 0.0
    assert body["max_gap_seconds"] == 600.0
    assert body["min_duration_seconds"] == 0.0
    assert body["sample_count"] == 4
    assert body["valid_sample_count"] == 4
    assert body["window_count"] == 1

    w = body["windows"][0]
    assert _ts(w["start_ts"]) == _ts("2026-09-01T00:05:00+00:00")
    assert _ts(w["end_ts"]) == _ts("2026-09-01T00:25:00+00:00")
    assert w["duration_seconds"] == pytest.approx(1200.0)
    assert w["min_margin_c"] == pytest.approx(-2.0)
    assert _ts(w["min_margin_ts"]) == _ts("2026-09-01T00:10:00+00:00")


def test_condensation_min_duration_param(client):
    _ingest_case_c1(client)
    url = "/api/v1/cases/CASE-C1/condensation-risk"
    body = client.get(
        url, params={"surface_temp_c": 10, "min_duration_seconds": 1200}
    ).json()
    assert body["window_count"] == 1
    body = client.get(
        url, params={"surface_temp_c": 10, "min_duration_seconds": 1201}
    ).json()
    assert body["window_count"] == 0
    assert body["windows"] == []


def test_condensation_max_gap_controls_break(client):
    _ingest(client, "CASE-C2", [
        _reading("CASE-C2", "2026-09-01T08:00:00+08:00", 12.0, 100.0),
        _reading("CASE-C2", "2026-09-01T08:10:00+08:00", 12.0, 100.0),
        _reading("CASE-C2", "2026-09-01T12:00:00+08:00", 12.0, 100.0),
        _reading("CASE-C2", "2026-09-01T12:10:00+08:00", 12.0, 100.0),
    ])
    url = "/api/v1/cases/CASE-C2/condensation-risk"
    # 默认 max_gap=600：08:10→12:00 断开，两个窗口
    body = client.get(url, params={"surface_temp_c": 10}).json()
    assert body["window_count"] == 2
    assert _ts(body["windows"][0]["end_ts"]) == _ts("2026-09-01T00:10:00+00:00")
    assert _ts(body["windows"][1]["start_ts"]) == _ts("2026-09-01T04:00:00+00:00")
    # max_gap=14400：整段连续，单个窗口（不跨断档合并的规则不再触发）
    body = client.get(
        url, params={"surface_temp_c": 10, "max_gap_seconds": 14400}
    ).json()
    assert body["window_count"] == 1
    assert body["windows"][0]["duration_seconds"] == pytest.approx(15000.0)


def test_condensation_zero_humidity_via_api(client):
    _ingest(client, "CASE-C3", [
        _reading("CASE-C3", "2026-09-01T08:00:00+08:00", 12.0, 100.0),
        _reading("CASE-C3", "2026-09-01T08:10:00+08:00", 12.0, 0.0),
        _reading("CASE-C3", "2026-09-01T08:20:00+08:00", 12.0, 100.0),
    ])
    body = client.get(
        "/api/v1/cases/CASE-C3/condensation-risk", params={"surface_temp_c": 10}
    ).json()
    assert body["sample_count"] == 3
    assert body["valid_sample_count"] == 2
    assert body["window_count"] == 0
    assert body["windows"] == []


def test_condensation_start_end_filter_scopes_samples(client):
    _ingest_case_c1(client)
    r = client.get(
        "/api/v1/cases/CASE-C1/condensation-risk",
        params={"surface_temp_c": 10, "start": "2026-09-01T08:10:00+08:00"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["sample_count"] == 3
    assert body["start"].startswith("2026-09-01T00:10:00")
    # 窗口内曲线仅从 08:10 开始：窗口 [08:10, 08:25]
    w = body["windows"][0]
    assert _ts(w["start_ts"]) == _ts("2026-09-01T00:10:00+00:00")
    assert _ts(w["end_ts"]) == _ts("2026-09-01T00:25:00+00:00")


def test_condensation_empty_range_returns_empty(client):
    _ingest_case_c1(client)
    r = client.get(
        "/api/v1/cases/CASE-C1/condensation-risk",
        params={
            "surface_temp_c": 10,
            "start": "2030-01-01T00:00:00Z",
            "end": "2030-02-01T00:00:00Z",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["windows"] == []
    assert body["window_count"] == 0
    assert body["sample_count"] == 0
    assert body["valid_sample_count"] == 0


def test_condensation_unknown_case_404(client):
    r = client.get("/api/v1/cases/NOPE/condensation-risk", params={"surface_temp_c": 10})
    assert r.status_code == 404


def test_condensation_validation_errors(client):
    _ingest_case_c1(client)
    url = "/api/v1/cases/CASE-C1/condensation-risk"
    # surface_temp_c 必填
    assert client.get(url).status_code == 422
    # max_gap_seconds 必须为正
    assert client.get(url, params={"surface_temp_c": 10, "max_gap_seconds": 0}).status_code == 422
    assert client.get(url, params={"surface_temp_c": 10, "max_gap_seconds": -5}).status_code == 422
    # min_duration_seconds 必须非负
    assert client.get(url, params={"surface_temp_c": 10, "min_duration_seconds": -1}).status_code == 422
    # start 晚于 end
    assert (
        client.get(
            url,
            params={
                "surface_temp_c": 10,
                "start": "2030-01-01T00:00:00Z",
                "end": "2029-01-01T00:00:00Z",
            },
        ).status_code
        == 422
    )
    # 表面温度 / 阈值超出量程
    assert client.get(url, params={"surface_temp_c": 150}).status_code == 422
    assert (
        client.get(url, params={"surface_temp_c": 10, "margin_threshold_c": 200}).status_code
        == 422
    )


def test_condensation_deterministic(client):
    _ingest_case_c1(client)
    params = {"surface_temp_c": 10, "margin_threshold_c": 0.5, "min_duration_seconds": 60}
    first = client.get("/api/v1/cases/CASE-C1/condensation-risk", params=params).json()
    second = client.get("/api/v1/cases/CASE-C1/condensation-risk", params=params).json()
    assert first == second
    starts = [w["start_ts"] for w in first["windows"]]
    assert starts == sorted(starts)


def test_condensation_in_openapi(client):
    r = client.get("/openapi.json")
    assert r.status_code == 200
    paths = r.json()["paths"]
    assert "/api/v1/cases/{device_sn}/condensation-risk" in paths
