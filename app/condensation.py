"""冷凝风险（condensation risk）计算。

按 **Magnus 公式**由温度、相对湿度逐点计算露点：

    γ  = ln(RH / 100) + 17.62·T / (243.12 + T)
    Td = 243.12·γ / (17.62 − γ)

**冷凝裕量 = 给定表面温度 − 露点**；裕量 ≤ ``margin_threshold_c`` 视为
风险（等于阈值也算）。相邻有效读数之间对裕量做**线性插值**，精确求出
进入与离开风险阈值的时刻，不做阶梯近似：

- 湿度为 0 的读数无法取对数、露点无定义：该点两侧区间一律**断开**；
- 相邻有效读数间隔**大于** ``max_gap_seconds`` 时曲线断开（恰好等于
  仍视为连续），未知区间不产生风险窗口；
- 窗口只在同一段连续曲线内相接时合并，**绝不跨断档合并**；
- 只返回持续**至少** ``min_duration_seconds`` 的风险窗口；孤立有效点
  （两侧区间均断开）不构成区间，不产生窗口。

全部计算确定性：相同输入（读数、表面温度、阈值、参数）恒得相同窗口
序列。
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from typing import Protocol

# Magnus 公式系数（温度 °C，湿度 %RH）
MAGNUS_A = 17.62
MAGNUS_B = 243.12


class _SampleLike(Protocol):
    ts: dt.datetime
    temperature: float
    humidity: float


def dew_point_c(temperature: float, humidity: float) -> float:
    """Magnus 公式露点（°C）；``humidity`` 必须大于 0（调用方保证）。"""
    gamma = math.log(humidity / 100.0) + MAGNUS_A * temperature / (
        MAGNUS_B + temperature
    )
    return MAGNUS_B * gamma / (MAGNUS_A - gamma)


@dataclass(frozen=True)
class RiskWindow:
    """一段持续的风险窗口（冷凝裕量 ≤ 阈值的连续区间）。"""

    start_ts: dt.datetime
    end_ts: dt.datetime
    duration_seconds: float
    min_margin_c: float  # 窗口内最低冷凝裕量（°C，必为窗口内某读数的裕量）
    min_margin_ts: dt.datetime


@dataclass(frozen=True)
class CondensationRisk:
    """冷凝风险计算结果（窗口按开始时间升序）。"""

    valid_sample_count: int  # 湿度 > 0、参与露点计算的读数数量
    windows: tuple[RiskWindow, ...]


def _cross_ts(t0: dt.datetime, t1: dt.datetime, frac: float) -> dt.datetime:
    """插值出交点时刻；端点分数直接返回原对象，保证相邻区间精确相接。"""
    if frac <= 0.0:
        return t0
    if frac >= 1.0:
        return t1
    return t0 + dt.timedelta(seconds=frac * (t1 - t0).total_seconds())


def _risk_fractions(
    m0: float, m1: float, threshold: float
) -> tuple[float, float] | None:
    """单采样区间的风险端点分数 ``(f0, f1)``；无风险返回 ``None``。

    裕量沿区间线性变化，风险为 m ≤ threshold 的部分；恰好一端在险时
    按线性插值求交点分数（交点处裕量恰为阈值）。
    """
    r0 = m0 <= threshold
    r1 = m1 <= threshold
    if not r0 and not r1:
        return None
    if r0 and r1:
        return 0.0, 1.0
    f = (threshold - m0) / (m1 - m0)  # 线性交点：m(f) = threshold
    return (0.0, f) if r0 else (f, 1.0)


def compute_condensation_risk(
    samples: list[_SampleLike],
    surface_temp_c: float,
    margin_threshold_c: float,
    max_gap_seconds: float,
    min_duration_seconds: float,
) -> CondensationRisk:
    """计算冷凝风险窗口序列（按开始时间升序）。

    samples 无需预先排序；同一展柜的读数应已按 (device_sn, ts) 合并。
    """
    ordered = sorted(samples, key=lambda s: s.ts)
    times = [s.ts for s in ordered]

    # 有效点：湿度 > 0；湿度为 0 时露点无定义，该点两侧区间断开
    margins: list[float | None] = []
    valid_count = 0
    for s in ordered:
        if s.humidity > 0.0:
            valid_count += 1
            margins.append(
                surface_temp_c - dew_point_c(s.temperature, s.humidity)
            )
        else:
            margins.append(None)

    windows: list[RiskWindow] = []
    open_start: dt.datetime | None = None
    open_end: dt.datetime | None = None
    open_min = math.inf
    open_min_ts: dt.datetime | None = None

    def close_window() -> None:
        nonlocal open_start, open_end, open_min, open_min_ts
        if open_start is None or open_end is None or open_min_ts is None:
            return
        duration = (open_end - open_start).total_seconds()
        if duration >= min_duration_seconds:
            windows.append(
                RiskWindow(
                    start_ts=open_start,
                    end_ts=open_end,
                    duration_seconds=duration,
                    min_margin_c=open_min,
                    min_margin_ts=open_min_ts,
                )
            )
        open_start = None
        open_end = None
        open_min = math.inf
        open_min_ts = None

    for i in range(len(ordered) - 1):
        m0, m1 = margins[i], margins[i + 1]
        if m0 is None or m1 is None:
            close_window()  # 湿度为 0：断开
            continue
        span = (times[i + 1] - times[i]).total_seconds()
        if span <= 0.0 or span > max_gap_seconds:
            close_window()  # 采样间隔超过 max_gap：断开
            continue
        fracs = _risk_fractions(m0, m1, margin_threshold_c)
        if fracs is None:
            close_window()
            continue
        f0, f1 = fracs
        a = _cross_ts(times[i], times[i + 1], f0)
        b = _cross_ts(times[i], times[i + 1], f1)
        if open_start is not None and a <= open_end:
            # 与未闭合窗口相接（共享读数点同样在险）：合并延伸；
            # 断档两侧的子区间在时间上必然严格分离，不会走到这里
            open_end = max(open_end, b)
        else:
            close_window()
            open_start, open_end = a, b
            open_min = math.inf
            open_min_ts = None
        # 窗口最低裕量必落在窗口内的读数上（交点处裕量恒为阈值）；
        # 并列时保留更早出现者（严格小于才更新）
        for m, t in ((m0, times[i]), (m1, times[i + 1])):
            if m <= margin_threshold_c and m < open_min:
                open_min = m
                open_min_ts = t
    close_window()

    return CondensationRisk(
        valid_sample_count=valid_count,
        windows=tuple(windows),
    )
