"""
Tests for TemporalReasoner — time-aware retrieval scoring
Inspired by Mem0 v3 Temporal Reasoning (shipped May 13, 2026)
"""
import time
import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

# === Models ===

class AnchorType(Enum):
    PAST_VAGUE = "PAST_VAGUE"
    PAST_RELATIVE = "PAST_RELATIVE"
    PAST_DISTANT = "PAST_DISTANT"
    PRESENT = "PRESENT"
    RECENT = "RECENT"
    FUTURE_RELATIVE = "FUTURE_RELATIVE"
    FUTURE_NEAR = "FUTURE_NEAR"
    FUTURE_VAGUE = "FUTURE_VAGUE"

class TemporalDirection(Enum):
    PAST = "PAST"
    PRESENT = "PRESENT"
    FUTURE = "FUTURE"

@dataclass
class TemporalAnchor:
    anchor_type: AnchorType
    anchor_time_ms: int
    confidence: float
    matched_text: str

@dataclass
class TemporalContext:
    has_temporal_intent: bool
    primary_anchor: Optional[TemporalAnchor]
    all_anchors: List[TemporalAnchor]
    reference_date: int
    temporal_direction: TemporalDirection


class TemporalReasoner:
    def __init__(self, half_life_days=30.0, min_score=0.1):
        self.half_life = half_life_days
        self.min_score = min_score
        
        self.past_patterns = [
            (re.compile(r'以前|之前|过去|曾经|原来'), AnchorType.PAST_VAGUE, -90.0),
            (re.compile(r'上周'), AnchorType.PAST_RELATIVE, -7.0),
            (re.compile(r'上个月'), AnchorType.PAST_RELATIVE, -30.0),
            (re.compile(r'去年'), AnchorType.PAST_RELATIVE, -365.0),
            (re.compile(r'前天'), AnchorType.PAST_RELATIVE, -2.0),
            (re.compile(r'昨天'), AnchorType.PAST_RELATIVE, -1.0),
            (re.compile(r'(\d+)天前'), AnchorType.PAST_RELATIVE, None),
            (re.compile(r'(\d+)个?月前'), AnchorType.PAST_RELATIVE, None),
            (re.compile(r'(\d+)年前'), AnchorType.PAST_RELATIVE, None),
            (re.compile(r'小时候|年轻的时候'), AnchorType.PAST_DISTANT, -3650.0),
        ]
        self.present_patterns = [
            (re.compile(r'现在|目前|当前|此刻'), AnchorType.PRESENT, 0.0),
            (re.compile(r'最近|近期|这几天'), AnchorType.RECENT, -3.0),
            (re.compile(r'今天'), AnchorType.PRESENT, 0.0),
        ]
        self.future_patterns = [
            (re.compile(r'明天'), AnchorType.FUTURE_RELATIVE, 1.0),
            (re.compile(r'后天'), AnchorType.FUTURE_RELATIVE, 2.0),
            (re.compile(r'下周'), AnchorType.FUTURE_RELATIVE, 7.0),
            (re.compile(r'下个月'), AnchorType.FUTURE_RELATIVE, 30.0),
            (re.compile(r'明年'), AnchorType.FUTURE_RELATIVE, 365.0),
            (re.compile(r'(\d+)天后'), AnchorType.FUTURE_RELATIVE, None),
            (re.compile(r'即将|马上|快要|将要'), AnchorType.FUTURE_NEAR, 1.0),
            (re.compile(r'以后|将来|未来'), AnchorType.FUTURE_VAGUE, 30.0),
        ]
    
    def analyze(self, query, reference_date=None):
        reference_date = reference_date or int(time.time() * 1000)
        anchors = []
        DAY = 86400 * 1000
        
        for regex, atype, offset in self.past_patterns:
            m = regex.search(query)
            if not m:
                continue
            if offset is None:
                # Dynamic: extract number
                num = int(m.group(1))
                if '年' in m.group(): mult = -365.0
                elif '月' in m.group(): mult = -30.0
                else: mult = -1.0
                actual_offset = num * mult
            elif atype == AnchorType.PAST_VAGUE:
                actual_offset = -90.0
            elif atype == AnchorType.PAST_DISTANT:
                actual_offset = -3650.0
            else:
                actual_offset = offset
            anchors.append(TemporalAnchor(
                atype, int(reference_date + actual_offset * DAY), 0.8, m.group()))
        
        for regex, atype, offset in self.present_patterns:
            m = regex.search(query)
            if m:
                anchors.append(TemporalAnchor(
                    atype, int(reference_date + offset * DAY), 0.9, m.group()))
        
        for regex, atype, offset in self.future_patterns:
            m = regex.search(query)
            if not m:
                continue
            if offset is None:
                num = int(m.group(1))
                actual_offset = num * 1.0  # days
            else:
                actual_offset = offset
            anchors.append(TemporalAnchor(
                atype, int(reference_date + actual_offset * DAY), 0.8, m.group()))
        
        primary = max(anchors, key=lambda a: a.confidence) if anchors else None
        
        if primary:
            if primary.anchor_type in (AnchorType.PAST_VAGUE, AnchorType.PAST_RELATIVE, AnchorType.PAST_DISTANT):
                direction = TemporalDirection.PAST
            elif primary.anchor_type in (AnchorType.FUTURE_RELATIVE, AnchorType.FUTURE_NEAR, AnchorType.FUTURE_VAGUE):
                direction = TemporalDirection.FUTURE
            else:
                direction = TemporalDirection.PRESENT
        else:
            direction = TemporalDirection.PRESENT
        
        return TemporalContext(
            has_temporal_intent=len(anchors) > 0,
            primary_anchor=primary,
            all_anchors=anchors,
            reference_date=reference_date,
            temporal_direction=direction
        )
    
    def compute_temporal_score(self, memory_timestamp, temporal_context):
        anchor = temporal_context.primary_anchor.anchor_time_ms if temporal_context.primary_anchor else temporal_context.reference_date
        
        distance_ms = abs(memory_timestamp - anchor)
        distance_days = distance_ms / (86400.0 * 1000.0)
        
        decay = math.exp(-0.693 * distance_days / self.half_life)
        
        # Direction bonus
        bonus = 0.0
        if temporal_context.temporal_direction == TemporalDirection.PAST:
            if temporal_context.primary_anchor and memory_timestamp <= temporal_context.primary_anchor.anchor_time_ms:
                bonus = 0.1
        elif temporal_context.temporal_direction == TemporalDirection.FUTURE:
            if memory_timestamp > temporal_context.reference_date:
                bonus = 0.1
        
        return max(self.min_score, min(1.0, decay + bonus))


# ============================
# Tests
# ============================

def test_no_temporal_intent():
    """Queries without time words default to PRESENT"""
    tr = TemporalReasoner()
    
    ctx = tr.analyze('我叫什么名字')
    assert not ctx.has_temporal_intent
    assert ctx.temporal_direction == TemporalDirection.PRESENT
    assert ctx.primary_anchor is None
    print("✓ test_no_temporal_intent")


def test_past_vague():
    """'以前/之前' triggers past anchor ~90 days ago"""
    tr = TemporalReasoner()
    now = int(time.time() * 1000)
    
    ctx = tr.analyze('我以前住在哪里', reference_date=now)
    assert ctx.has_temporal_intent
    assert ctx.temporal_direction == TemporalDirection.PAST
    assert ctx.primary_anchor.anchor_type == AnchorType.PAST_VAGUE
    
    # Anchor should be ~90 days before reference
    expected = now - 90 * 86400 * 1000
    assert abs(ctx.primary_anchor.anchor_time_ms - expected) < 86400 * 1000  # within 1 day
    print(f"  '以前' → anchor ~90 days ago")
    print("✓ test_past_vague")


def test_past_relative():
    """'上周/去年' triggers specific relative anchors"""
    tr = TemporalReasoner()
    now = int(time.time() * 1000)
    DAY = 86400 * 1000
    
    ctx_week = tr.analyze('上周发生了什么', reference_date=now)
    assert ctx_week.temporal_direction == TemporalDirection.PAST
    expected_week = now - 7 * DAY
    assert abs(ctx_week.primary_anchor.anchor_time_ms - expected_week) < DAY
    
    ctx_year = tr.analyze('去年我做了什么', reference_date=now)
    assert ctx_year.temporal_direction == TemporalDirection.PAST
    expected_year = now - 365 * DAY
    assert abs(ctx_year.primary_anchor.anchor_time_ms - expected_year) < DAY
    
    print("✓ test_past_relative")


def test_past_dynamic():
    """'N天前/N个月前' extracts numeric offset"""
    tr = TemporalReasoner()
    now = int(time.time() * 1000)
    DAY = 86400 * 1000
    
    ctx = tr.analyze('3天前我说了什么', reference_date=now)
    assert ctx.has_temporal_intent
    expected = now - 3 * DAY
    assert abs(ctx.primary_anchor.anchor_time_ms - expected) < DAY
    
    ctx2 = tr.analyze('2个月前的事情', reference_date=now)
    assert ctx2.has_temporal_intent
    expected2 = now - 60 * DAY
    assert abs(ctx2.primary_anchor.anchor_time_ms - expected2) < 2 * DAY
    
    print("✓ test_past_dynamic")


def test_present():
    """'现在/今天' anchors to reference_date"""
    tr = TemporalReasoner()
    now = int(time.time() * 1000)
    
    ctx = tr.analyze('我现在住在哪里', reference_date=now)
    assert ctx.has_temporal_intent
    assert ctx.temporal_direction == TemporalDirection.PRESENT
    # Anchor should be at reference_date (±0)
    assert abs(ctx.primary_anchor.anchor_time_ms - now) < 86400 * 1000
    print("✓ test_present")


def test_recent():
    """'最近' anchors to ~3 days ago"""
    tr = TemporalReasoner()
    now = int(time.time() * 1000)
    DAY = 86400 * 1000
    
    ctx = tr.analyze('最近发生了什么', reference_date=now)
    assert ctx.has_temporal_intent
    # RECENT has higher confidence (0.9) than PAST patterns (0.8), so it wins
    assert ctx.primary_anchor.anchor_type == AnchorType.RECENT
    expected = now - 3 * DAY
    assert abs(ctx.primary_anchor.anchor_time_ms - expected) < DAY
    print("✓ test_recent")


def test_future():
    """'明天/下周' triggers future anchors"""
    tr = TemporalReasoner()
    now = int(time.time() * 1000)
    DAY = 86400 * 1000
    
    ctx = tr.analyze('明天有什么安排', reference_date=now)
    assert ctx.has_temporal_intent
    assert ctx.temporal_direction == TemporalDirection.FUTURE
    expected = now + 1 * DAY
    assert abs(ctx.primary_anchor.anchor_time_ms - expected) < DAY
    
    ctx2 = tr.analyze('下周要做什么', reference_date=now)
    assert ctx2.temporal_direction == TemporalDirection.FUTURE
    expected2 = now + 7 * DAY
    assert abs(ctx2.primary_anchor.anchor_time_ms - expected2) < DAY
    
    print("✓ test_future")


def test_scoring_past_query():
    """
    Past query: memories close to the anchor score higher than recent memories.
    
    Scenario: "我以前住在哪里" (anchor ~90 days ago)
    - Memory A: "住在北京" (100 days old) → CLOSE to anchor → HIGH score
    - Memory B: "搬到上海" (1 day old) → FAR from anchor → LOWER score
    """
    tr = TemporalReasoner()
    now = int(time.time() * 1000)
    DAY = 86400 * 1000
    
    ctx = tr.analyze('我以前住在哪里', reference_date=now)
    
    # Memory from 100 days ago (close to 90-day anchor)
    score_old = tr.compute_temporal_score(now - 100 * DAY, ctx)
    # Memory from 1 day ago (far from 90-day anchor)
    score_new = tr.compute_temporal_score(now - 1 * DAY, ctx)
    
    print(f"  Past query anchor ~90 days ago:")
    print(f"    100-day-old memory: score={score_old:.3f}")
    print(f"    1-day-old memory:   score={score_new:.3f}")
    
    assert score_old > score_new, \
        f"Past query should prefer old memory: {score_old:.3f} vs {score_new:.3f}"
    print("✓ test_scoring_past_query")


def test_scoring_present_query():
    """
    Present query: most recent memory scores highest (default behavior).
    """
    tr = TemporalReasoner()
    now = int(time.time() * 1000)
    DAY = 86400 * 1000
    
    ctx = tr.analyze('我现在住在哪里', reference_date=now)
    
    score_new = tr.compute_temporal_score(now - 1 * DAY, ctx)
    score_old = tr.compute_temporal_score(now - 100 * DAY, ctx)
    
    assert score_new > score_old, "Present query should prefer recent memory"
    print(f"  Present query: new={score_new:.3f} > old={score_old:.3f}")
    print("✓ test_scoring_present_query")


def test_scoring_future_query():
    """
    Future query: memories about future events score higher.
    
    Scenario: "明天有什么安排" (anchor = tomorrow)
    - Memory "明天要面试" (stored today, event_time=tomorrow) → HIGH
    - Memory "昨天的会议" (stored yesterday) → LOW
    """
    tr = TemporalReasoner()
    now = int(time.time() * 1000)
    DAY = 86400 * 1000
    
    ctx = tr.analyze('明天有什么安排', reference_date=now)
    
    # Memory stored today about tomorrow (close to anchor = tomorrow)
    # In practice, the memory's timestamp is when it was stored (today)
    # But its CONTENT mentions tomorrow → the memory timestamp is "today"
    # The anchor is "tomorrow" → distance = 1 day → still good score
    score_tomorrow = tr.compute_temporal_score(now, ctx)  # stored today, anchor is tomorrow
    # Memory from 30 days ago
    score_old = tr.compute_temporal_score(now - 30 * DAY, ctx)
    
    assert score_tomorrow > score_old, "Future query should prefer recent/future memories"
    print(f"  Future query: today's mem={score_tomorrow:.3f} > 30d-old={score_old:.3f}")
    print("✓ test_scoring_future_query")


def test_reference_date_reproducibility():
    """reference_date enables reproducible testing (Mem0 v3 feature)"""
    tr = TemporalReasoner()
    
    # Fixed reference date: 2026-05-14 00:00:00
    fixed_ref = 1778688000000  # approximate
    
    ctx1 = tr.analyze('上周发生了什么', reference_date=fixed_ref)
    ctx2 = tr.analyze('上周发生了什么', reference_date=fixed_ref)
    
    # Same input, same reference → same output (deterministic)
    assert ctx1.primary_anchor.anchor_time_ms == ctx2.primary_anchor.anchor_time_ms
    print("✓ test_reference_date_reproducibility")


def test_key_scenario_location_history():
    """
    KEY SCENARIO: Same data, different temporal views.
    
    Data: "住在北京" (old), "搬到了上海" (recent)
    Query A: "我现在住在哪里" → 上海 scores higher
    Query B: "我以前住在哪里" → 北京 scores higher
    
    This is the "view function" philosophy in action!
    """
    tr = TemporalReasoner()
    now = int(time.time() * 1000)
    DAY = 86400 * 1000
    
    mem_beijing_ts = now - 100 * DAY  # 100 days old
    mem_shanghai_ts = now - 1 * DAY   # 1 day old
    
    # Query A: "现在"
    ctx_now = tr.analyze('我现在住在哪里', reference_date=now)
    score_bj_now = tr.compute_temporal_score(mem_beijing_ts, ctx_now)
    score_sh_now = tr.compute_temporal_score(mem_shanghai_ts, ctx_now)
    
    # Query B: "以前"
    ctx_before = tr.analyze('我以前住在哪里', reference_date=now)
    score_bj_before = tr.compute_temporal_score(mem_beijing_ts, ctx_before)
    score_sh_before = tr.compute_temporal_score(mem_shanghai_ts, ctx_before)
    
    print(f"  Query '现在住哪': 北京={score_bj_now:.3f}, 上海={score_sh_now:.3f}")
    print(f"  Query '以前住哪': 北京={score_bj_before:.3f}, 上海={score_sh_before:.3f}")
    
    # "现在" → 上海 wins (recency)
    assert score_sh_now > score_bj_now, "Present query: 上海 should win"
    # "以前" → 北京 wins (closer to past anchor)
    assert score_bj_before > score_sh_before, "Past query: 北京 should win"
    
    print("  ✓ Same data, different views — philosophy validated!")
    print("✓ test_key_scenario_location_history")


def test_multi_anchor_selection():
    """When multiple temporal words exist, highest confidence wins"""
    tr = TemporalReasoner()
    
    # "最近" (confidence 0.9) + "以前" (confidence 0.8)
    ctx = tr.analyze('最近以前的事情还记得吗')
    # "最近" has higher confidence (0.9 vs 0.8)
    assert ctx.primary_anchor.matched_text == '最近'
    print("✓ test_multi_anchor_selection")


if __name__ == '__main__':
    tests = [
        test_no_temporal_intent,
        test_past_vague,
        test_past_relative,
        test_past_dynamic,
        test_present,
        test_recent,
        test_future,
        test_scoring_past_query,
        test_scoring_present_query,
        test_scoring_future_query,
        test_reference_date_reproducibility,
        test_key_scenario_location_history,
        test_multi_anchor_selection,
    ]
    
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed, {passed+failed} total")
    if failed == 0:
        print("ALL TESTS PASSED ✓")
