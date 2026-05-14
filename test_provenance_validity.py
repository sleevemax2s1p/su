"""
Tests: ProvenanceTracker + ValidityWindow
验证记忆溯源和事实有效期推断
"""

import math
import sys
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional

TESTS_PASSED = 0
TESTS_FAILED = 0

def run_test(name, fn):
    global TESTS_PASSED, TESTS_FAILED
    try:
        fn()
        print(f"✓ {name}")
        TESTS_PASSED += 1
    except AssertionError as e:
        print(f"✗ {name}: {e}")
        TESTS_FAILED += 1
    except Exception as e:
        print(f"✗ {name}: EXCEPTION {e}")
        TESTS_FAILED += 1


# === Python simulation ===

class AdmissionOutcome(Enum):
    ADMITTED = "ADMITTED"
    FAST_PATH = "FAST_PATH"
    REJECTED = "REJECTED"
    FILTERED = "FILTERED"

@dataclass
class ProvenanceRecord:
    memory_id: str
    turn_id: str
    session_id: str
    user_id: str
    original_message: str
    extraction_confidence: float
    admission_decision: AdmissionOutcome
    extraction_method: str
    created_at: int

class ProvenanceTracker:
    def __init__(self):
        self.records: Dict[str, ProvenanceRecord] = {}
    
    def record(self, memory_id, turn_id, session_id, user_id, original_message,
               extraction_confidence=1.0, admission_decision=AdmissionOutcome.ADMITTED,
               extraction_method="fact_extractor_v1", timestamp=0):
        rec = ProvenanceRecord(
            memory_id=memory_id, turn_id=turn_id, session_id=session_id,
            user_id=user_id, original_message=original_message,
            extraction_confidence=extraction_confidence,
            admission_decision=admission_decision,
            extraction_method=extraction_method, created_at=timestamp
        )
        self.records[memory_id] = rec
        return rec
    
    def get_provenance(self, memory_id):
        return self.records.get(memory_id)
    
    def get_by_turn(self, turn_id):
        return [r for r in self.records.values() if r.turn_id == turn_id]
    
    def get_by_session(self, session_id):
        return [r for r in self.records.values() if r.session_id == session_id]
    
    def get_siblings(self, memory_id):
        rec = self.records.get(memory_id)
        if not rec:
            return []
        return [r for r in self.records.values() if r.turn_id == rec.turn_id and r.memory_id != memory_id]
    
    def get_low_confidence(self, threshold=0.5):
        return [r for r in self.records.values() if r.extraction_confidence < threshold]
    
    def compute_provenance_signal(self, memory_id):
        rec = self.records.get(memory_id)
        if not rec:
            return 1.0
        signal = rec.extraction_confidence
        if rec.admission_decision == AdmissionOutcome.FAST_PATH:
            signal = min(1.0, signal * 1.1)
        return max(0.3, min(1.0, signal))


import re

class ValidityCategory(Enum):
    PERMANENT = "PERMANENT"
    LONG_TERM = "LONG_TERM"
    MEDIUM_TERM = "MEDIUM_TERM"
    SHORT_TERM = "SHORT_TERM"
    EPHEMERAL = "EPHEMERAL"

@dataclass
class ValidityInfo:
    category: ValidityCategory
    estimated_days: float  # -1 = permanent
    confidence: float
    reason: str

class ValidityWindow:
    PERMANENT_PATTERNS = [
        (re.compile(r'过敏|不耐受|恐惧症'), ValidityCategory.PERMANENT),
        (re.compile(r'血型|生日|出生|姓名|身份证'), ValidityCategory.PERMANENT),
        (re.compile(r'母语|国籍|民族|宗教'), ValidityCategory.PERMANENT),
        (re.compile(r'一直|永远|从小|天生'), ValidityCategory.PERMANENT),
        (re.compile(r'学历|毕业|学位'), ValidityCategory.PERMANENT),
    ]
    EPHEMERAL_PATTERNS = [
        (re.compile(r'现在|此刻|刚才|刚刚'), 0.5, ValidityCategory.EPHEMERAL),
        (re.compile(r'今天|今晚|今早'), 1.0, ValidityCategory.EPHEMERAL),
        (re.compile(r'明天|后天'), 2.0, ValidityCategory.EPHEMERAL),
        (re.compile(r'这周|本周'), 7.0, ValidityCategory.SHORT_TERM),
        (re.compile(r'心情|感觉|情绪|状态'), 1.0, ValidityCategory.EPHEMERAL),
        (re.compile(r'饿|累|困|渴|疼'), 0.25, ValidityCategory.EPHEMERAL),
    ]
    MEDIUM_PATTERNS = [
        (re.compile(r'减肥|健身|练习|节食'), 60.0, ValidityCategory.MEDIUM_TERM),
        (re.compile(r'在学|正在看|在读'), 90.0, ValidityCategory.MEDIUM_TERM),
        (re.compile(r'项目|工作任务|deadline'), 30.0, ValidityCategory.MEDIUM_TERM),
        (re.compile(r'住在|租房|合租'), 180.0, ValidityCategory.MEDIUM_TERM),
        (re.compile(r'男朋友|女朋友|对象|约会'), 90.0, ValidityCategory.MEDIUM_TERM),
        (re.compile(r'工作|上班|公司'), 365.0, ValidityCategory.LONG_TERM),
    ]
    
    def __init__(self, default_validity_days=365.0, grace_factor=0.3):
        self.default_validity_days = default_validity_days
        self.grace_factor = grace_factor
    
    def infer_validity(self, content, entities=None):
        for pattern, category in self.PERMANENT_PATTERNS:
            if pattern.search(content):
                return ValidityInfo(category, -1.0, 0.9, f"Permanent: {pattern.pattern}")
        for pattern, days, category in self.EPHEMERAL_PATTERNS:
            if pattern.search(content):
                return ValidityInfo(category, days, 0.7, f"Ephemeral: {pattern.pattern}")
        for pattern, days, category in self.MEDIUM_PATTERNS:
            if pattern.search(content):
                return ValidityInfo(category, days, 0.6, f"Medium: {pattern.pattern}")
        return ValidityInfo(ValidityCategory.LONG_TERM, self.default_validity_days, 0.3, "Default")
    
    def compute_validity_score(self, fact_timestamp, query_time, validity_info):
        if validity_info.estimated_days < 0:
            return 1.0
        age_days = (query_time - fact_timestamp) / 86400000.0
        if age_days <= validity_info.estimated_days:
            return 1.0
        overage = age_days - validity_info.estimated_days
        decay = math.exp(-0.1 * overage)
        return self.grace_factor + (1.0 - self.grace_factor) * decay
    
    def is_likely_valid(self, fact_timestamp, query_time, validity_info, threshold=0.5):
        return self.compute_validity_score(fact_timestamp, query_time, validity_info) >= threshold


# === Provenance Tests ===

DAY = 86400000

def test_provenance_basic():
    """基本溯源记录和查询"""
    tracker = ProvenanceTracker()
    tracker.record("mem1", "turn_1", "sess_1", "user_1", "我对花生过敏", 0.95)
    rec = tracker.get_provenance("mem1")
    assert rec is not None
    assert rec.original_message == "我对花生过敏"
    assert rec.extraction_confidence == 0.95

def test_provenance_by_turn():
    """按 turn 查询"""
    tracker = ProvenanceTracker()
    tracker.record("mem1", "turn_1", "sess_1", "u", "我叫小明，住在北京", 0.9)
    tracker.record("mem2", "turn_1", "sess_1", "u", "我叫小明，住在北京", 0.85)
    tracker.record("mem3", "turn_2", "sess_1", "u", "我喜欢猫", 0.8)
    results = tracker.get_by_turn("turn_1")
    assert len(results) == 2

def test_provenance_siblings():
    """同源记忆（同一条消息提取的多个 fact）"""
    tracker = ProvenanceTracker()
    tracker.record("name", "t1", "s1", "u", "我叫小明，在北京工作", 0.9)
    tracker.record("loc", "t1", "s1", "u", "我叫小明，在北京工作", 0.85)
    siblings = tracker.get_siblings("name")
    assert len(siblings) == 1
    assert siblings[0].memory_id == "loc"

def test_provenance_low_confidence():
    """低置信度过滤"""
    tracker = ProvenanceTracker()
    tracker.record("m1", "t1", "s1", "u", "嗯好的吧", 0.3)
    tracker.record("m2", "t2", "s1", "u", "我对花生过敏", 0.95)
    low = tracker.get_low_confidence(0.5)
    assert len(low) == 1
    assert low[0].memory_id == "m1"

def test_provenance_signal_neutral():
    """无溯源 → 中性信号"""
    tracker = ProvenanceTracker()
    assert tracker.compute_provenance_signal("unknown") == 1.0

def test_provenance_signal_fast_path():
    """Fast-path admission → slight boost"""
    tracker = ProvenanceTracker()
    tracker.record("m1", "t1", "s1", "u", "过敏", 0.9, AdmissionOutcome.FAST_PATH)
    signal = tracker.compute_provenance_signal("m1")
    assert signal > 0.9, f"Fast path should boost, got {signal}"

def test_provenance_signal_low_confidence():
    """低置信度 → 降权但不为0"""
    tracker = ProvenanceTracker()
    tracker.record("m1", "t1", "s1", "u", "可能是...", 0.3)
    signal = tracker.compute_provenance_signal("m1")
    assert signal == 0.3, f"Should be clamped to 0.3, got {signal}"


# === Validity Window Tests ===

def test_validity_permanent():
    """过敏 → 永久有效"""
    vw = ValidityWindow()
    info = vw.infer_validity("我对花生过敏")
    assert info.category == ValidityCategory.PERMANENT
    assert info.estimated_days == -1.0

def test_validity_ephemeral():
    """今天的事 → 短期"""
    vw = ValidityWindow()
    info = vw.infer_validity("我今天好累")
    # "今天" matches ephemeral, but "累" also matches — first wins
    assert info.category == ValidityCategory.EPHEMERAL
    assert info.estimated_days <= 1.0

def test_validity_medium():
    """减肥 → 中期"""
    vw = ValidityWindow()
    info = vw.infer_validity("我最近在减肥")
    assert info.category == ValidityCategory.MEDIUM_TERM
    assert 30 <= info.estimated_days <= 90

def test_validity_default():
    """无特征 → 默认长期"""
    vw = ValidityWindow()
    info = vw.infer_validity("我喜欢看科幻电影")
    assert info.category == ValidityCategory.LONG_TERM
    assert info.estimated_days == 365.0

def test_validity_score_within_window():
    """在有效期内 → score=1.0"""
    vw = ValidityWindow()
    info = ValidityInfo(ValidityCategory.MEDIUM_TERM, 60.0, 0.7, "test")
    score = vw.compute_validity_score(0, 30 * DAY, info)  # 30 days old, 60 day window
    assert score == 1.0

def test_validity_score_expired():
    """过期后 → 衰减但不为0"""
    vw = ValidityWindow()
    info = ValidityInfo(ValidityCategory.EPHEMERAL, 1.0, 0.7, "test")
    score = vw.compute_validity_score(0, 30 * DAY, info)  # 30 days past 1-day window
    assert 0.3 <= score < 0.5, f"Should be near grace factor, got {score:.3f}"
    print(f"  Expired 29 days: score={score:.3f}")

def test_validity_score_permanent():
    """永久 → 始终1.0"""
    vw = ValidityWindow()
    info = ValidityInfo(ValidityCategory.PERMANENT, -1.0, 0.9, "test")
    score = vw.compute_validity_score(0, 3650 * DAY, info)  # 10 years later
    assert score == 1.0

def test_validity_gradual_decay():
    """过期后渐进衰减，不是悬崖"""
    vw = ValidityWindow()
    info = ValidityInfo(ValidityCategory.SHORT_TERM, 7.0, 0.7, "test")
    scores = []
    for days in [7, 8, 14, 30, 60]:
        s = vw.compute_validity_score(0, days * DAY, info)
        scores.append(s)
    # Should be monotonically decreasing
    for i in range(len(scores)-1):
        assert scores[i] >= scores[i+1], f"Not monotonic: {scores}"
    # Day 7 = within window = 1.0
    assert scores[0] == 1.0
    # Day 60 should be near grace
    assert scores[-1] < 0.5
    print(f"  Decay: day7={scores[0]:.2f}, day14={scores[2]:.2f}, day60={scores[4]:.2f}")

def test_validity_is_likely_valid():
    """便捷判断"""
    vw = ValidityWindow()
    info = ValidityInfo(ValidityCategory.EPHEMERAL, 1.0, 0.7, "test")
    assert vw.is_likely_valid(0, int(0.5 * DAY), info) == True   # half day: valid
    assert vw.is_likely_valid(0, 60 * DAY, info) == False         # 60 days: expired

def test_validity_real_scenario():
    """真实场景：减肥声明 2 个月后是否仍有效"""
    vw = ValidityWindow()
    info = vw.infer_validity("我最近在减肥")
    # 60-day window for 减肥
    assert info.estimated_days == 60.0
    
    # After 30 days: still valid
    assert vw.is_likely_valid(0, 30 * DAY, info)
    # After 90 days: probably expired
    score_90 = vw.compute_validity_score(0, 90 * DAY, info)
    print(f"  减肥 after 90 days: score={score_90:.3f}, valid={vw.is_likely_valid(0, 90*DAY, info)}")


# === Run ===
print("=" * 60)
print("ProvenanceTracker + ValidityWindow Tests")
print("=" * 60)
print()
print("--- Provenance ---")
tests_prov = [
    ("provenance_basic", test_provenance_basic),
    ("provenance_by_turn", test_provenance_by_turn),
    ("provenance_siblings", test_provenance_siblings),
    ("provenance_low_confidence", test_provenance_low_confidence),
    ("provenance_signal_neutral", test_provenance_signal_neutral),
    ("provenance_signal_fast_path", test_provenance_signal_fast_path),
    ("provenance_signal_low_confidence", test_provenance_signal_low_confidence),
]
for name, fn in tests_prov:
    run_test(name, fn)

print()
print("--- Validity Window ---")
tests_val = [
    ("validity_permanent", test_validity_permanent),
    ("validity_ephemeral", test_validity_ephemeral),
    ("validity_medium", test_validity_medium),
    ("validity_default", test_validity_default),
    ("validity_score_within_window", test_validity_score_within_window),
    ("validity_score_expired", test_validity_score_expired),
    ("validity_score_permanent", test_validity_score_permanent),
    ("validity_gradual_decay", test_validity_gradual_decay),
    ("validity_is_likely_valid", test_validity_is_likely_valid),
    ("validity_real_scenario", test_validity_real_scenario),
]
for name, fn in tests_val:
    run_test(name, fn)

print()
print("=" * 50)
print(f"Results: {TESTS_PASSED} passed, {TESTS_FAILED} failed, {TESTS_PASSED + TESTS_FAILED} total")
if TESTS_FAILED == 0:
    print("ALL TESTS PASSED ✓")
else:
    print(f"FAILURES: {TESTS_FAILED}")
    sys.exit(1)
