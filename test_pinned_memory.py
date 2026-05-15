"""
Test: PinnedMemoryGuard — Safety-critical memory exemption
验证安全关键记忆的 pin 保护机制
"""

import math
import sys
import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional, Set

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

class PinReason(Enum):
    NOT_PINNED = "NOT_PINNED"
    HEALTH_ALLERGY = "HEALTH_ALLERGY"
    HEALTH_MEDICATION = "HEALTH_MEDICATION"
    HEALTH_PROCEDURE = "HEALTH_PROCEDURE"
    HEALTH_CONDITION = "HEALTH_CONDITION"
    HEALTH_RESTRICTION = "HEALTH_RESTRICTION"
    EMERGENCY_CONTACT = "EMERGENCY_CONTACT"
    SENSITIVE_CREDENTIAL = "SENSITIVE_CREDENTIAL"
    USER_EMPHASIS = "USER_EMPHASIS"
    USER_EXPLICIT = "USER_EXPLICIT"

class PinCategory(Enum):
    NONE = "NONE"
    HEALTH = "HEALTH"
    SAFETY = "SAFETY"
    SECURITY = "SECURITY"
    USER_MARKED = "USER_MARKED"

@dataclass
class PinPattern:
    regex: object
    reason: PinReason
    category: PinCategory

@dataclass
class PinDecision:
    pinned: bool
    reason: PinReason
    category: PinCategory
    explanation: str

DEFAULT_PIN_PATTERNS = [
    PinPattern(re.compile(r'过敏|不耐受|禁忌'), PinReason.HEALTH_ALLERGY, PinCategory.HEALTH),
    PinPattern(re.compile(r'药物|用药|服药|处方'), PinReason.HEALTH_MEDICATION, PinCategory.HEALTH),
    PinPattern(re.compile(r'手术|住院|急诊'), PinReason.HEALTH_PROCEDURE, PinCategory.HEALTH),
    PinPattern(re.compile(r'糖尿病|高血压|心脏病|哮喘'), PinReason.HEALTH_CONDITION, PinCategory.HEALTH),
    PinPattern(re.compile(r'紧急联系|急救|SOS'), PinReason.EMERGENCY_CONTACT, PinCategory.SAFETY),
    PinPattern(re.compile(r'不能吃|不能喝|不能碰'), PinReason.HEALTH_RESTRICTION, PinCategory.HEALTH),
    PinPattern(re.compile(r'密码|账号|银行'), PinReason.SENSITIVE_CREDENTIAL, PinCategory.SECURITY),
    PinPattern(re.compile(r'很重要|千万别忘|一定要记住|务必'), PinReason.USER_EMPHASIS, PinCategory.USER_MARKED),
]

class PinnedMemoryGuard:
    def __init__(self, floor_score=0.5, patterns=None):
        self.floor_score = floor_score
        self.patterns = patterns or DEFAULT_PIN_PATTERNS
        self.pinned_ids = set()
        self.auto_pin_cache = {}
    
    def evaluate_for_pin(self, memory_id, content, entities=None):
        for p in self.patterns:
            if p.regex.search(content):
                self.auto_pin_cache[memory_id] = p.reason
                return PinDecision(True, p.reason, p.category,
                    f"Auto-pinned: matched '{p.regex.pattern}'")
        return PinDecision(False, PinReason.NOT_PINNED, PinCategory.NONE, "No match")
    
    def manual_pin(self, memory_id, reason=PinReason.USER_EXPLICIT):
        self.pinned_ids.add(memory_id)
        self.auto_pin_cache[memory_id] = reason
    
    def unpin(self, memory_id):
        self.pinned_ids.discard(memory_id)
        self.auto_pin_cache.pop(memory_id, None)
    
    def is_pinned(self, memory_id):
        return memory_id in self.pinned_ids or memory_id in self.auto_pin_cache
    
    def get_pin_reason(self, memory_id):
        return self.auto_pin_cache.get(memory_id)
    
    def apply_pin_protection(self, memory_id, current_score):
        if not self.is_pinned(memory_id):
            return current_score
        return max(current_score, self.floor_score)
    
    def get_stats(self):
        return {
            "total": len(self.auto_pin_cache),
            "manual": len(self.pinned_ids),
            "auto": len(self.auto_pin_cache) - len(self.pinned_ids)
        }


# === Tests ===

def test_auto_pin_allergy():
    """过敏信息自动 pin"""
    guard = PinnedMemoryGuard()
    decision = guard.evaluate_for_pin("m1", "我对花生过敏")
    assert decision.pinned == True
    assert decision.reason == PinReason.HEALTH_ALLERGY
    assert decision.category == PinCategory.HEALTH
    assert guard.is_pinned("m1")

def test_auto_pin_medication():
    """用药信息自动 pin"""
    guard = PinnedMemoryGuard()
    decision = guard.evaluate_for_pin("m2", "我每天要服药降压")
    assert decision.pinned == True
    assert decision.reason == PinReason.HEALTH_MEDICATION

def test_auto_pin_restriction():
    """饮食限制自动 pin"""
    guard = PinnedMemoryGuard()
    decision = guard.evaluate_for_pin("m3", "我不能吃海鲜")
    assert decision.pinned == True
    assert decision.reason == PinReason.HEALTH_RESTRICTION

def test_auto_pin_user_emphasis():
    """用户强调自动 pin"""
    guard = PinnedMemoryGuard()
    decision = guard.evaluate_for_pin("m4", "这个很重要：下周三开会")
    assert decision.pinned == True
    assert decision.reason == PinReason.USER_EMPHASIS
    assert decision.category == PinCategory.USER_MARKED

def test_no_pin_normal():
    """普通记忆不 pin"""
    guard = PinnedMemoryGuard()
    decision = guard.evaluate_for_pin("m5", "我喜欢看科幻电影")
    assert decision.pinned == False
    assert not guard.is_pinned("m5")

def test_manual_pin():
    """手动 pin"""
    guard = PinnedMemoryGuard()
    guard.manual_pin("m6")
    assert guard.is_pinned("m6")
    assert guard.get_pin_reason("m6") == PinReason.USER_EXPLICIT

def test_unpin():
    """解除 pin"""
    guard = PinnedMemoryGuard()
    guard.evaluate_for_pin("m1", "我对花生过敏")
    assert guard.is_pinned("m1")
    guard.unpin("m1")
    assert not guard.is_pinned("m1")

def test_floor_score_protection():
    """Pin 保护：分数不低于 floor"""
    guard = PinnedMemoryGuard(floor_score=0.5)
    guard.evaluate_for_pin("m1", "我对花生过敏")
    
    # Very low score (e.g., old + rarely accessed + low semantic match)
    protected = guard.apply_pin_protection("m1", 0.1)
    assert protected == 0.5, f"Should be floor 0.5, got {protected}"
    
    # Score already above floor → unchanged
    high = guard.apply_pin_protection("m1", 0.8)
    assert high == 0.8, f"Above floor should be unchanged, got {high}"

def test_non_pinned_unaffected():
    """非 pin 记忆不受影响"""
    guard = PinnedMemoryGuard(floor_score=0.5)
    score = guard.apply_pin_protection("unpinned", 0.1)
    assert score == 0.1, f"Non-pinned should be unchanged, got {score}"

def test_pin_prevents_decay_death():
    """核心场景：pin 防止安全关键信息被衰减到不可见"""
    guard = PinnedMemoryGuard(floor_score=0.5)
    
    # Simulate: allergy memory, very old, never accessed
    guard.evaluate_for_pin("allergy", "我对花生严重过敏")
    
    # Simulate decay signals
    base_score = 0.3       # low semantic match for generic query
    validity_score = 1.0   # permanent
    frequency_boost = 1.0  # never accessed (neutral)
    provenance_signal = 0.7 # moderate confidence
    
    decayed_score = base_score * validity_score * frequency_boost * provenance_signal
    # Without pin: 0.3 * 1.0 * 1.0 * 0.7 = 0.21 → might be below top-K threshold
    
    protected = guard.apply_pin_protection("allergy", decayed_score)
    assert protected == 0.5, f"Pin should protect to floor, got {protected}"
    assert protected > decayed_score, "Pin should raise the score"
    print(f"  Decayed: {decayed_score:.3f} → Protected: {protected:.3f}")

def test_pin_does_not_override_high_scores():
    """Pin 不会降低已经很高的分数"""
    guard = PinnedMemoryGuard(floor_score=0.5)
    guard.evaluate_for_pin("m1", "我对花生过敏")
    
    # High relevance query about allergies → high base score
    result = guard.apply_pin_protection("m1", 0.95)
    assert result == 0.95

def test_multiple_pins():
    """多个 pin 独立工作"""
    guard = PinnedMemoryGuard()
    guard.evaluate_for_pin("a", "我对花生过敏")
    guard.evaluate_for_pin("b", "我每天服药")
    guard.evaluate_for_pin("c", "我喜欢电影")  # not pinned
    
    assert guard.is_pinned("a")
    assert guard.is_pinned("b")
    assert not guard.is_pinned("c")
    
    stats = guard.get_stats()
    assert stats["total"] == 2

def test_security_pin():
    """安全敏感信息 pin"""
    guard = PinnedMemoryGuard()
    decision = guard.evaluate_for_pin("s1", "我的银行卡尾号是1234")
    assert decision.pinned == True
    assert decision.category == PinCategory.SECURITY

def test_integration_with_v11_pipeline():
    """模拟与 v11 pipeline 集成"""
    guard = PinnedMemoryGuard(floor_score=0.5)
    
    # Ingestion: pin evaluation
    memories = [
        ("m1", "我对花生过敏", 0.15),    # safety-critical, very low score
        ("m2", "我住在上海", 0.7),         # normal, decent score
        ("m3", "我不能吃海鲜", 0.12),     # safety-critical, very low score
        ("m4", "今天天气不错", 0.05),      # noise
    ]
    
    for mid, content, _ in memories:
        guard.evaluate_for_pin(mid, content)
    
    # Retrieval: apply pin protection
    results = []
    for mid, content, score in memories:
        protected_score = guard.apply_pin_protection(mid, score)
        results.append((mid, content, score, protected_score))
    
    # Sort by protected score
    results.sort(key=lambda x: x[3], reverse=True)
    
    # Safety-critical memories should NOT be at the bottom
    top_ids = [r[0] for r in results[:3]]
    assert "m1" in top_ids, f"Allergy should be in top 3, got {top_ids}"
    assert "m3" in top_ids, f"Seafood restriction should be in top 3, got {top_ids}"
    
    # Noise should still be at bottom
    assert results[-1][0] == "m4", "Noise should be last"
    
    print(f"  Ranking after pin protection:")
    for mid, content, orig, prot in results:
        pin_mark = "📌" if guard.is_pinned(mid) else "  "
        print(f"    {pin_mark} {content}: {orig:.2f} → {prot:.2f}")


# === Run ===
print("=" * 60)
print("PinnedMemoryGuard Tests — Safety-Critical Exemption")
print("=" * 60)

tests = [
    ("auto_pin_allergy", test_auto_pin_allergy),
    ("auto_pin_medication", test_auto_pin_medication),
    ("auto_pin_restriction", test_auto_pin_restriction),
    ("auto_pin_user_emphasis", test_auto_pin_user_emphasis),
    ("no_pin_normal", test_no_pin_normal),
    ("manual_pin", test_manual_pin),
    ("unpin", test_unpin),
    ("floor_score_protection", test_floor_score_protection),
    ("non_pinned_unaffected", test_non_pinned_unaffected),
    ("pin_prevents_decay_death", test_pin_prevents_decay_death),
    ("pin_does_not_override_high", test_pin_does_not_override_high_scores),
    ("multiple_pins", test_multiple_pins),
    ("security_pin", test_security_pin),
    ("integration_v11_pipeline", test_integration_with_v11_pipeline),
]

for name, fn in tests:
    run_test(name, fn)

print()
print("=" * 50)
print(f"Results: {TESTS_PASSED} passed, {TESTS_FAILED} failed, {TESTS_PASSED + TESTS_FAILED} total")
if TESTS_FAILED == 0:
    print("ALL TESTS PASSED ✓")
else:
    print(f"FAILURES: {TESTS_FAILED}")
    sys.exit(1)
