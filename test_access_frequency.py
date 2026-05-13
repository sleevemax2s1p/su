"""
Test: AccessFrequencyTracker (Mem0 Memory Decay inspired)
验证访问频率追踪和 boost 计算
"""

import math
import sys

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


class AccessFrequencyTracker:
    def __init__(self, log_factor=0.1, max_boost=1.5, min_hits_for_boost=1):
        self.log_factor = log_factor
        self.max_boost = max_boost
        self.min_hits_for_boost = min_hits_for_boost
        self.access_counts = {}
    
    def record_access(self, memory_id):
        self.access_counts[memory_id] = self.access_counts.get(memory_id, 0) + 1
    
    def record_access_batch(self, memory_ids):
        for mid in memory_ids:
            self.record_access(mid)
    
    def compute_boost(self, memory_id):
        count = self.access_counts.get(memory_id, 0)
        if count < self.min_hits_for_boost:
            return 1.0
        boost = 1.0 + self.log_factor * math.log(1 + count)
        return min(self.max_boost, max(1.0, boost))
    
    def get_access_count(self, memory_id):
        return self.access_counts.get(memory_id, 0)
    
    def get_stats(self):
        if not self.access_counts:
            return {"tracked": 0, "total": 0, "avg": 0, "max": 0}
        counts = list(self.access_counts.values())
        return {
            "tracked": len(self.access_counts),
            "total": sum(counts),
            "avg": sum(counts)/len(counts),
            "max": max(counts)
        }
    
    def reset(self):
        self.access_counts.clear()
    
    def export_state(self):
        return dict(self.access_counts)
    
    def import_state(self, data):
        self.access_counts = dict(data)


# === Tests ===

def test_zero_access_neutral():
    """未被访问的记忆 boost=1.0 (neutral)"""
    tracker = AccessFrequencyTracker()
    assert tracker.compute_boost("mem1") == 1.0

def test_single_access_small_boost():
    """单次访问 → 小幅 boost"""
    tracker = AccessFrequencyTracker()
    tracker.record_access("mem1")
    boost = tracker.compute_boost("mem1")
    assert 1.05 < boost < 1.15, f"Single access boost should be small, got {boost:.3f}"
    print(f"  1 access → boost={boost:.3f}")

def test_logarithmic_growth():
    """boost 随访问次数对数增长，不是线性"""
    tracker = AccessFrequencyTracker()
    boosts = []
    for i in range(1, 101):
        tracker.record_access("mem1")
        boosts.append(tracker.compute_boost("mem1"))
    
    # Check diminishing returns
    diff_1_to_10 = boosts[9] - boosts[0]
    diff_90_to_100 = boosts[99] - boosts[89]
    assert diff_1_to_10 > diff_90_to_100, \
        f"Growth should diminish: first 10={diff_1_to_10:.4f} > last 10={diff_90_to_100:.4f}"
    print(f"  1→10 gain={diff_1_to_10:.4f}, 90→100 gain={diff_90_to_100:.4f}")

def test_max_boost_cap():
    """boost 有上限，不会无限增长"""
    tracker = AccessFrequencyTracker(max_boost=1.5)
    for _ in range(10000):
        tracker.record_access("mem1")
    boost = tracker.compute_boost("mem1")
    assert boost <= 1.5, f"Boost should be capped at 1.5, got {boost:.3f}"

def test_min_hits_threshold():
    """低于 min_hits 不 boost"""
    tracker = AccessFrequencyTracker(min_hits_for_boost=3)
    tracker.record_access("mem1")  # count=1
    assert tracker.compute_boost("mem1") == 1.0, "Below threshold should be 1.0"
    tracker.record_access("mem1")  # count=2
    assert tracker.compute_boost("mem1") == 1.0, "Still below threshold"
    tracker.record_access("mem1")  # count=3
    assert tracker.compute_boost("mem1") > 1.0, "At threshold should boost"

def test_batch_record():
    """批量记录"""
    tracker = AccessFrequencyTracker()
    tracker.record_access_batch(["a", "b", "c", "a", "a"])
    assert tracker.get_access_count("a") == 3
    assert tracker.get_access_count("b") == 1
    assert tracker.get_access_count("c") == 1

def test_independent_tracking():
    """不同记忆独立追踪"""
    tracker = AccessFrequencyTracker()
    for _ in range(10):
        tracker.record_access("hot")
    tracker.record_access("cold")
    
    hot_boost = tracker.compute_boost("hot")
    cold_boost = tracker.compute_boost("cold")
    assert hot_boost > cold_boost, f"hot={hot_boost:.3f} should > cold={cold_boost:.3f}"

def test_stats():
    """统计信息"""
    tracker = AccessFrequencyTracker()
    tracker.record_access_batch(["a", "a", "a", "b", "b", "c"])
    stats = tracker.get_stats()
    assert stats["tracked"] == 3
    assert stats["total"] == 6
    assert stats["max"] == 3

def test_export_import():
    """导出/导入状态"""
    tracker1 = AccessFrequencyTracker()
    tracker1.record_access_batch(["a", "a", "b"])
    state = tracker1.export_state()
    
    tracker2 = AccessFrequencyTracker()
    tracker2.import_state(state)
    assert tracker2.get_access_count("a") == 2
    assert tracker2.get_access_count("b") == 1
    assert tracker2.compute_boost("a") == tracker1.compute_boost("a")

def test_reset():
    """重置"""
    tracker = AccessFrequencyTracker()
    tracker.record_access("mem1")
    tracker.reset()
    assert tracker.get_access_count("mem1") == 0
    assert tracker.compute_boost("mem1") == 1.0

def test_concrete_boost_values():
    """验证具体 boost 数值（文档中列出的）"""
    tracker = AccessFrequencyTracker(log_factor=0.1)
    
    expected = [
        (0, 1.0),
        (1, 1.0 + 0.1 * math.log(2)),     # ~1.069
        (5, 1.0 + 0.1 * math.log(6)),      # ~1.179
        (20, 1.0 + 0.1 * math.log(21)),    # ~1.304
        (100, 1.0 + 0.1 * math.log(101)),  # ~1.461
    ]
    
    for count, expected_boost in expected:
        t = AccessFrequencyTracker(log_factor=0.1)
        for _ in range(count):
            t.record_access("x")
        actual = t.compute_boost("x")
        assert abs(actual - expected_boost) < 0.01, \
            f"count={count}: expected {expected_boost:.3f}, got {actual:.3f}"
    print(f"  Boost curve: 0→1.000, 1→1.069, 5→1.179, 20→1.304, 100→1.461")

def test_no_matthew_effect():
    """验证不会出现马太效应：高频记忆 boost 有限，不压制新记忆"""
    tracker = AccessFrequencyTracker(log_factor=0.1, max_boost=1.5)
    
    # "Hot" memory accessed 1000 times
    for _ in range(1000):
        tracker.record_access("hot")
    
    # "New" memory accessed once
    tracker.record_access("new")
    
    hot_boost = tracker.compute_boost("hot")
    new_boost = tracker.compute_boost("new")
    
    # Hot should NOT be more than 50% better than new
    ratio = hot_boost / new_boost
    assert ratio < 1.5, f"Hot/new ratio={ratio:.2f}, should be <1.5 (no matthew effect)"
    print(f"  hot(1000)={hot_boost:.3f}, new(1)={new_boost:.3f}, ratio={ratio:.2f}")


# === Run ===
print("=" * 60)
print("AccessFrequencyTracker Tests")
print("=" * 60)

tests = [
    ("zero_access_neutral", test_zero_access_neutral),
    ("single_access_small_boost", test_single_access_small_boost),
    ("logarithmic_growth", test_logarithmic_growth),
    ("max_boost_cap", test_max_boost_cap),
    ("min_hits_threshold", test_min_hits_threshold),
    ("batch_record", test_batch_record),
    ("independent_tracking", test_independent_tracking),
    ("stats", test_stats),
    ("export_import", test_export_import),
    ("reset", test_reset),
    ("concrete_boost_values", test_concrete_boost_values),
    ("no_matthew_effect", test_no_matthew_effect),
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
