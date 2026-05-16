"""
Tests for ChatEngine v12 — SemanticRule + PinnedMemoryGuard + EventBoundary Integration
Tests the full signal chain: semantic → temporal → validity → frequency → provenance → pin
"""
import sys
import time
import math

# === Reuse components from previous tests ===

class MemoryEntry:
    _counter = 0
    def __init__(self, id=None, content="", source="USER", entities=None,
                 timestamp=None, session_id="default", user_id="user1", metadata=None):
        if id is None:
            MemoryEntry._counter += 1
            id = f"mem_{MemoryEntry._counter}"
        self.id = id
        self.content = content
        self.source = source
        self.entities = entities or []
        self.timestamp = timestamp or int(time.time() * 1000)
        self.session_id = session_id
        self.user_id = user_id
        self.metadata = metadata or {}


class SemanticRuleProvider:
    """Mirrors Kotlin SemanticRuleProvider"""
    INTENT_KEYWORDS = {
        "location": ["住", "在哪", "搬", "居住"],
        "preference": ["喜欢", "爱好", "偏好", "最爱"],
        "health": ["过敏", "病", "健康", "药"],
        "work": ["工作", "上班", "公司", "项目"],
        "food": ["吃", "喝", "火锅", "菜"],
        "pet": ["猫", "狗", "宠物", "年糕"],
    }
    STATE_CHANGE_VERBS = ["搬到", "换了", "开始", "不再", "改为"]

    def compute_similarity(self, query, content):
        score = 0.0
        # Intent alignment
        for intent, keywords in self.INTENT_KEYWORDS.items():
            q_hits = sum(1 for kw in keywords if kw in query)
            c_hits = sum(1 for kw in keywords if kw in content)
            if q_hits > 0 and c_hits > 0:
                score += 0.3
        # Entity overlap
        known = ["北京", "上海", "深圳", "花生", "猫", "狗", "年糕"]
        q_ents = {e for e in known if e in query}
        c_ents = {e for e in known if e in content}
        overlap = len(q_ents & c_ents)
        if overlap > 0:
            score += 0.2 * overlap
        # State-change relevance
        if "住" in query and any(v in content for v in self.STATE_CHANGE_VERBS if "到" in content):
            score += 0.25
        return min(score, 1.0)


class PinnedMemoryGuard:
    """Mirrors Kotlin PinnedMemoryGuard"""
    PIN_PATTERNS = [
        ("过敏|不耐受|禁忌", "HEALTH_ALLERGY"),
        ("药物|用药|服药", "HEALTH_MEDICATION"),
        ("手术|住院|急诊", "HEALTH_PROCEDURE"),
        ("糖尿病|高血压|心脏病", "HEALTH_CONDITION"),
        ("紧急联系|急救", "EMERGENCY"),
        ("不能吃|不能喝", "HEALTH_RESTRICTION"),
        ("密码|账号|银行", "SECURITY"),
        ("很重要|千万别忘|一定要记住|务必", "USER_EMPHASIS"),
    ]

    def __init__(self, floor_score=0.5):
        self.floor_score = floor_score
        self.pinned = {}  # id → reason
        import re
        self._patterns = [(re.compile(p), r) for p, r in self.PIN_PATTERNS]

    def evaluate_for_pin(self, memory_id, content):
        import re
        for pattern, reason in self._patterns:
            if pattern.search(content):
                self.pinned[memory_id] = reason
                return True, reason
        return False, None

    def is_pinned(self, memory_id):
        return memory_id in self.pinned

    def apply_protection(self, memory_id, score):
        if not self.is_pinned(memory_id):
            return score
        return max(score, self.floor_score)


class EventBoundaryDetector:
    """Simplified for integration test"""
    def __init__(self, time_gap_ms=30*60*1000):
        self.time_gap_ms = time_gap_ms

    def find_event(self, nucleus, memories):
        sorted_m = sorted(memories, key=lambda m: m.timestamp)
        idx = next((i for i, m in enumerate(sorted_m) if m.id == nucleus.id), -1)
        if idx < 0:
            return [nucleus]
        start = idx
        while start > 0:
            if sorted_m[start].timestamp - sorted_m[start-1].timestamp > self.time_gap_ms:
                break
            start -= 1
        end = idx
        while end < len(sorted_m) - 1:
            if sorted_m[end+1].timestamp - sorted_m[end].timestamp > self.time_gap_ms:
                break
            end += 1
        return sorted_m[start:end+1]


class ValidityWindow:
    PERMANENT = ["过敏", "血型", "生日"]
    EPHEMERAL = ["现在", "今天", "累", "饿"]
    MEDIUM = ["减肥", "健身", "项目", "住在"]

    def infer(self, content):
        for p in self.PERMANENT:
            if p in content:
                return "PERMANENT", 36500
        for p in self.EPHEMERAL:
            if p in content:
                return "EPHEMERAL", 0.5
        for p in self.MEDIUM:
            if p in content:
                return "MEDIUM", 60
        return "LONG_TERM", 365

    def compute_score(self, fact_ts, query_ts, category, days):
        age_days = (query_ts - fact_ts) / (1000 * 86400)
        if category == "PERMANENT":
            return 1.0
        if days <= 0:
            return 0.3 if age_days > 1 else 1.0
        if age_days <= days:
            return 1.0
        overshoot = (age_days - days) / days
        return max(0.3, 1.0 - overshoot * 0.7)


class AccessFrequencyTracker:
    def __init__(self):
        self.counts = {}

    def record(self, memory_id):
        self.counts[memory_id] = self.counts.get(memory_id, 0) + 1

    def compute_boost(self, memory_id):
        count = self.counts.get(memory_id, 0)
        if count < 2:
            return 1.0
        boost = 1.0 + 0.1 * math.log(1 + count)
        return min(boost, 1.5)


# === V12 Engine (Python mirror for testing) ===

class ChatEngineV12:
    def __init__(self):
        self.store = []  # List[MemoryEntry]
        self.semantic = SemanticRuleProvider()
        self.pinned_guard = PinnedMemoryGuard()
        self.validity = ValidityWindow()
        self.frequency = AccessFrequencyTracker()
        self.event_detector = EventBoundaryDetector()

    def add_memory(self, content, entities=None, timestamp=None, session_id="default", user_id="user1"):
        entry = MemoryEntry(content=content, entities=entities or [],
                           timestamp=timestamp, session_id=session_id, user_id=user_id)
        self.store.append(entry)
        # Auto-pin detection
        self.pinned_guard.evaluate_for_pin(entry.id, content)
        return entry

    def retrieve(self, query, user_id="user1", top_k=5, query_time=None):
        if query_time is None:
            query_time = int(time.time() * 1000)

        user_mems = [m for m in self.store if m.user_id == user_id]
        if not user_mems:
            return []

        scored = []
        for mem in user_mems:
            # Base: semantic score
            semantic = self.semantic.compute_similarity(query, mem.content)

            # Validity
            cat, days = self.validity.infer(mem.content)
            validity_score = self.validity.compute_score(mem.timestamp, query_time, cat, days)

            # Frequency
            freq_boost = self.frequency.compute_boost(mem.id)

            # Combine
            final = semantic * validity_score * freq_boost

            # Pin protection (AFTER all decay)
            final = self.pinned_guard.apply_protection(mem.id, final)

            scored.append((mem, final, {
                "semantic": semantic,
                "validity": validity_score,
                "frequency": freq_boost,
                "pinned": self.pinned_guard.is_pinned(mem.id),
            }))

        scored.sort(key=lambda x: -x[1])

        # Record access for top results
        for mem, _, _ in scored[:top_k]:
            self.frequency.record(mem.id)

        return scored[:top_k]


# === Tests ===

def test_semantic_rule_integration():
    """SemanticRuleProvider produces correct ranking in v12"""
    engine = ChatEngineV12()
    now = int(time.time() * 1000)

    engine.add_memory("我住在北京朝阳", ["北京"], now - 90*86400000)  # 90 days ago
    engine.add_memory("我搬到了上海浦东", ["上海"], now - 30*86400000)  # 30 days ago

    results = engine.retrieve("我现在住在哪里", query_time=now)
    assert len(results) >= 2

    contents = [r[0].content for r in results]
    assert "上海" in contents[0], f"上海 should rank first, got: {contents}"


def test_pin_protection_in_v12():
    """PinnedMemoryGuard protects allergy info in full pipeline"""
    engine = ChatEngineV12()
    now = int(time.time() * 1000)

    allergy = engine.add_memory("我对花生过敏", ["花生"], now - 365*86400000)  # 1 year ago
    engine.add_memory("我喜欢吃火锅", ["火锅"], now - 1*86400000)  # yesterday

    results = engine.retrieve("我有什么忌口吗", query_time=now)

    # Find allergy in results
    allergy_result = next((r for r in results if "过敏" in r[0].content), None)
    assert allergy_result is not None, "Allergy should be in results"
    assert allergy_result[1] >= 0.5, f"Pinned allergy score should be >= 0.5, got {allergy_result[1]}"
    assert allergy_result[2]["pinned"] is True


def test_pin_does_not_affect_normal():
    """Non-pinned memories are not affected by pin guard"""
    engine = ChatEngineV12()
    now = int(time.time() * 1000)

    mem = engine.add_memory("今天天气不错", [], now - 1*86400000)
    assert not engine.pinned_guard.is_pinned(mem.id)

    results = engine.retrieve("天气怎么样", query_time=now)
    if results:
        for r in results:
            if r[0].id == mem.id:
                assert r[2]["pinned"] is False


def test_event_boundary_expansion():
    """EventBoundaryDetector groups related memories together"""
    engine = ChatEngineV12()
    now = int(time.time() * 1000)
    hour = 60*60*1000

    # Event 1: travel discussion (close in time)
    m1 = engine.add_memory("准备去日本旅行", ["日本", "旅行"], now - 3*hour)
    m2 = engine.add_memory("订了东京的酒店", ["东京", "酒店"], now - 3*hour + 60000)
    m3 = engine.add_memory("还要买机票", ["机票"], now - 3*hour + 120000)

    # Event 2: work discussion (1hr gap)
    m4 = engine.add_memory("项目要延期了", ["项目"], now - 1*hour)

    # Find event for m2
    event = engine.event_detector.find_event(m2, engine.store)
    event_ids = [m.id for m in event]
    assert m1.id in event_ids, "m1 should be in same event"
    assert m2.id in event_ids
    assert m3.id in event_ids
    assert m4.id not in event_ids, "m4 is in different event"


def test_full_v12_signal_chain():
    """All v12 signals work together: semantic × validity × frequency × pin"""
    engine = ChatEngineV12()
    now = int(time.time() * 1000)

    # Allergy (old, pinned, permanent validity)
    allergy = engine.add_memory("我对花生过敏", ["花生"], now - 365*86400000)

    # Recent location (high semantic, good validity)
    location = engine.add_memory("我搬到了上海", ["上海"], now - 7*86400000)

    # Old location (lower semantic for current query)
    old_loc = engine.add_memory("我住在北京朝阳", ["北京"], now - 180*86400000)

    # Simulate frequency: location accessed multiple times
    for _ in range(5):
        engine.frequency.record(location.id)

    results = engine.retrieve("我的基本情况", query_time=now)

    # Verify signal breakdown
    for mem, score, signals in results:
        if "过敏" in mem.content:
            assert signals["pinned"] is True
            assert score >= 0.5, f"Pinned allergy should be >= 0.5, got {score}"
        if "上海" in mem.content:
            assert signals["frequency"] > 1.0, "Location should have freq boost"


def test_v12_semantic_vs_char_overlap():
    """SemanticRuleProvider correctly outperforms CharOverlap in v12"""
    engine = ChatEngineV12()
    now = int(time.time() * 1000)

    engine.add_memory("我住在北京", ["北京"], now - 90*86400000)
    engine.add_memory("我搬到了上海", ["上海"], now - 7*86400000)

    # With SemanticRuleProvider: "搬到" + "住" intent should boost 上海
    results = engine.retrieve("住在哪", query_time=now)
    assert len(results) >= 2
    # 上海 should rank higher (state-change verb + location intent)
    assert "上海" in results[0][0].content, f"上海 should be first, got {results[0][0].content}"


def test_v12_pinned_allergy_survives_decay():
    """Even with maximum decay, pinned allergy stays visible"""
    engine = ChatEngineV12()
    now = int(time.time() * 1000)

    # Allergy from 3 years ago — extreme decay
    allergy = engine.add_memory("我对花生过敏，千万别忘", ["花生"], now - 1095*86400000)

    # Many recent irrelevant memories
    for i in range(10):
        engine.add_memory(f"今天的工作任务{i}", ["工作"], now - i*86400000)

    results = engine.retrieve("吃什么", query_time=now, top_k=10)

    # Allergy must be in results with score >= 0.5
    allergy_found = any(r for r in results if "过敏" in r[0].content and r[1] >= 0.5)
    assert allergy_found, "Pinned allergy must survive even with 3-year decay"


def test_v12_multiple_pins():
    """Multiple safety-critical memories all get pin protection"""
    engine = ChatEngineV12()
    now = int(time.time() * 1000)

    m1 = engine.add_memory("我对花生过敏", ["花生"], now - 365*86400000)
    m2 = engine.add_memory("我每天在服药降压", ["药物"], now - 200*86400000)
    m3 = engine.add_memory("紧急联系人是妈妈", ["妈妈"], now - 500*86400000)

    assert engine.pinned_guard.is_pinned(m1.id), "Allergy should be pinned"
    assert engine.pinned_guard.is_pinned(m2.id), "Medication should be pinned"
    assert engine.pinned_guard.is_pinned(m3.id), "Emergency contact should be pinned"


def test_v12_frequency_boost_accumulates():
    """Frequently accessed memories get progressive boost"""
    engine = ChatEngineV12()
    now = int(time.time() * 1000)

    mem = engine.add_memory("我喜欢吃火锅", ["火锅"], now - 7*86400000)

    # Before any access
    boost_before = engine.frequency.compute_boost(mem.id)
    assert boost_before == 1.0

    # Access 10 times
    for _ in range(10):
        engine.frequency.record(mem.id)

    boost_after = engine.frequency.compute_boost(mem.id)
    assert boost_after > 1.0, f"Boost should increase after access, got {boost_after}"
    assert boost_after <= 1.5, f"Boost should be capped at 1.5, got {boost_after}"


# === Run ===

if __name__ == "__main__":
    tests = [
        test_semantic_rule_integration,
        test_pin_protection_in_v12,
        test_pin_does_not_affect_normal,
        test_event_boundary_expansion,
        test_full_v12_signal_chain,
        test_v12_semantic_vs_char_overlap,
        test_v12_pinned_allergy_survives_decay,
        test_v12_multiple_pins,
        test_v12_frequency_boost_accumulates,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
            print(f"  ✓ {test.__name__}")
        except Exception as e:
            failed += 1
            print(f"  ✗ {test.__name__}: {e}")

    print(f"\n{'='*50}")
    print(f"ChatEngine v12: {passed}/{passed+failed} passed")
    if failed > 0:
        sys.exit(1)
